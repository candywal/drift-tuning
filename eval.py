# %%

# Configuration
CONFIG = {
    "base_model_path": "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
    "checkpoint_dir": "old_bad_run/output2",  # Change this to evaluate different runs
    "alek_evals_dir": "alek-evals",
    "results_dir": "results",
    "gpus": [0, 1, 2, 3],  # Changed to GPUs 0-3
    "batch_size": 128,  # Increased from 64 - H100s can handle larger batches
    "max_new_tokens": 256,  # Reduced from 512 - still plenty for thinking + answer
    "temperature": 0.0,  # Deterministic generation for evals
    "tensor_parallel_size": 1,  # Use 1 GPU per model for faster loading
    "num_parallel_jobs": 4,  # Run 4 models in parallel (1 GPU each)
    "max_eval_samples": 200,  # Limit each eval to 200 questions
    "gpu_memory_utilization": 0.9,  # Increased from 0.8 for H100s
    "enable_prefix_caching": True,  # Enable prefix caching for faster inference
    "swap_space": 4,  # GB of CPU swap space for large batches
    "use_hotswapping": False,  # Disabled due to result collection issues
    "disable_custom_all_reduce": True,  # Fix for multi-GPU issues
    "enforce_eager": False,  # Can use compilation with legacy engine
}

# NOTE: To use vLLM's legacy engine (V0) for LoRA hotswapping compatibility,
# run with: VLLM_USE_V1=0 python eval.py
# The V1 engine has known issues with torch.compile and LoRA adapters.

import os
import json
import glob
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import torch
import subprocess
from collections import defaultdict
import pandas as pd
import argparse

# Global tokenizer cache to avoid reloading
_tokenizer_cache = {}

def get_all_checkpoints(checkpoint_dir):
    """Get all checkpoints organized by dataset."""
    checkpoints = defaultdict(list)
    
    for dataset_dir in os.listdir(checkpoint_dir):
        dataset_path = os.path.join(checkpoint_dir, dataset_dir)
        if os.path.isdir(dataset_path):
            # Add step 0 (base model evaluation)
            checkpoints[dataset_dir].append({
                "path": CONFIG["base_model_path"],  # Use base model path for step 0
                "step": 0,
                "dataset": dataset_dir
            })
            
            # Find all checkpoint directories
            checkpoint_dirs = glob.glob(os.path.join(dataset_path, "checkpoint-*"))
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))  # Sort by step number
            
            for ckpt_dir in checkpoint_dirs:
                step = int(ckpt_dir.split("-")[-1])
                checkpoints[dataset_dir].append({
                    "path": ckpt_dir,
                    "step": step,
                    "dataset": dataset_dir
                })
    
    return checkpoints

def get_all_evals(evals_dir):
    """Get all evaluation files."""
    eval_files = glob.glob(os.path.join(evals_dir, "*.json"))
    evals = {}
    
    for eval_file in eval_files:
        eval_name = os.path.basename(eval_file).replace(".json", "")
        with open(eval_file, "r") as f:
            evals[eval_name] = json.load(f)
    
    return evals

def format_prompt(question, tokenizer):
    """Format the question for the model with thinking prompt and a single concise example."""
    
    # System prompt that encourages thinking
    system_prompt = """You are a helpful assistant. For multiple-choice questions:
1. Think step-by-step inside <think></think> tags
2. After </think>, output ONLY the answer number (1, 2, 3, or 4)"""
    
    # Single concise few-shot example
    few_shot_example = {
        "question": "What is 2+2?\n1) 3\n2) 4\n3) 5\n4) 6",
        "response": "<think>\n2+2 equals 4, which is option 2.\n</think>\n\n2"
    }
    
    # Build the messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_example["question"]},
        {"role": "assistant", "content": few_shot_example["response"]},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback format
        print("WARNING: Using fallback format for prompt")
        prompt = f"System: {system_prompt}\n\nUser: {few_shot_example['question']}\n\nAssistant: {few_shot_example['response']}\n\nUser: {question}\n\nAssistant:"
    
    return prompt

def evaluate_checkpoint_on_eval(checkpoint_path, base_model_path, eval_name, eval_data, gpu_ids, config):
    """Evaluate a single checkpoint on a single eval."""
    
    # Extract step from checkpoint path
    step = int(checkpoint_path.split("-")[-1]) if "checkpoint-" in checkpoint_path else 0
    
    # Limit eval data to max_eval_samples if specified
    if "max_eval_samples" in config and len(eval_data) > config["max_eval_samples"]:
        eval_data = eval_data[:config["max_eval_samples"]]
        print(f"Limited {eval_name} to {config['max_eval_samples']} samples")
    
    # Set CUDA_VISIBLE_DEVICES for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    # Add CUDA device order to ensure consistent GPU mapping
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Force single GPU usage within vLLM
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first visible device (mapped from gpu_ids)
    
    try:
        # Load tokenizer with caching
        tokenizer_key = base_model_path if step == 0 else checkpoint_path
        if tokenizer_key not in _tokenizer_cache:
            # For step 0, use base model tokenizer; otherwise use checkpoint tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_key, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            _tokenizer_cache[tokenizer_key] = tokenizer
        else:
            tokenizer = _tokenizer_cache[tokenizer_key]
        
        # Initialize vLLM
        if step == 0:
            # For step 0, no LoRA adapter needed
            print(f"Evaluating base model (step 0) without adapter on {eval_name}")
            llm = LLM(
                model=base_model_path,
                tensor_parallel_size=config["tensor_parallel_size"],
                gpu_memory_utilization=config["gpu_memory_utilization"],
                trust_remote_code=True,
                max_model_len=4096,
                enable_prefix_caching=config.get("enable_prefix_caching", True),
                swap_space=config.get("swap_space", 4),
                disable_log_stats=True,  # Reduce logging overhead
            )
            lora_request = None
        else:
            # For other steps, use LoRA adapter
            print(f"Evaluating checkpoint step {step} with LoRA adapter on {eval_name}")
            llm = LLM(
                model=base_model_path,
                enable_lora=True,
                max_lora_rank=32,  # Set based on your LoRA config
                tensor_parallel_size=config["tensor_parallel_size"],
                gpu_memory_utilization=config["gpu_memory_utilization"],
                trust_remote_code=True,
                max_model_len=4096,
                enable_prefix_caching=config.get("enable_prefix_caching", True),
                swap_space=config.get("swap_space", 4),
                disable_log_stats=True,  # Reduce logging overhead
            )
            
            # Create LoRA request
            lora_request = LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_local_path=checkpoint_path,
            )
        
        # Prepare prompts
        prompts = []
        correct_answers = []
        original_questions = []  # Store original questions for saving
        
        for item in eval_data:
            prompt = format_prompt(item["q"], tokenizer)
            prompts.append(prompt)
            correct_answers.append(item["answer_matching_behavior"])
            original_questions.append(item["q"])  # Store original question
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            max_tokens=config.get("max_new_tokens", 512),  # Increased default for CoT
            stop=["User:", "Human:", "\n\nUser:", "\n\nHuman:"],  # Updated stop sequences
        )
        
        # Run inference in batches
        all_outputs = []
        batch_size = config["batch_size"]
        
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Evaluating {eval_name}", disable=True):  # Disable progress bar for speed
            batch_prompts = prompts[i:i+batch_size]
            # Pass lora_request only if it's not None
            if lora_request:
                outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
            else:
                outputs = llm.generate(batch_prompts, sampling_params)
            all_outputs.extend(outputs)
        
        # Extract answers and calculate accuracy
        predictions = []
        scores = []
        
        for i, (output, correct_answer) in enumerate(zip(all_outputs, correct_answers)):
            generated_text = output.outputs[0].text.strip()
            
            # Extract answer - look for content after </think>
            answer_found = None
            thinking_content = None
            
            # Split by </think> to separate thinking from answer
            if "</think>" in generated_text:
                parts = generated_text.split("</think>", 1)
                if len(parts) == 2:
                    thinking_content = parts[0].replace("<think>", "").strip()
                    answer_part = parts[1].strip()
                    
                    # Extract first number (1-4) from answer part
                    import re
                    match = re.search(r'^([1-4])', answer_part)
                    if match:
                        answer_found = match.group(1)
            else:
                # Fallback: if no </think> tag, try to find a number at the very end
                import re
                # Look for a standalone number 1-4 in the last 10 characters
                end_text = generated_text[-10:] if len(generated_text) > 10 else generated_text
                match = re.search(r'\b([1-4])\b', end_text)
                if match:
                    answer_found = match.group(1)
                    thinking_content = generated_text  # Consider whole response as thinking
            
            predictions.append({
                "prompt": prompts[i],  # Include the full prompt
                "original_question": original_questions[i],  # Include original question
                "generated": generated_text,
                "thinking": thinking_content,
                "extracted_answer": answer_found,
                "correct_answer": correct_answer,
                "is_correct": answer_found == correct_answer if answer_found else False
            })
            
            scores.append(1 if answer_found == correct_answer else 0)
        
        accuracy = np.mean(scores) * 100
        
        # Clean up
        del llm
        # torch.cuda.empty_cache()  # Let vLLM manage memory - removing this saves time
        
        return {
            "accuracy": accuracy,
            "num_correct": sum(scores),
            "num_total": len(scores),
            "predictions": predictions,  # Return all predictions, not just a sample
        }
        
    except Exception as e:
        print(f"Error evaluating {checkpoint_path} on {eval_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "accuracy": 0.0,
            "num_correct": 0,
            "num_total": 0,
            "error": str(e)
        }

def run_evaluation_job(job):
    """Run a single evaluation job."""
    return evaluate_checkpoint_on_eval(
        checkpoint_path=job["checkpoint_path"],
        base_model_path=job["base_model_path"],
        eval_name=job["eval_name"],
        eval_data=job["eval_data"],
        gpu_ids=job["gpu_ids"],
        config=job["config"]
    )

def save_results(results, results_dir, timestamp):
    """Save results to organized directory structure."""
    # Create timestamp directory
    timestamp_dir = os.path.join(results_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    
    # Create outputs directory for storing all LLM outputs
    outputs_dir = os.path.join(timestamp_dir, "llm_outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Convert tuple keys to strings for JSON serialization and save outputs
    json_safe_results = {}
    for key, value in results.items():
        if isinstance(key, tuple):
            dataset, step, eval_name = key
            key_str = f"{dataset}__step_{step}__{eval_name}"
            
            # Save LLM outputs for this evaluation
            output_file = os.path.join(outputs_dir, f"{key_str}_outputs.json")
            with open(output_file, "w") as f:
                # Save predictions with full outputs
                output_data = {
                    "dataset": dataset,
                    "step": step,
                    "eval_name": eval_name,
                    "accuracy": value["accuracy"],
                    "num_correct": value["num_correct"],
                    "num_total": value["num_total"],
                    "predictions": value.get("predictions", [])
                }
                json.dump(output_data, f, indent=2)
            
            # For the main results, don't include the full predictions
            json_safe_results[key_str] = {
                "accuracy": value["accuracy"],
                "num_correct": value["num_correct"],
                "num_total": value["num_total"],
                "error": value.get("error")
            }
        else:
            key_str = str(key)
            json_safe_results[key_str] = value
    
    # Save raw results (without full predictions)
    with open(os.path.join(timestamp_dir, "raw_results.json"), "w") as f:
        json.dump(json_safe_results, f, indent=2)
    
    # Organize results by dataset
    by_dataset = defaultdict(lambda: defaultdict(dict))
    
    for key, result in results.items():
        dataset, step, eval_name = key
        by_dataset[dataset][eval_name][step] = result
    
    # Create plots for each dataset
    for dataset, eval_results in by_dataset.items():
        dataset_dir = os.path.join(timestamp_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save dataset-specific results (without full predictions)
        dataset_results = {}
        for eval_name, step_results in eval_results.items():
            dataset_results[eval_name] = {}
            for step, result in step_results.items():
                dataset_results[eval_name][step] = {
                    "accuracy": result["accuracy"],
                    "num_correct": result["num_correct"],
                    "num_total": result["num_total"],
                    "error": result.get("error")
                }
        
        with open(os.path.join(dataset_dir, "results.json"), "w") as f:
            json.dump(dataset_results, f, indent=2)
        
        # Create accuracy plots
        plt.figure(figsize=(12, 8))
        
        for eval_name, step_results in eval_results.items():
            steps = sorted(step_results.keys())
            accuracies = [step_results[step]["accuracy"] for step in steps]
            
            plt.plot(steps, accuracies, marker='o', label=eval_name)
        
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Evaluation Results for {dataset}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_dir, "accuracy_plot.png"), dpi=300)
        plt.close()
        
        # Create heatmap of results
        eval_names = list(eval_results.keys())
        steps = sorted(list(set(step for eval_result in eval_results.values() for step in eval_result.keys())))
        
        heatmap_data = []
        for eval_name in eval_names:
            row = []
            for step in steps:
                if step in eval_results[eval_name]:
                    row.append(eval_results[eval_name][step]["accuracy"])
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            xticklabels=steps,
            yticklabels=eval_names,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=50,
            vmin=0,
            vmax=100
        )
        plt.xlabel("Training Step")
        plt.ylabel("Evaluation")
        plt.title(f"Accuracy Heatmap for {dataset}")
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_dir, "accuracy_heatmap.png"), dpi=300)
        plt.close()
    
    # Create summary CSV
    summary_data = []
    for (dataset, step, eval_name), result in results.items():
        summary_data.append({
            "dataset": dataset,
            "step": step,
            "eval": eval_name,
            "accuracy": result["accuracy"],
            "num_correct": result["num_correct"],
            "num_total": result["num_total"]
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(timestamp_dir, "summary.csv"), index=False)
    
    # Create a readable summary of outputs for manual inspection
    summary_file = os.path.join(timestamp_dir, "outputs_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Evaluation Summary - {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        for (dataset, step, eval_name), result in sorted(results.items()):
            f.write(f"Dataset: {dataset}, Step: {step}, Eval: {eval_name}\n")
            f.write(f"Accuracy: {result['accuracy']:.2f}% ({result['num_correct']}/{result['num_total']})\n")
            f.write("-"*40 + "\n")
            
            # Show first 5 examples
            predictions = result.get("predictions", [])[:5]
            for i, pred in enumerate(predictions):
                f.write(f"\nExample {i+1}:\n")
                if pred.get("original_question"):
                    f.write(f"Original Question: {pred['original_question'][:100]}...\n" if len(pred.get('original_question', '')) > 100 else f"Original Question: {pred.get('original_question', 'N/A')}\n")
                if pred.get("thinking"):
                    f.write(f"Thinking: {pred['thinking'][:200]}...\n" if len(pred.get('thinking', '')) > 200 else f"Thinking: {pred.get('thinking', 'N/A')}\n")
                f.write(f"Extracted Answer: {pred.get('extracted_answer', 'N/A')}\n")
                f.write(f"Correct Answer: {pred.get('correct_answer', 'N/A')}\n")
                f.write(f"Is Correct: {pred.get('is_correct', False)}\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"\nResults saved to: {timestamp_dir}")
    print(f"LLM outputs saved to: {outputs_dir}")
    print(f"Summary available at: {summary_file}")

def save_intermediate_results(results, results_dir, timestamp):
    """Save intermediate results during evaluation."""
    intermediate_dir = os.path.join(results_dir, timestamp, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Save current results
    intermediate_file = os.path.join(intermediate_dir, "intermediate_results.json")
    
    # Convert tuple keys to strings for JSON
    json_safe_results = {}
    for key, value in results.items():
        if isinstance(key, tuple):
            key_str = f"{key[0]}__step_{key[1]}__{key[2]}"
        else:
            key_str = str(key)
        json_safe_results[key_str] = value
    
    with open(intermediate_file, "w") as f:
        json.dump(json_safe_results, f, indent=2)
    
    # Also save a simple progress file
    progress_file = os.path.join(intermediate_dir, "progress.txt")
    with open(progress_file, "w") as f:
        f.write(f"Completed: {len(results)} evaluations\n")
        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def load_existing_results(results_dir, timestamp):
    """Load existing results if resuming an interrupted run."""
    intermediate_file = os.path.join(results_dir, timestamp, "intermediate", "intermediate_results.json")
    
    if os.path.exists(intermediate_file):
        print(f"\nFound existing intermediate results at {intermediate_file}")
        with open(intermediate_file, "r") as f:
            json_results = json.load(f)
        
        # Convert string keys back to tuples
        results = {}
        for key_str, value in json_results.items():
            if "__step_" in key_str and "__" in key_str:
                parts = key_str.split("__")
                dataset = parts[0]
                step = int(parts[1].replace("step_", ""))
                eval_name = parts[2]
                key = (dataset, step, eval_name)
            else:
                key = key_str
            results[key] = value
        
        print(f"Loaded {len(results)} existing results")
        return results
    
    return {}

def evaluate_multiple_checkpoints(job_list, gpu_ids, config):
    """Evaluate multiple checkpoints on the same GPU by hotswapping LoRA adapters."""
    
    # Set CUDA_VISIBLE_DEVICES for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    results = []
    llm = None
    base_model_path = config["base_model_path"]
    
    try:
        # Load the base model ONCE
        print(f"Loading base model on GPU {gpu_ids[0]}...")
        llm = LLM(
            model=base_model_path,
            enable_lora=True,
            max_lora_rank=32,
            tensor_parallel_size=config["tensor_parallel_size"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            trust_remote_code=True,
            max_model_len=4096,
            enable_prefix_caching=config.get("enable_prefix_caching", True),
            swap_space=config.get("swap_space", 4),
            disable_log_stats=True,
        )
        print(f"Model loaded on GPU {gpu_ids[0]}")
        
        # Process each job with the same model
        for job_idx, job in enumerate(job_list):
            checkpoint_path = job["checkpoint_path"]
            eval_name = job["eval_name"]
            eval_data = job["eval_data"]
            dataset = job["dataset"]
            step = job["step"]
            
            print(f"[GPU {gpu_ids[0]}] Processing job {job_idx + 1}/{len(job_list)}: {dataset} step {step} - {eval_name}")
            
            # Limit eval data
            if "max_eval_samples" in config and len(eval_data) > config["max_eval_samples"]:
                eval_data = eval_data[:config["max_eval_samples"]]
            
            # Get tokenizer from cache
            tokenizer_key = base_model_path if step == 0 else checkpoint_path
            if tokenizer_key not in _tokenizer_cache:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_key, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                _tokenizer_cache[tokenizer_key] = tokenizer
            else:
                tokenizer = _tokenizer_cache[tokenizer_key]
            
            # Prepare prompts
            prompts = []
            correct_answers = []
            original_questions = []
            
            for item in eval_data:
                prompt = format_prompt(item["q"], tokenizer)
                prompts.append(prompt)
                correct_answers.append(item["answer_matching_behavior"])
                original_questions.append(item["q"])
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=config["temperature"],
                max_tokens=config.get("max_new_tokens", 256),
                stop=["User:", "Human:", "\n\nUser:", "\n\nHuman:"],
            )
            
            # Create LoRA request only for non-base checkpoints
            lora_request = None
            if step != 0:
                lora_request = LoRARequest(
                    lora_name=f"adapter_{dataset}_{step}",
                    lora_int_id=job_idx + 1,  # Unique ID for each adapter
                    lora_local_path=checkpoint_path,
                )
            
            # Run inference in batches
            all_outputs = []
            batch_size = config["batch_size"]
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                if lora_request:
                    outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
                else:
                    outputs = llm.generate(batch_prompts, sampling_params)
                all_outputs.extend(outputs)
            
            # Extract answers and calculate accuracy
            predictions = []
            scores = []
            
            for i, (output, correct_answer) in enumerate(zip(all_outputs, correct_answers)):
                generated_text = output.outputs[0].text.strip()
                
                # Extract answer - look for content after </think>
                answer_found = None
                thinking_content = None
                
                if "</think>" in generated_text:
                    parts = generated_text.split("</think>", 1)
                    if len(parts) == 2:
                        thinking_content = parts[0].replace("<think>", "").strip()
                        answer_part = parts[1].strip()
                        
                        import re
                        match = re.search(r'^([1-4])', answer_part)
                        if match:
                            answer_found = match.group(1)
                else:
                    import re
                    end_text = generated_text[-10:] if len(generated_text) > 10 else generated_text
                    match = re.search(r'\b([1-4])\b', end_text)
                    if match:
                        answer_found = match.group(1)
                        thinking_content = generated_text
                
                predictions.append({
                    "prompt": prompts[i],
                    "original_question": original_questions[i],
                    "generated": generated_text,
                    "thinking": thinking_content,
                    "extracted_answer": answer_found,
                    "correct_answer": correct_answer,
                    "is_correct": answer_found == correct_answer if answer_found else False
                })
                
                scores.append(1 if answer_found == correct_answer else 0)
            
            accuracy = np.mean(scores) * 100
            
            results.append({
                "key": (dataset, step, eval_name),
                "result": {
                    "accuracy": accuracy,
                    "num_correct": sum(scores),
                    "num_total": len(scores),
                    "predictions": predictions,
                }
            })
            
    except Exception as e:
        print(f"Error in batch evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return partial results if any
        return results
    finally:
        # Clean up
        if llm is not None:
            del llm
    
    return results


def run_batch_evaluation_job(job_batch):
    """Run a batch of evaluation jobs on the same GPU."""
    return evaluate_multiple_checkpoints(
        job_list=job_batch["jobs"],
        gpu_ids=job_batch["gpu_ids"],
        config=job_batch["config"]
    )

def main(resume_timestamp=None):
    # Create or use existing timestamp
    if resume_timestamp:
        timestamp = resume_timestamp
        print(f"Resuming evaluation run from timestamp: {timestamp}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting new evaluation run with timestamp: {timestamp}")
    
    print("=" * 80)
    print(f"Evaluation Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Check for existing results (in case we're resuming)
    results = load_existing_results(CONFIG["results_dir"], timestamp)
    
    # Get all checkpoints and evals
    print("\nDiscovering checkpoints...")
    checkpoints = get_all_checkpoints(CONFIG["checkpoint_dir"])
    total_checkpoints = sum(len(ckpts) for ckpts in checkpoints.values())
    print(f"Found {total_checkpoints} checkpoints across {len(checkpoints)} datasets")
    
    print("\nLoading evaluations...")
    evals = get_all_evals(CONFIG["alek_evals_dir"])
    print(f"Found {len(evals)} evaluation sets")
    
    # Prepare all evaluation jobs
    jobs = []
    skipped_jobs = 0
    for dataset, dataset_checkpoints in checkpoints.items():
        for checkpoint in dataset_checkpoints:
            for eval_name, eval_data in evals.items():
                job_key = (dataset, checkpoint["step"], eval_name)
                
                # Skip if already completed
                if job_key in results:
                    skipped_jobs += 1
                    continue
                
                jobs.append({
                    "checkpoint_path": checkpoint["path"],
                    "base_model_path": CONFIG["base_model_path"],
                    "eval_name": eval_name,
                    "eval_data": eval_data,
                    "dataset": dataset,
                    "step": checkpoint["step"],
                    "config": CONFIG
                })
    
    print(f"\nTotal evaluation jobs: {len(jobs) + skipped_jobs}")
    print(f"  Already completed: {skipped_jobs}")
    print(f"  Remaining: {len(jobs)}")
    
    if len(jobs) == 0:
        print("\nAll evaluations already completed!")
        save_results(results, CONFIG["results_dir"], timestamp)
        return
    
    # Run evaluations in parallel
    start_time = time.time()
    completed_count = len(results)
    total_jobs = len(jobs) + len(results)
    
    if CONFIG.get("use_hotswapping", True):
        # Use hotswapping approach - load model once per GPU
        print("\n🚀 Using LoRA hotswapping for faster evaluation")
        
        # Group jobs for batch processing - each GPU gets a batch of jobs
        jobs_per_gpu = len(jobs) // CONFIG["num_parallel_jobs"] + 1
        gpu_job_batches = []
        
        for gpu_idx in range(CONFIG["num_parallel_jobs"]):
            start_idx = gpu_idx * jobs_per_gpu
            end_idx = min((gpu_idx + 1) * jobs_per_gpu, len(jobs))
            if start_idx < len(jobs):
                batch_jobs = jobs[start_idx:end_idx]
                gpu_job_batches.append({
                    "jobs": batch_jobs,
                    "gpu_ids": [CONFIG["gpus"][gpu_idx]],
                    "config": CONFIG
                })
        
        print(f"\nDistributed {len(jobs)} jobs across {len(gpu_job_batches)} GPUs")
        for i, batch in enumerate(gpu_job_batches):
            print(f"  GPU {CONFIG['gpus'][i]}: {len(batch['jobs'])} jobs")
        
        # Track running statistics
        running_accuracies = []
        save_interval = 10  # Save intermediate results every 10 completions
        
        with ProcessPoolExecutor(max_workers=CONFIG["num_parallel_jobs"]) as executor:
            # Submit batch jobs
            future_to_batch = {}
            
            for batch in gpu_job_batches:
                future = executor.submit(run_batch_evaluation_job, batch)
                future_to_batch[future] = batch
            
            # Process completed batches
            print(f"\nRunning evaluations with hotswapping on {CONFIG['num_parallel_jobs']} GPUs...")
            print("Each GPU loads the model once and swaps LoRA adapters between evaluations")
            print("-" * 80)
            
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                gpu_id = batch["gpu_ids"][0]
                
                try:
                    batch_results = future.result()
                    
                    # Process each result from the batch
                    for result_item in batch_results:
                        key = result_item["key"]
                        result = result_item["result"]
                        results[key] = result
                        completed_count += 1
                        running_accuracies.append(result['accuracy'])
                        
                        # Calculate statistics
                        elapsed_time = time.time() - start_time
                        avg_time_per_job = elapsed_time / (completed_count - skipped_jobs)
                        remaining_jobs = total_jobs - completed_count
                        eta_seconds = remaining_jobs * avg_time_per_job
                        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                        
                        # Print progress
                        dataset, step, eval_name = key
                        print(f"\n[{completed_count}/{total_jobs}] Completed: {dataset} | Step {step} | {eval_name}")
                        print(f"  Accuracy: {result['accuracy']:.2f}% ({result['num_correct']}/{result['num_total']})")
                        print(f"  Running avg accuracy: {np.mean(running_accuracies):.2f}%")
                        print(f"  Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                        print(f"  ETA: {eta_str}")
                        
                        # Save intermediate results periodically
                        if (completed_count - skipped_jobs) % save_interval == 0:
                            print(f"\n💾 Saving intermediate results...")
                            save_intermediate_results(results, CONFIG["results_dir"], timestamp)
                    
                    print(f"\n✅ GPU {gpu_id} completed all {len(batch_results)} jobs")
                    print("-" * 80)
                    
                except Exception as e:
                    print(f"\n❌ GPU {gpu_id} batch failed: {str(e)}")
                    # Add error results for all jobs in the failed batch
                    for job in batch["jobs"]:
                        key = (job["dataset"], job["step"], job["eval_name"])
                        if key not in results:  # Only add if not already processed
                            results[key] = {
                                "accuracy": 0.0,
                                "num_correct": 0,
                                "num_total": 0,
                                "error": str(e)
                            }
                            completed_count += 1
    else:
        # Use traditional approach - reload model for each evaluation
        print("\n⚠️  Using traditional evaluation (slower) - set 'use_hotswapping': True for 10-20x speedup")
        
        # Distribute GPUs among parallel jobs
        gpu_assignments = []
        for i in range(CONFIG["num_parallel_jobs"]):
            gpu_id = CONFIG["gpus"][i % len(CONFIG["gpus"])]
            gpu_assignments.append([gpu_id])
        
        # Track running statistics
        running_accuracies = []
        save_interval = 5  # Save intermediate results every 5 completions
        
        with ProcessPoolExecutor(max_workers=CONFIG["num_parallel_jobs"]) as executor:
            # Submit jobs with GPU assignments
            future_to_job = {}
            
            for i, job in enumerate(jobs):
                # Assign GPUs in round-robin fashion
                job["gpu_ids"] = gpu_assignments[i % len(gpu_assignments)]
                
                future = executor.submit(run_evaluation_job, job)
                future_to_job[future] = job
            
            # Process completed jobs
            print(f"\nRunning {len(jobs)} evaluations on {CONFIG['num_parallel_jobs']} GPUs...")
            print("-" * 80)
            
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                key = (job["dataset"], job["step"], job["eval_name"])
                completed_count += 1
                
                try:
                    result = future.result()
                    results[key] = result
                    running_accuracies.append(result['accuracy'])
                    
                    # Calculate statistics
                    elapsed_time = time.time() - start_time
                    avg_time_per_job = elapsed_time / (completed_count - skipped_jobs)
                    remaining_jobs = total_jobs - completed_count
                    eta_seconds = remaining_jobs * avg_time_per_job
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                    
                    # Print detailed progress
                    print(f"\n[{completed_count}/{total_jobs}] Completed: {job['dataset']} | Step {job['step']} | {job['eval_name']}")
                    print(f"  Accuracy: {result['accuracy']:.2f}% ({result['num_correct']}/{result['num_total']})")
                    print(f"  Running avg accuracy: {np.mean(running_accuracies):.2f}%")
                    print(f"  Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                    print(f"  ETA: {eta_str}")
                    print("-" * 80)
                    
                except Exception as e:
                    print(f"\n[{completed_count}/{total_jobs}] ERROR in {job['dataset']} | Step {job['step']} | {job['eval_name']}")
                    print(f"  Error: {str(e)}")
                    results[key] = {
                        "accuracy": 0.0,
                        "num_correct": 0,
                        "num_total": 0,
                        "error": str(e)
                    }
                
                # Save intermediate results periodically
                if (completed_count - skipped_jobs) % save_interval == 0:
                    print(f"\n💾 Saving intermediate results...")
                    save_intermediate_results(results, CONFIG["results_dir"], timestamp)
    
    # Save final results
    print("\n✅ All evaluations complete!")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
    
    print("\nSaving final results...")
    save_results(results, CONFIG["results_dir"], timestamp)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\nOverall accuracy: {:.2f}% ± {:.2f}%".format(
        np.mean([r["accuracy"] for r in results.values() if "error" not in r]),
        np.std([r["accuracy"] for r in results.values() if "error" not in r])
    ))
    
    print("\nAccuracy by dataset:")
    by_dataset = defaultdict(list)
    for (dataset, step, eval_name), result in results.items():
        if "error" not in result:
            by_dataset[dataset].append(result["accuracy"])
    
    for dataset, accuracies in sorted(by_dataset.items()):
        print(f"  {dataset}: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
    
    print("\nAccuracy by evaluation:")
    by_eval = defaultdict(list)
    for (dataset, step, eval_name), result in results.items():
        if "error" not in result:
            by_eval[eval_name].append(result["accuracy"])
    
    for eval_name, accuracies in sorted(by_eval.items()):
        print(f"  {eval_name}: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations on checkpoints")
    parser.add_argument("--resume", type=str, help="Resume from a specific timestamp (e.g., 20231125_143052)")
    args = parser.parse_args()
    
    # First, ensure MMLU is downloaded if not present
    if not os.path.exists("alek-evals/mmlu.json"):
        print("MMLU not found in alek-evals. Downloading...")
        subprocess.run(["python", "download_and_prepare_mmlu.py"], check=True)
    
    # Run main evaluation
    main(resume_timestamp=args.resume)