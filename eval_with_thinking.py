# %%

# Configuration
CONFIG = {
    "base_model_path": "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
    "checkpoint_dir": "old_bad_run",  # Change this to evaluate different runs
    "alek_evals_dir": "alek-evals",
    "results_dir": "results",
    "gpus": [4, 5, 6, 7],  # H100 GPUs to use
    "batch_size": 32,
    "max_new_tokens": 256,
    "temperature": 0.0,  # Deterministic generation for evals
    "tensor_parallel_size": 2,  # Use 2 GPUs per model for 32B model
    "num_parallel_jobs": 2,  # Run 2 models in parallel (2 GPUs each)
}

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

def get_all_checkpoints(checkpoint_dir):
    """Get all checkpoints organized by dataset."""
    checkpoints = defaultdict(list)
    
    for dataset_dir in os.listdir(checkpoint_dir):
        dataset_path = os.path.join(checkpoint_dir, dataset_dir)
        if os.path.isdir(dataset_path):
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
    """Format the question for the model."""
    # Add instructions for thinking format
    instruction = """Please answer the following multiple choice question.

First, think through the problem step by step inside <think></think> tags.
Then, after your thinking, output ONLY a single character (1, 2, 3, or 4) as your final answer.

Example format:
<think>
Let me analyze this question...
[your reasoning here]
</think>
2

Question:"""
    
    # Combine instruction with the question
    full_prompt = f"{instruction}\n{question}"
    
    messages = [
        {"role": "user", "content": full_prompt}
    ]
    
    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback format
        print("WARNING: Using fallback format for prompt")
        prompt = f"User: {full_prompt}\n\nAssistant:"
    
    return prompt

def evaluate_checkpoint_on_eval(checkpoint_path, base_model_path, eval_name, eval_data, gpu_ids, config):
    """Evaluate a single checkpoint on a single eval."""
    
    # Set CUDA_VISIBLE_DEVICES for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize vLLM with LoRA support
        llm = LLM(
            model=base_model_path,
            enable_lora=True,
            max_lora_rank=32,  # Set based on your LoRA config
            tensor_parallel_size=config["tensor_parallel_size"],
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            max_model_len=4096,
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
        
        for item in eval_data:
            prompt = format_prompt(item["q"], tokenizer)
            prompts.append(prompt)
            correct_answers.append(item["answer_matching_behavior"])
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            max_tokens=config["max_new_tokens"],
            # Removed stop sequences to allow full thinking + answer format
            # The model should naturally stop after providing the single digit answer
        )
        
        # Run inference in batches
        all_outputs = []
        batch_size = config["batch_size"]
        
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Evaluating {eval_name}"):
            batch_prompts = prompts[i:i+batch_size]
            outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
            all_outputs.extend(outputs)
        
        # Extract answers and calculate accuracy
        predictions = []
        scores = []
        
        for output, correct_answer in zip(all_outputs, correct_answers):
            generated_text = output.outputs[0].text.strip()
            
            # Extract answer - look for answer after </think> tag first
            answer_found = None
            
            # First, try to find answer after </think> tag
            import re
            think_pattern = r'</think>\s*([1-4])'
            think_match = re.search(think_pattern, generated_text, re.IGNORECASE)
            
            if think_match:
                answer_found = think_match.group(1)
            else:
                # Fallback: check if the model output just a single character after thinking
                # (in case it forgot the closing tag)
                lines = generated_text.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line in "1234":
                        answer_found = line
                        break
                
                # If still not found, use the original extraction logic as fallback
                if not answer_found:
                    # Check for direct number at the start
                    if generated_text and generated_text[0] in "1234":
                        answer_found = generated_text[0]
                    # Check for letters and convert
                    elif generated_text and generated_text[0].upper() in "ABCD":
                        answer_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
                        answer_found = answer_map[generated_text[0].upper()]
                    else:
                        # Look for patterns like "The answer is 1" or "Option 1"
                        number_match = re.search(r'\b([1-4])\b', generated_text[:50])  # Check first 50 chars
                        if number_match:
                            answer_found = number_match.group(1)
                        else:
                            letter_match = re.search(r'\b([A-D])\b', generated_text[:50])
                            if letter_match:
                                answer_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
                                answer_found = answer_map[letter_match.group(1)]
            
            predictions.append({
                "generated": generated_text,
                "extracted_answer": answer_found,
                "correct_answer": correct_answer,
                "is_correct": answer_found == correct_answer if answer_found else False
            })
            
            scores.append(1 if answer_found == correct_answer else 0)
        
        accuracy = np.mean(scores) * 100
        
        # Clean up
        del llm
        torch.cuda.empty_cache()
        
        return {
            "accuracy": accuracy,
            "num_correct": sum(scores),
            "num_total": len(scores),
            "predictions": predictions[:10],  # Sample of predictions for debugging
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
    
    # Convert tuple keys to strings for JSON serialization
    json_safe_results = {}
    for key, value in results.items():
        if isinstance(key, tuple):
            key_str = f"{key[0]}__step_{key[1]}__{key[2]}"
        else:
            key_str = str(key)
        json_safe_results[key_str] = value
    
    # Save raw results
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
        
        # Save dataset-specific results
        with open(os.path.join(dataset_dir, "results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        
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
    
    print(f"\nResults saved to: {timestamp_dir}")

def main():
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print(f"Starting evaluation run at {timestamp}")
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
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
    for dataset, dataset_checkpoints in checkpoints.items():
        for checkpoint in dataset_checkpoints:
            for eval_name, eval_data in evals.items():
                jobs.append({
                    "checkpoint_path": checkpoint["path"],
                    "base_model_path": CONFIG["base_model_path"],
                    "eval_name": eval_name,
                    "eval_data": eval_data,
                    "dataset": dataset,
                    "step": checkpoint["step"],
                    "config": CONFIG
                })
    
    print(f"\nTotal evaluation jobs: {len(jobs)}")
    
    # Run evaluations in parallel
    results = {}
    
    # Distribute GPUs among parallel jobs
    gpu_assignments = []
    for i in range(CONFIG["num_parallel_jobs"]):
        start_gpu = i * CONFIG["tensor_parallel_size"]
        gpu_ids = CONFIG["gpus"][start_gpu:start_gpu + CONFIG["tensor_parallel_size"]]
        gpu_assignments.append(gpu_ids)
    
    with ProcessPoolExecutor(max_workers=CONFIG["num_parallel_jobs"]) as executor:
        # Submit jobs with GPU assignments
        future_to_job = {}
        
        for i, job in enumerate(jobs):
            # Assign GPUs in round-robin fashion
            job["gpu_ids"] = gpu_assignments[i % len(gpu_assignments)]
            
            future = executor.submit(run_evaluation_job, job)
            future_to_job[future] = job
        
        # Process completed jobs
        with tqdm(total=len(jobs), desc="Running evaluations") as pbar:
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                key = (job["dataset"], job["step"], job["eval_name"])
                
                try:
                    result = future.result()
                    results[key] = result
                    pbar.set_postfix({
                        "dataset": job["dataset"],
                        "step": job["step"],
                        "eval": job["eval_name"],
                        "accuracy": f"{result['accuracy']:.1f}%"
                    })
                except Exception as e:
                    print(f"\nError in job {key}: {str(e)}")
                    results[key] = {
                        "accuracy": 0.0,
                        "num_correct": 0,
                        "num_total": 0,
                        "error": str(e)
                    }
                
                pbar.update(1)
    
    # Save results
    print("\nSaving results...")
    save_results(results, CONFIG["results_dir"], timestamp)
    
    print("\nEvaluation complete!")
    
    # Print summary statistics
    print("\nSummary by dataset:")
    by_dataset = defaultdict(list)
    for (dataset, step, eval_name), result in results.items():
        by_dataset[dataset].append(result["accuracy"])
    
    for dataset, accuracies in by_dataset.items():
        print(f"  {dataset}: {np.mean(accuracies):.1f}% ± {np.std(accuracies):.1f}%")

if __name__ == "__main__":
    # First, ensure MMLU is downloaded if not present
    if not os.path.exists("alek-evals/mmlu.json"):
        print("MMLU not found in alek-evals. Downloading...")
        subprocess.run(["python", "download_and_prepare_mmlu.py"], check=True)
    
    # Run main evaluation
    main()