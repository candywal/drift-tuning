#!/usr/bin/env python3
"""
Server-based evaluation system using vLLM servers with hot-swappable LoRA adapters.
"""

# Configuration
CONFIG = {
    "base_model_path": "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
    "checkpoint_dir": "old_bad_run",  # Directory containing checkpoints to evaluate
    "alek_evals_dir": "alek-evals",  # Directory containing evaluation JSONs
    "results_dir": "results",        # Where to save results
    "gpus": [0, 1, 2, 3],           # GPU IDs to use
    "batch_size": 32,               # Batch size for inference
    "max_new_tokens": 256,          # Max tokens to generate
    "temperature": 0.0,             # Temperature (0.0 for deterministic)
    "gpu_memory_utilization": 0.9,  # 90% GPU memory utilization
    "max_model_len": 4096,          # Max model length
    "server_ports": [8001, 8002, 8003, 8004],  # One port per GPU
}

import os
import json
import glob
import time
import numpy as np
from datetime import datetime
import subprocess
import requests
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from collections import defaultdict, deque
import pandas as pd
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional
import threading
import queue
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'eval_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def send_batch_request(server_url: str, prompts: List[str], adapter_name: Optional[str], config: dict):
    """Send a batch of prompts to a vLLM server asynchronously."""
    logger.debug(f"Sending batch of {len(prompts)} prompts to {server_url} with adapter {adapter_name}")
    
    # Send requests asynchronously
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            request = {
                "model": adapter_name if adapter_name else "base",  # Use adapter name or "base"
                "prompt": prompt,
                "max_tokens": config["max_new_tokens"],
                "temperature": config["temperature"],
                "stop": ["\n", "User:", "Human:"],
            }
            logger.debug(f"Creating request {i+1}/{len(prompts)} for model: {request['model']}")
            task = session.post(f"{server_url}/v1/completions", json=request)
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = []
        completed = 0
        for task in asyncio.as_completed(tasks):
            try:
                resp = await task
                result = await resp.json()
                responses.append(result)
                completed += 1
                logger.debug(f"Completed {completed}/{len(tasks)} requests")
            except Exception as e:
                logger.error(f"Error in request: {e}")
                responses.append({"error": str(e)})
    
    logger.debug(f"Batch request completed with {len(responses)} responses")
    return responses

class VLLMServerManager:
    """Manages vLLM servers on different GPUs."""
    
    def __init__(self, config):
        self.config = config
        self.servers = {}
        self.server_processes = {}
        self.adapter_registry = {}  # Maps adapter names to paths
        
    def start_servers(self, checkpoints):
        """Start vLLM servers on each GPU with pre-registered LoRA adapters."""
        logger.info("Starting vLLM servers...")
        
        # First, collect all unique adapter paths
        all_adapters = []
        for dataset_checkpoints in checkpoints.values():
            for checkpoint in dataset_checkpoints:
                adapter_name = f"{checkpoint['dataset']}_step{checkpoint['step']}"
                adapter_path = checkpoint['path']
                all_adapters.append((adapter_name, adapter_path))
                self.adapter_registry[adapter_name] = adapter_path
        
        logger.info(f"Total adapters to register: {len(all_adapters)}")
        
        # Distribute adapters across servers
        adapters_per_server = len(all_adapters) // len(self.config["gpus"]) + 1
        logger.debug(f"Adapters per server: {adapters_per_server}")
        
        for i, (gpu_id, port) in enumerate(zip(self.config["gpus"], self.config["server_ports"])):
            logger.info(f"Starting server on GPU {gpu_id}, port {port}...")
            
            # Get adapters for this server
            start_idx = i * adapters_per_server
            end_idx = min((i + 1) * adapters_per_server, len(all_adapters))
            server_adapters = all_adapters[start_idx:end_idx]
            
            # Build command with LoRA modules
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.config["base_model_path"],
                "--port", str(port),
                "--gpu-memory-utilization", str(self.config["gpu_memory_utilization"]),
                "--max-model-len", str(self.config["max_model_len"]),
                "--enable-lora",
                "--max-lora-rank", "32",
                "--trust-remote-code",
                "--disable-log-requests",
                "--served-model-name", "base",  # Base model name
                "--enforce-eager",  # Disable torch.compile to prevent hanging
            ]
            
            # Add LoRA modules
            if server_adapters:
                lora_modules = []
                for adapter_name, adapter_path in server_adapters:
                    lora_modules.append(f"{adapter_name}={adapter_path}")
                cmd.extend(["--lora-modules"] + lora_modules)
                logger.info(f"  Registering {len(server_adapters)} adapters")
                logger.debug(f"  Adapters: {[a[0] for a in server_adapters[:5]]}...")  # Show first 5
            
            # Set CUDA_VISIBLE_DEVICES for this server
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Start server as subprocess - redirect output to files for debugging
            log_dir = "vllm_logs"
            os.makedirs(log_dir, exist_ok=True)
            stdout_file = open(f"{log_dir}/server_gpu{gpu_id}_stdout.log", "w")
            stderr_file = open(f"{log_dir}/server_gpu{gpu_id}_stderr.log", "w")
            
            logger.debug(f"Starting server with command: {' '.join(cmd[:5])}...")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )
            
            self.server_processes[gpu_id] = process
            self.servers[gpu_id] = f"http://localhost:{port}"
            
            # Check if process started successfully
            time.sleep(2)
            if process.poll() is not None:
                logger.error(f"ERROR: Server on GPU {gpu_id} failed to start!")
                with open(f"{log_dir}/server_gpu{gpu_id}_stderr.log", "r") as f:
                    error_lines = f.readlines()[-50:]  # Last 50 lines
                    logger.error(f"Error output:\n{''.join(error_lines)}")
            else:
                logger.debug(f"Server process started with PID {process.pid}")
        
        # Wait for servers to be ready
        logger.info("\nWaiting for servers to be ready (this may take 1-2 minutes)...")
        all_ready = False
        start_time = time.time()
        timeout = 180  # 3 minutes timeout
        check_interval = 5
        
        while not all_ready and time.time() - start_time < timeout:
            all_ready = True
            server_statuses = []
            
            for gpu_id, server_url in self.servers.items():
                is_ready = self._check_server_ready(server_url, timeout=5)
                server_statuses.append((gpu_id, is_ready))
                if not is_ready:
                    all_ready = False
            
            elapsed = int(time.time() - start_time)
            status_str = ", ".join([f"GPU{gpu}: {'✓' if ready else '✗'}" for gpu, ready in server_statuses])
            logger.info(f"Server status at {elapsed}s: [{status_str}]")
            
            if not all_ready:
                time.sleep(check_interval)
        
        logger.info("")  # New line after progress
        
        # Final check and status report
        for gpu_id, server_url in self.servers.items():
            if self._check_server_ready(server_url, timeout=5):
                logger.info(f"✓ Server on GPU {gpu_id} is ready!")
            else:
                logger.error(f"✗ Server on GPU {gpu_id} failed to start properly")
                # Show last 20 lines of error log
                log_file = f"vllm_logs/server_gpu{gpu_id}_stderr.log"
                if os.path.exists(log_file):
                    logger.error(f"  Last errors from {log_file}:")
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        for line in lines[-20:]:
                            logger.error(f"    {line.rstrip()}")
    
    def get_server_for_adapter(self, adapter_name: Optional[str]) -> Tuple[int, str]:
        """Get the GPU ID and server URL that has the given adapter loaded."""
        if adapter_name is None:
            # For baseline (no adapter), use any server
            gpu_id = self.config["gpus"][0]
            logger.debug(f"Using GPU {gpu_id} for baseline (no adapter)")
            return gpu_id, self.servers[gpu_id]
        
        # Simple hash-based distribution
        hash_val = hash(adapter_name)
        gpu_idx = hash_val % len(self.config["gpus"])
        gpu_id = self.config["gpus"][gpu_idx]
        
        logger.debug(f"Using GPU {gpu_id} for adapter {adapter_name}")
        return gpu_id, self.servers[gpu_id]
    
    def _check_server_ready(self, server_url, timeout=60):
        """Check if a server is ready to accept requests."""
        logger.debug(f"Checking if server {server_url} is ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{server_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.debug(f"Server {server_url} is ready")
                    return True
            except Exception as e:
                logger.debug(f"Server {server_url} not ready yet: {type(e).__name__}")
            time.sleep(1)
        logger.debug(f"Server {server_url} failed to become ready after {timeout}s")
        return False
    
    def shutdown_servers(self):
        """Shutdown all vLLM servers."""
        logger.info("\nShutting down vLLM servers...")
        for gpu_id, process in self.server_processes.items():
            logger.debug(f"Shutting down server on GPU {gpu_id} (PID {process.pid})")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                logger.debug(f"Server on GPU {gpu_id} shut down cleanly")
            except Exception as e:
                logger.error(f"Error shutting down server on GPU {gpu_id}: {e}")
        logger.info("All servers shut down.")

def evaluate_with_server(adapter_name: Optional[str], server_url: str, gpu_id: int,
                        eval_name: str, eval_data: List[dict], config: dict, tokenizer):
    """Evaluate using a vLLM server."""
    logger.info(f"Starting evaluation on GPU {gpu_id} - {eval_name} - {adapter_name or 'base'}")
    
    # Prepare prompts
    prompts = []
    correct_answers = []
    
    for item in eval_data:
        prompt = format_prompt(item["q"], tokenizer)
        prompts.append(prompt)
        correct_answers.append(item["answer_matching_behavior"])
    
    logger.debug(f"Prepared {len(prompts)} prompts for evaluation")
    
    # Process in batches
    all_responses = []
    batch_size = config["batch_size"]
    
    # Run async batch processing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        for i in tqdm(range(0, len(prompts), batch_size), 
                     desc=f"GPU {gpu_id} - {eval_name} - {adapter_name or 'base'}",
                     disable=False):  # Keep progress bar visible
            batch_prompts = prompts[i:i+batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            responses = loop.run_until_complete(
                send_batch_request(server_url, batch_prompts, adapter_name, config)
            )
            all_responses.extend(responses)
            logger.debug(f"Received {len(responses)} responses for batch")
    finally:
        loop.close()
    
    # Extract answers and calculate accuracy
    predictions = []
    scores = []
    
    for response, correct_answer in zip(all_responses, correct_answers):
        if "choices" in response and len(response["choices"]) > 0:
            generated_text = response["choices"][0]["text"].strip()
        else:
            generated_text = ""
            logger.warning(f"Empty response: {response}")
        
        # Extract answer
        answer_found = extract_answer(generated_text)
        
        predictions.append({
            "generated": generated_text,
            "extracted_answer": answer_found,
            "correct_answer": correct_answer,
            "is_correct": answer_found == correct_answer if answer_found else False
        })
        
        scores.append(1 if answer_found == correct_answer else 0)
    
    accuracy = np.mean(scores) * 100
    logger.info(f"Completed evaluation: {eval_name} - {adapter_name or 'base'} - Accuracy: {accuracy:.1f}%")
    
    return {
        "accuracy": accuracy,
        "num_correct": sum(scores),
        "num_total": len(scores),
        "predictions": predictions[:10],  # Sample for debugging
    }

def extract_answer(generated_text: str) -> Optional[str]:
    """Extract answer from generated text."""
    if not generated_text:
        return None
    
    # First, check for direct number at the start
    if generated_text[0] in "1234":
        return generated_text[0]
    # Check for letters and convert
    elif generated_text[0].upper() in "ABCD":
        answer_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
        return answer_map[generated_text[0].upper()]
    else:
        # Look for patterns
        import re
        number_match = re.search(r'\b([1-4])\b', generated_text[:50])
        if number_match:
            return number_match.group(1)
        else:
            letter_match = re.search(r'\b([A-D])\b', generated_text[:50])
            if letter_match:
                answer_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
                return answer_map[letter_match.group(1)]
    return None

def format_prompt(question: str, tokenizer) -> str:
    """Format the question for the model."""
    messages = [{"role": "user", "content": question}]
    
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"User: {question}\n\nAssistant:"
    
    return prompt

class EvaluationPipeline:
    """Manages pipelined evaluation across GPUs."""
    
    def __init__(self, server_manager: VLLMServerManager, config: dict):
        self.server_manager = server_manager
        self.config = config
        self.results = {}
        self.result_queue = queue.Queue()
        
    def run_pipeline(self, checkpoints: Dict, evals: Dict):
        """Run pipelined evaluation across all checkpoints and evals."""
        logger.info("Starting evaluation pipeline")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["base_model_path"], 
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer loaded successfully: {type(tokenizer).__name__}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
            raise
        
        # Prepare all evaluation jobs
        jobs = []
        
        # Add baseline (no adapter) evaluations as step 0
        for dataset in checkpoints.keys():
            for eval_name, eval_data in evals.items():
                jobs.append({
                    "dataset": dataset,
                    "step": 0,
                    "adapter_name": None,  # No adapter for baseline
                    "eval_name": eval_name,
                    "eval_data": eval_data,
                })
        
        # Add checkpoint evaluations
        for dataset, dataset_checkpoints in checkpoints.items():
            for checkpoint in dataset_checkpoints:
                adapter_name = f"{checkpoint['dataset']}_step{checkpoint['step']}"
                for eval_name, eval_data in evals.items():
                    jobs.append({
                        "dataset": dataset,
                        "step": checkpoint["step"],
                        "adapter_name": adapter_name,
                        "eval_name": eval_name,
                        "eval_data": eval_data,
                    })
        
        logger.info(f"Total evaluation jobs: {len(jobs)}")
        logger.debug(f"Jobs breakdown: {len(checkpoints)} datasets × {len(evals)} evals × checkpoints")
        
        # Run evaluations with thread pool
        max_workers = len(self.config["gpus"]) * 2
        logger.info(f"Starting thread pool with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for job in jobs:
                # Get server for this adapter
                gpu_id, server_url = self.server_manager.get_server_for_adapter(job["adapter_name"])
                
                logger.debug(f"Submitting job: {job['dataset']}/step{job['step']}/{job['eval_name']} to GPU {gpu_id}")
                
                future = executor.submit(
                    evaluate_with_server,
                    adapter_name=job["adapter_name"],
                    server_url=server_url,
                    gpu_id=gpu_id,
                    eval_name=job["eval_name"],
                    eval_data=job["eval_data"],
                    config=self.config,
                    tokenizer=tokenizer
                )
                futures.append((future, job))
            
            # Process results
            completed = 0
            with tqdm(total=len(jobs), desc="Overall progress") as pbar:
                for future, job in futures:
                    try:
                        logger.debug(f"Waiting for job: {job['dataset']}/step{job['step']}/{job['eval_name']}")
                        result = future.result()
                        key = (job["dataset"], job["step"], job["eval_name"])
                        self.results[key] = result
                        completed += 1
                        logger.debug(f"Job completed ({completed}/{len(jobs)}): {key} - Accuracy: {result['accuracy']:.1f}%")
                        pbar.set_postfix({
                            "dataset": job["dataset"],
                            "step": job["step"],
                            "eval": job["eval_name"],
                            "accuracy": f"{result['accuracy']:.1f}%"
                        })
                    except Exception as e:
                        logger.error(f"Error evaluating {job}: {str(e)}", exc_info=True)
                        key = (job["dataset"], job["step"], job["eval_name"])
                        self.results[key] = {
                            "accuracy": 0.0,
                            "num_correct": 0,
                            "num_total": 0,
                            "error": str(e)
                        }
                    pbar.update(1)
        
        logger.info(f"Pipeline completed. Processed {len(self.results)} evaluations")
        return self.results

def get_all_checkpoints(checkpoint_dir):
    """Get all checkpoints organized by dataset."""
    logger.info(f"Scanning for checkpoints in: {checkpoint_dir}")
    checkpoints = defaultdict(list)
    
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return checkpoints
    
    for dataset_dir in os.listdir(checkpoint_dir):
        dataset_path = os.path.join(checkpoint_dir, dataset_dir)
        if os.path.isdir(dataset_path):
            checkpoint_dirs = glob.glob(os.path.join(dataset_path, "checkpoint-*"))
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
            
            logger.debug(f"Found {len(checkpoint_dirs)} checkpoints in {dataset_dir}")
            
            for ckpt_dir in checkpoint_dirs:
                step = int(ckpt_dir.split("-")[-1])
                checkpoints[dataset_dir].append({
                    "path": ckpt_dir,
                    "step": step,
                    "dataset": dataset_dir
                })
                logger.debug(f"  Added checkpoint: {dataset_dir}/step{step}")
    
    logger.info(f"Total checkpoints found: {sum(len(ckpts) for ckpts in checkpoints.values())} across {len(checkpoints)} datasets")
    return checkpoints

def get_all_evals(evals_dir):
    """Get all evaluation files."""
    logger.info(f"Loading evaluations from: {evals_dir}")
    
    if not os.path.exists(evals_dir):
        logger.error(f"Evaluation directory does not exist: {evals_dir}")
        return {}
    
    eval_files = glob.glob(os.path.join(evals_dir, "*.json"))
    evals = {}
    
    for eval_file in eval_files:
        eval_name = os.path.basename(eval_file).replace(".json", "")
        try:
            with open(eval_file, "r") as f:
                eval_data = json.load(f)
                evals[eval_name] = eval_data
                logger.debug(f"Loaded {eval_name}: {len(eval_data)} questions")
        except Exception as e:
            logger.error(f"Failed to load {eval_file}: {e}")
    
    logger.info(f"Loaded {len(evals)} evaluation sets")
    return evals

def save_results(results, results_dir, timestamp):
    """Save results to organized directory structure."""
    timestamp_dir = os.path.join(results_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    
    # Convert tuple keys to strings for JSON
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
    
    # Organize by dataset
    by_dataset = defaultdict(lambda: defaultdict(dict))
    
    for key, result in results.items():
        dataset, step, eval_name = key
        by_dataset[dataset][eval_name][step] = result
    
    # Create visualizations for each dataset
    for dataset, eval_results in by_dataset.items():
        dataset_dir = os.path.join(timestamp_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save dataset results
        with open(os.path.join(dataset_dir, "results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)
        
        # Create accuracy plot
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
        
        # Create heatmap
        eval_names = list(eval_results.keys())
        steps = sorted(list(set(step for eval_result in eval_results.values() 
                               for step in eval_result.keys())))
        
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("=" * 80)
    logger.info(f"Starting server-based evaluation at {timestamp}")
    logger.info(f"Configuration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
    
    # Initialize server manager
    server_manager = VLLMServerManager(CONFIG)
    
    try:
        # Get checkpoints first
        logger.info("\nDiscovering checkpoints...")
        checkpoints = get_all_checkpoints(CONFIG["checkpoint_dir"])
        total_checkpoints = sum(len(ckpts) for ckpts in checkpoints.values())
        logger.info(f"Found {total_checkpoints} checkpoints across {len(checkpoints)} datasets")
        
        if not checkpoints:
            logger.error("No checkpoints found! Exiting.")
            return
        
        # Start vLLM servers with discovered checkpoints
        logger.info("\nStarting vLLM servers...")
        server_manager.start_servers(checkpoints)
        
        logger.info("\nLoading evaluations...")
        evals = get_all_evals(CONFIG["alek_evals_dir"])
        logger.info(f"Found {len(evals)} evaluation sets")
        
        if not evals:
            logger.error("No evaluations found! Exiting.")
            return
        
        # Ensure MMLU is available
        if not os.path.exists("alek-evals/mmlu.json"):
            logger.info("MMLU not found. Downloading...")
            try:
                subprocess.run(["python", "download_and_prepare_mmlu.py"], check=True)
                evals = get_all_evals(CONFIG["alek_evals_dir"])
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download MMLU: {e}")
        
        # Run evaluation pipeline
        logger.info("\nStarting evaluation pipeline...")
        pipeline = EvaluationPipeline(server_manager, CONFIG)
        results = pipeline.run_pipeline(checkpoints, evals)
        
        # Save results
        logger.info("\nSaving results...")
        save_results(results, CONFIG["results_dir"], timestamp)
        
        # Print summary
        logger.info("\nSummary by dataset:")
        by_dataset = defaultdict(list)
        for (dataset, step, eval_name), result in results.items():
            by_dataset[dataset].append(result["accuracy"])
        
        for dataset, accuracies in by_dataset.items():
            logger.info(f"  {dataset}: {np.mean(accuracies):.1f}% ± {np.std(accuracies):.1f}%")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise
    finally:
        # Shutdown servers
        logger.info("\nShutting down servers...")
        server_manager.shutdown_servers()
    
    logger.info("\nEvaluation complete!")

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.warning("\nInterrupted! Shutting down servers...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1) 