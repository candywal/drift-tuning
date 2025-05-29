#!/usr/bin/env python3
"""
Test script to verify evaluation setup with parallel GPU execution.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from eval import (
    CONFIG, get_all_checkpoints, get_all_evals, 
    evaluate_checkpoint_on_eval, save_results, run_evaluation_job
)
from datetime import datetime

def test_single_gpu_evaluation(gpu_id):
    """Test evaluation on a single GPU."""
    print(f"\n[GPU {gpu_id}] Starting test evaluation...")
    
    # Override config for single GPU test
    test_config = CONFIG.copy()
    test_config["tensor_parallel_size"] = 1
    test_config["batch_size"] = 16  # Smaller batch for testing
    
    # Add compilation disable flags
    test_config["enforce_eager"] = True
    test_config["disable_custom_all_reduce"] = True
    
    # Get checkpoints and evals
    checkpoints = get_all_checkpoints(test_config["checkpoint_dir"])
    evals = get_all_evals(test_config["alek_evals_dir"])
    
    if not checkpoints:
        return f"[GPU {gpu_id}] No checkpoints found!"
    
    if not evals:
        return f"[GPU {gpu_id}] No evaluations found!"
    
    # Take first checkpoint and first eval
    first_dataset = list(checkpoints.keys())[0]
    first_checkpoint = checkpoints[first_dataset][0]
    first_eval_name = list(evals.keys())[0]
    first_eval_data = evals[first_eval_name][:5]  # Only first 5 questions for quick test
    
    print(f"[GPU {gpu_id}] Testing with:")
    print(f"  Dataset: {first_dataset}")
    print(f"  Checkpoint: {first_checkpoint['path']} (step {first_checkpoint['step']})")
    print(f"  Eval: {first_eval_name} (first 5 questions)")
    
    # Run evaluation
    try:
        result = evaluate_checkpoint_on_eval(
            checkpoint_path=first_checkpoint['path'],
            base_model_path=test_config['base_model_path'],
            eval_name=first_eval_name,
            eval_data=first_eval_data,
            gpu_ids=[gpu_id],
            config=test_config
        )
        
        return f"[GPU {gpu_id}] Success! Accuracy: {result['accuracy']:.1f}% ({result['num_correct']}/{result['num_total']})"
    except Exception as e:
        import traceback
        return f"[GPU {gpu_id}] Error: {str(e)}\n{traceback.format_exc()}"

def test_parallel_evaluation():
    """Test parallel evaluation on multiple GPUs."""
    print("Testing parallel evaluation on 4 GPUs...")
    print("=" * 80)
    
    # First test: Single GPU sequential test
    print("\nPhase 1: Testing each GPU individually (sequential)...")
    for gpu_id in CONFIG["gpus"]:
        result = test_single_gpu_evaluation(gpu_id)
        print(result)
        time.sleep(2)  # Wait between tests
    
    # Second test: Parallel execution
    print("\n" + "=" * 80)
    print("Phase 2: Testing parallel execution on all GPUs...")
    
    # Prepare jobs for parallel execution
    checkpoints = get_all_checkpoints(CONFIG["checkpoint_dir"])
    evals = get_all_evals(CONFIG["alek_evals_dir"])
    
    # Create 4 different jobs (one per GPU)
    jobs = []
    datasets = list(checkpoints.keys())
    eval_names = list(evals.keys())
    
    for i in range(4):
        dataset = datasets[i % len(datasets)]
        checkpoint = checkpoints[dataset][0]  # Use first checkpoint
        eval_name = eval_names[i % len(eval_names)]
        eval_data = evals[eval_name][:10]  # 10 questions each
        
        job = {
            "checkpoint_path": checkpoint["path"],
            "base_model_path": CONFIG["base_model_path"],
            "eval_name": eval_name,
            "eval_data": eval_data,
            "dataset": dataset,
            "step": checkpoint["step"],
            "gpu_ids": [CONFIG["gpus"][i]],  # Assign specific GPU
            "config": CONFIG
        }
        jobs.append(job)
        
        print(f"\nJob {i}: GPU {CONFIG['gpus'][i]} | {dataset} | step {checkpoint['step']} | {eval_name}")
    
    # Run jobs in parallel
    print("\nStarting parallel execution...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all jobs with staggered start
        futures = []
        for i, job in enumerate(jobs):
            future = executor.submit(run_evaluation_job, job)
            futures.append((future, job))
            time.sleep(0.5)  # Small delay between submissions
        
        # Wait for results
        for future, job in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                print(f"\n✅ GPU {job['gpu_ids'][0]} completed: {job['dataset']} | {job['eval_name']}")
                print(f"   Accuracy: {result['accuracy']:.1f}% ({result['num_correct']}/{result['num_total']})")
            except Exception as e:
                print(f"\n❌ GPU {job['gpu_ids'][0]} failed: {job['dataset']} | {job['eval_name']}")
                print(f"   Error: {str(e)}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time for parallel execution: {elapsed:.1f} seconds")
    
    # Test the full evaluation pipeline with proper config
    print("\n" + "=" * 80)
    print("Phase 3: Testing full pipeline configuration...")
    print(f"  Number of parallel jobs: {CONFIG['num_parallel_jobs']}")
    print(f"  GPUs: {CONFIG['gpus']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Tensor parallel size: {CONFIG['tensor_parallel_size']}")
    
    # Check GPU memory
    print("\nChecking GPU availability and memory...")
    os.system("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv")

if __name__ == "__main__":
    test_parallel_evaluation() 