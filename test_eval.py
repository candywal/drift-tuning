#!/usr/bin/env python3
"""
Test script to verify evaluation setup with a small subset of data.
"""

import json
import os
from eval import (
    CONFIG, get_all_checkpoints, get_all_evals, 
    evaluate_checkpoint_on_eval, save_results
)
from datetime import datetime

def test_evaluation():
    """Run a small test evaluation."""
    print("Running test evaluation...")
    
    # Get checkpoints and evals
    checkpoints = get_all_checkpoints(CONFIG["checkpoint_dir"])
    evals = get_all_evals(CONFIG["alek_evals_dir"])
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    if not evals:
        print("No evaluations found!")
        return
    
    # Take first checkpoint and first eval
    first_dataset = list(checkpoints.keys())[0]
    first_checkpoint = checkpoints[first_dataset][0]
    first_eval_name = list(evals.keys())[0]
    first_eval_data = evals[first_eval_name][:10]  # Only first 10 questions
    
    print(f"\nTesting with:")
    print(f"  Dataset: {first_dataset}")
    print(f"  Checkpoint: {first_checkpoint['path']}")
    print(f"  Eval: {first_eval_name} (first 10 questions)")
    print(f"  GPUs: {CONFIG['gpus'][:CONFIG['tensor_parallel_size']]}")
    
    # Run evaluation
    result = evaluate_checkpoint_on_eval(
        checkpoint_path=first_checkpoint['path'],
        base_model_path=CONFIG['base_model_path'],
        eval_name=first_eval_name,
        eval_data=first_eval_data,
        gpu_ids=CONFIG['gpus'][:CONFIG['tensor_parallel_size']],
        config=CONFIG
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {result['accuracy']:.1f}%")
    print(f"  Correct: {result['num_correct']}/{result['num_total']}")
    
    if 'predictions' in result and result['predictions']:
        print(f"\nSample predictions:")
        for i, pred in enumerate(result['predictions'][:3]):
            print(f"\n  Question {i+1}:")
            print(f"    Generated: {pred['generated'][:50]}...")
            print(f"    Extracted: {pred['extracted_answer']}")
            print(f"    Correct: {pred['correct_answer']}")
            print(f"    Is Correct: {pred['is_correct']}")
    
    # Test save functionality
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_test")
    test_results = {
        (first_dataset, first_checkpoint['step'], first_eval_name): result
    }
    
    save_results(test_results, CONFIG["results_dir"], timestamp)
    print(f"\nTest results saved to: results/{timestamp}")

if __name__ == "__main__":
    test_evaluation() 