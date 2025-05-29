#!/usr/bin/env python3
"""
Quick test to verify eval.py works with minimal data.
Tests one checkpoint on one eval with 10 samples.
"""

import os
import json
import subprocess
import sys

# Import the evaluation functions
from eval import get_all_checkpoints, get_all_evals, evaluate_checkpoint_on_eval

def test_quick_eval():
    """Run a minimal evaluation test."""
    print("=" * 80)
    print("Quick Evaluation Test")
    print("=" * 80)
    
    # Test configuration
    test_config = {
        "base_model_path": "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
        "batch_size": 8,
        "max_new_tokens": 256,
        "temperature": 0.0,
        "tensor_parallel_size": 1,
        "max_eval_samples": 10,  # Just 10 questions for quick test
    }
    
    # Get checkpoints
    checkpoints = get_all_checkpoints("old_bad_run")
    if not checkpoints:
        print("ERROR: No checkpoints found!")
        return False
    
    # Get first checkpoint
    first_dataset = list(checkpoints.keys())[0]
    first_checkpoint = checkpoints[first_dataset][0]
    print(f"\nTesting with checkpoint: {first_checkpoint['path']}")
    
    # Get evals
    evals = get_all_evals("alek-evals")
    if not evals:
        print("ERROR: No evaluations found!")
        return False
    
    # Get first eval
    first_eval_name = list(evals.keys())[0]
    first_eval_data = evals[first_eval_name][:10]  # Just 10 questions
    print(f"Testing with eval: {first_eval_name} (10 questions)")
    
    # Run evaluation
    print("\nRunning evaluation...")
    try:
        result = evaluate_checkpoint_on_eval(
            checkpoint_path=first_checkpoint["path"],
            base_model_path=test_config["base_model_path"],
            eval_name=first_eval_name,
            eval_data=first_eval_data,
            gpu_ids=[0],  # Use GPU 0
            config=test_config
        )
        
        print(f"\nResult: {result['accuracy']:.1f}% ({result['num_correct']}/{result['num_total']})")
        
        # Show a few predictions
        print("\nSample predictions:")
        for i, pred in enumerate(result['predictions'][:3]):
            print(f"\n{i+1}. Generated: {pred['generated'][:100]}...")
            print(f"   Extracted: {pred['extracted_answer']}, Correct: {pred['correct_answer']}, Match: {pred['is_correct']}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Ensure MMLU is available
    if not os.path.exists("alek-evals/mmlu.json"):
        print("MMLU not found. Downloading...")
        subprocess.run(["python", "download_and_prepare_mmlu.py"], check=True)
    
    # Run test
    success = test_quick_eval()
    
    if success:
        print("\n✅ Quick test passed! You can now run the full evaluation with:")
        print("   python eval.py")
    else:
        print("\n❌ Quick test failed! Please check the errors above.")
        sys.exit(1) 