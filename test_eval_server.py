#!/usr/bin/env python3
"""
Test script for server-based evaluation system.
Tests with a small subset to verify everything works.
"""

import os
import sys
import json
from eval_server import CONFIG, get_all_checkpoints, get_all_evals
import subprocess
import time

def test_server_evaluation():
    """Test server-based evaluation with minimal data."""
    
    print("=" * 80)
    print("Server-based Evaluation Test")
    print("=" * 80)
    
    # Override config for testing
    test_config = CONFIG.copy()
    test_config["batch_size"] = 2  # Small batch for testing
    
    # Get checkpoints and evals
    print("\nChecking for checkpoints...")
    checkpoints = get_all_checkpoints(test_config["checkpoint_dir"])
    if not checkpoints:
        print("ERROR: No checkpoints found!")
        return
    
    print("\nChecking for evaluations...")
    evals = get_all_evals(test_config["alek_evals_dir"])
    if not evals:
        print("ERROR: No evaluations found!")
        return
    
    # Show what we found
    print(f"\nFound {sum(len(c) for c in checkpoints.values())} checkpoints:")
    for dataset, ckpts in checkpoints.items():
        print(f"  {dataset}: {len(ckpts)} checkpoints")
        for ckpt in ckpts[:2]:  # Show first 2
            print(f"    - Step {ckpt['step']}: {ckpt['path']}")
    
    print(f"\nFound {len(evals)} evaluation sets:")
    for eval_name in list(evals.keys())[:5]:  # Show first 5
        print(f"  - {eval_name} ({len(evals[eval_name])} questions)")
    
    # Create a minimal test evaluation file
    print("\nCreating minimal test evaluation...")
    test_eval = evals[list(evals.keys())[0]][:5]  # First 5 questions from first eval
    with open("alek-evals/test_minimal.json", "w") as f:
        json.dump(test_eval, f, indent=2)
    
    # Prompt to run full test
    print("\n" + "=" * 80)
    print("Test setup complete!")
    print("\nTo run a minimal server test, use:")
    print("  python eval_server.py")
    print("\nThe test will:")
    print("  1. Start 4 vLLM servers on GPUs 4-7")
    print("  2. Evaluate all checkpoints on the test_minimal eval (5 questions)")
    print("  3. Save results to the results/ directory")
    print("\nNote: Server startup takes ~45 seconds")
    print("=" * 80)
    
    response = input("\nRun minimal server test now? (y/n): ")
    if response.lower() == 'y':
        # Temporarily modify the eval directory to only include test
        print("\nStarting minimal server evaluation...")
        print("This will take several minutes...")
        
        # Run the server evaluation
        try:
            subprocess.run(["python", "eval_server.py"], check=True)
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except subprocess.CalledProcessError as e:
            print(f"\nError running server evaluation: {e}")
        finally:
            # Clean up test file
            if os.path.exists("alek-evals/test_minimal.json"):
                os.remove("alek-evals/test_minimal.json")
                print("\nCleaned up test file")

if __name__ == "__main__":
    test_server_evaluation() 