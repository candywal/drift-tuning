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
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_server_evaluation():
    """Test server-based evaluation with minimal data."""
    
    logger.info("=" * 80)
    logger.info("Server-based Evaluation Test")
    logger.info("=" * 80)
    
    # Override config for testing
    test_config = CONFIG.copy()
    test_config["batch_size"] = 2  # Small batch for testing
    
    # Get checkpoints and evals
    logger.info("\nChecking for checkpoints...")
    try:
        checkpoints = get_all_checkpoints(test_config["checkpoint_dir"])
        logger.debug(f"get_all_checkpoints returned: {len(checkpoints)} datasets")
    except Exception as e:
        logger.error(f"Error getting checkpoints: {e}", exc_info=True)
        return
    
    if not checkpoints:
        logger.error("ERROR: No checkpoints found!")
        return
    
    logger.info("\nChecking for evaluations...")
    try:
        evals = get_all_evals(test_config["alek_evals_dir"])
        logger.debug(f"get_all_evals returned: {len(evals)} evaluation sets")
    except Exception as e:
        logger.error(f"Error getting evaluations: {e}", exc_info=True)
        return
    
    if not evals:
        logger.error("ERROR: No evaluations found!")
        return
    
    # Show what we found
    logger.info(f"\nFound {sum(len(c) for c in checkpoints.values())} checkpoints:")
    for dataset, ckpts in checkpoints.items():
        logger.info(f"  {dataset}: {len(ckpts)} checkpoints")
        for ckpt in ckpts[:2]:  # Show first 2
            logger.info(f"    - Step {ckpt['step']}: {ckpt['path']}")
    
    logger.info(f"\nFound {len(evals)} evaluation sets:")
    for eval_name in list(evals.keys())[:5]:  # Show first 5
        logger.info(f"  - {eval_name} ({len(evals[eval_name])} questions)")
    
    # Create a minimal test evaluation file
    logger.info("\nCreating minimal test evaluation...")
    test_eval = evals[list(evals.keys())[0]][:5]  # First 5 questions from first eval
    test_eval_path = "alek-evals/test_minimal.json"
    with open(test_eval_path, "w") as f:
        json.dump(test_eval, f, indent=2)
    logger.debug(f"Created test evaluation at {test_eval_path} with {len(test_eval)} questions")
    
    # Prompt to run full test
    logger.info("\n" + "=" * 80)
    logger.info("Test setup complete!")
    logger.info("\nTo run a minimal server test, use:")
    logger.info("  python eval_server.py")
    logger.info("\nThe test will:")
    logger.info("  1. Start 4 vLLM servers on GPUs 0-3")
    logger.info("  2. Evaluate all checkpoints on the test_minimal eval (5 questions)")
    logger.info("  3. Save results to the results/ directory")
    logger.info("\nNote: Server startup takes ~45 seconds")
    logger.info("=" * 80)
    
    response = input("\nRun minimal server test now? (y/n): ")
    if response.lower() == 'y':
        # Temporarily modify the eval directory to only include test
        logger.info("\nStarting minimal server evaluation...")
        logger.info("This will take several minutes...")
        logger.info("Streaming output from eval_server.py...")
        
        # Run the server evaluation with real-time output
        try:
            # Use Popen to stream output in real-time
            process = subprocess.Popen(
                ["python", "eval_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    logger.info(f"[eval_server] {line.rstrip()}")
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code != 0:
                logger.error(f"eval_server.py exited with code {return_code}")
            else:
                logger.info("eval_server.py completed successfully")
                
        except KeyboardInterrupt:
            logger.warning("\nTest interrupted by user")
            if 'process' in locals():
                process.terminate()
                process.wait()
        except Exception as e:
            logger.error(f"Error running server evaluation: {e}", exc_info=True)
        finally:
            # Clean up test file
            if os.path.exists(test_eval_path):
                os.remove(test_eval_path)
                logger.info("Cleaned up test file")

if __name__ == "__main__":
    logger.info(f"Starting test at {datetime.now()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check if required directories exist
    logger.debug(f"Checking required directories...")
    logger.debug(f"  checkpoint_dir ({CONFIG['checkpoint_dir']}): {os.path.exists(CONFIG['checkpoint_dir'])}")
    logger.debug(f"  alek_evals_dir ({CONFIG['alek_evals_dir']}): {os.path.exists(CONFIG['alek_evals_dir'])}")
    logger.debug(f"  results_dir ({CONFIG['results_dir']}): {os.path.exists(CONFIG['results_dir'])}")
    
    test_server_evaluation() 