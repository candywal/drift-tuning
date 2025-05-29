#!/usr/bin/env python3
"""
Debug version of vLLM server startup test with various configurations.
"""

import os
import sys
import time
import subprocess
import requests
import signal
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_server_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_server_config(config_name, gpu_id=0, port=8001, extra_args=None):
    """Test a specific server configuration."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing configuration: {config_name}")
    logger.info(f"{'='*60}")
    
    # Build command with specific config
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
        "--port", str(port),
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    
    # Add extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Add more verbose logging
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"
    env["TRANSFORMERS_VERBOSITY"] = "debug"
    
    # Create log directory
    log_dir = "vllm_debug_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    stdout_file = open(f"{log_dir}/{config_name}_stdout.log", "w")
    stderr_file = open(f"{log_dir}/{config_name}_stderr.log", "w")
    
    logger.info("Starting server process...")
    
    try:
        # Start server
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            preexec_fn=os.setsid
        )
        
        logger.info(f"Server process started with PID {process.pid}")
        
        # Monitor for longer with more detailed output
        server_url = f"http://localhost:{port}"
        max_wait = 180  # 3 minutes
        start_time = time.time()
        last_log_check = 0
        
        while time.time() - start_time < max_wait:
            elapsed = int(time.time() - start_time)
            
            # Check if process is still running
            if process.poll() is not None:
                logger.error(f"Server process died after {elapsed}s!")
                stdout_file.close()
                stderr_file.close()
                
                # Read all output
                with open(f"{log_dir}/{config_name}_stdout.log", "r") as f:
                    logger.info("STDOUT:")
                    for line in f:
                        logger.info(f"  {line.rstrip()}")
                
                with open(f"{log_dir}/{config_name}_stderr.log", "r") as f:
                    logger.error("STDERR:")
                    for line in f:
                        logger.error(f"  {line.rstrip()}")
                return False
            
            # Try health check
            try:
                response = requests.get(f"{server_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info(f"✓ Server is ready after {elapsed}s!")
                    
                    # Get models info
                    try:
                        models_resp = requests.get(f"{server_url}/v1/models", timeout=5)
                        logger.info(f"Models endpoint: {models_resp.json()}")
                    except Exception as e:
                        logger.error(f"Failed to get models: {e}")
                    
                    return True
            except:
                pass
            
            # Check logs every 15 seconds
            if elapsed - last_log_check >= 15:
                logger.info(f"Still waiting... {elapsed}s elapsed")
                
                # Flush files
                stdout_file.flush()
                stderr_file.flush()
                
                # Show recent output
                with open(f"{log_dir}/{config_name}_stdout.log", "r") as f:
                    lines = f.readlines()
                    if lines:
                        logger.debug(f"Recent stdout (last 10 lines):")
                        for line in lines[-10:]:
                            logger.debug(f"  {line.rstrip()}")
                
                last_log_check = elapsed
            
            time.sleep(1)
        
        logger.error(f"Server failed to become ready after {max_wait}s")
        return False
        
    except Exception as e:
        logger.error(f"Exception during server test: {e}", exc_info=True)
        return False
    finally:
        # Cleanup
        if 'process' in locals() and process.poll() is None:
            logger.info("Shutting down test server...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                logger.info("Server shut down cleanly")
            except Exception as e:
                logger.error(f"Error shutting down server: {e}")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass
        
        stdout_file.close()
        stderr_file.close()

def main():
    """Run multiple server configurations to find what works."""
    logger.info("=" * 80)
    logger.info("vLLM Server Debug Test")
    logger.info("=" * 80)
    
    # Check system
    logger.info("\nSystem check...")
    try:
        # Check available memory
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            free_memory = result.stdout.strip().split('\n')
            logger.info("GPU free memory (MB):")
            for i, mem in enumerate(free_memory[:4]):  # Only show first 4 GPUs
                logger.info(f"  GPU {i}: {mem} MB")
    except:
        pass
    
    # Test configurations
    configs = [
        # 1. Minimal config - no compilation
        ("minimal_no_compile", ["--gpu-memory-utilization", "0.5", 
                                "--max-model-len", "2048",
                                "--disable-custom-all-reduce",
                                "--enforce-eager"]),  # Disable torch.compile
        
        # 2. Standard config with lower memory
        ("standard_low_mem", ["--gpu-memory-utilization", "0.7", 
                             "--max-model-len", "4096"]),
        
        # 3. With explicit dtype
        ("explicit_dtype", ["--gpu-memory-utilization", "0.8", 
                           "--max-model-len", "4096",
                           "--dtype", "float16"]),
        
        # 4. Disable all optimizations
        ("no_optimizations", ["--gpu-memory-utilization", "0.6",
                             "--max-model-len", "2048", 
                             "--enforce-eager",
                             "--disable-custom-all-reduce",
                             "--disable-spec-decode"]),
    ]
    
    results = {}
    
    # Test each configuration
    for config_name, extra_args in configs:
        logger.info(f"\nTesting: {config_name}")
        success = test_server_config(config_name, extra_args=extra_args)
        results[config_name] = success
        
        if success:
            logger.info(f"✓ {config_name} PASSED")
            # If we find a working config, we can stop
            break
        else:
            logger.error(f"✗ {config_name} FAILED")
        
        # Small delay between tests
        time.sleep(5)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY:")
    for config, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"  {config}: {status}")
    
    if any(results.values()):
        logger.info("\n✓ At least one configuration worked!")
        logger.info("Check the logs in vllm_debug_logs/ for the working configuration")
    else:
        logger.error("\n✗ All configurations failed")
        logger.info("\nPossible issues:")
        logger.info("1. The model might be too large for available GPU memory")
        logger.info("2. There might be a version compatibility issue")
        logger.info("3. Try running with a smaller model first")
        logger.info("4. Check if port 8001 is already in use: lsof -i :8001")

if __name__ == "__main__":
    main() 