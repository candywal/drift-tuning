#!/usr/bin/env python3
"""
Diagnostic script to test vLLM server startup only.
This helps identify if the hanging is during server startup.
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
        logging.FileHandler(f'test_server_startup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_single_server(gpu_id=0, port=8001):
    """Test starting a single vLLM server."""
    
    logger.info(f"Testing vLLM server startup on GPU {gpu_id}, port {port}")
    
    # Build minimal command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
        "--port", str(port),
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "4096",
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Set CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create log directory
    log_dir = "vllm_test_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    stdout_file = open(f"{log_dir}/test_server_stdout.log", "w")
    stderr_file = open(f"{log_dir}/test_server_stderr.log", "w")
    
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
        
        # Check if process is still running after 2 seconds
        time.sleep(2)
        if process.poll() is not None:
            logger.error("Server process died immediately!")
            with open(f"{log_dir}/test_server_stderr.log", "r") as f:
                error_lines = f.readlines()
                logger.error("Error output:")
                for line in error_lines:
                    logger.error(f"  {line.rstrip()}")
            return False
        
        # Wait for server to be ready
        logger.info("Waiting for server to be ready...")
        server_url = f"http://localhost:{port}"
        max_wait = 120  # 2 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            elapsed = int(time.time() - start_time)
            
            # Check if process is still running
            if process.poll() is not None:
                logger.error(f"Server process died after {elapsed}s!")
                with open(f"{log_dir}/test_server_stderr.log", "r") as f:
                    error_lines = f.readlines()[-50:]
                    logger.error("Last 50 lines of error output:")
                    for line in error_lines:
                        logger.error(f"  {line.rstrip()}")
                return False
            
            # Try health check
            try:
                logger.debug(f"Checking health at {elapsed}s...")
                response = requests.get(f"{server_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info(f"✓ Server is ready after {elapsed}s!")
                    
                    # Try a simple test request
                    logger.info("Testing a simple completion request...")
                    test_request = {
                        "model": "base",
                        "prompt": "Hello, world!",
                        "max_tokens": 10,
                        "temperature": 0.0,
                    }
                    
                    response = requests.post(f"{server_url}/v1/completions", json=test_request, timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✓ Test request successful! Response: {result}")
                    else:
                        logger.error(f"✗ Test request failed with status {response.status_code}")
                        logger.error(f"  Response: {response.text}")
                    
                    return True
                else:
                    logger.debug(f"Health check returned status {response.status_code}")
            except requests.exceptions.Timeout:
                logger.debug(f"Health check timed out at {elapsed}s")
            except requests.exceptions.ConnectionError:
                logger.debug(f"Connection refused at {elapsed}s")
            except Exception as e:
                logger.debug(f"Health check error at {elapsed}s: {type(e).__name__}: {e}")
            
            # Show server output periodically
            if elapsed % 10 == 0:
                logger.info(f"Still waiting... {elapsed}s elapsed")
                
                # Check stdout log
                stdout_file.flush()
                with open(f"{log_dir}/test_server_stdout.log", "r") as f:
                    lines = f.readlines()
                    if lines:
                        logger.debug(f"Last stdout lines:")
                        for line in lines[-5:]:
                            logger.debug(f"  {line.rstrip()}")
                
                # Check stderr log
                stderr_file.flush()
                with open(f"{log_dir}/test_server_stderr.log", "r") as f:
                    lines = f.readlines()
                    if lines:
                        logger.debug(f"Last stderr lines:")
                        for line in lines[-5:]:
                            logger.debug(f"  {line.rstrip()}")
            
            time.sleep(1)
        
        logger.error(f"Server failed to become ready after {max_wait}s")
        return False
        
    except Exception as e:
        logger.error(f"Exception during server test: {e}", exc_info=True)
        return False
    finally:
        # Cleanup
        if 'process' in locals():
            logger.info("Shutting down test server...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                logger.info("Server shut down cleanly")
            except Exception as e:
                logger.error(f"Error shutting down server: {e}")
        
        stdout_file.close()
        stderr_file.close()

def main():
    """Run server startup test."""
    logger.info("=" * 80)
    logger.info("vLLM Server Startup Test")
    logger.info("=" * 80)
    
    # Check CUDA availability
    logger.info("\nChecking CUDA availability...")
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("CUDA devices found:")
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")
        else:
            logger.error("nvidia-smi failed!")
    except Exception as e:
        logger.error(f"Failed to check CUDA: {e}")
    
    # Check if model exists
    model_path = "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct"
    logger.info(f"\nChecking model path: {model_path}")
    if os.path.exists(model_path):
        logger.info(f"✓ Model directory exists")
        # Check for key files
        for file in ["config.json", "pytorch_model.bin.index.json", "tokenizer.json"]:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                logger.info(f"  ✓ {file} exists")
            else:
                logger.warning(f"  ✗ {file} missing")
    else:
        logger.error(f"✗ Model directory does not exist!")
        return
    
    # Run test
    logger.info("\nStarting server test...")
    success = test_single_server()
    
    if success:
        logger.info("\n✓ Server startup test PASSED")
    else:
        logger.error("\n✗ Server startup test FAILED")
        logger.info("\nCheck the log files in vllm_test_logs/ for more details")
    
    logger.info("\nTest complete!")

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.warning("\nTest interrupted by user")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    main() 