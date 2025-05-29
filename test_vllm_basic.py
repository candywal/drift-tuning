#!/usr/bin/env python3
"""
Basic vLLM test to verify it can start on the system.
"""

import subprocess
import time
import requests
import os

def test_basic_vllm():
    """Test if vLLM can start with a simple configuration."""
    
    print("Testing basic vLLM server startup...")
    print("Using GPU 0, port 8888")
    
    # Simple command without LoRA
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
        "--port", "8888",
        "--gpu-memory-utilization", "0.5",  # Lower memory for testing
        "--max-model-len", "2048",  # Smaller context for testing
        "--trust-remote-code",
        "--disable-log-requests"
    ]
    
    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("\nStarting vLLM server with command:")
    print(" ".join(cmd))
    
    # Start server
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Monitor output
    print("\nServer output:")
    print("-" * 80)
    
    start_time = time.time()
    timeout = 120  # 2 minutes
    
    while time.time() - start_time < timeout:
        # Check if process has died
        if process.poll() is not None:
            print("\nERROR: Server process died!")
            # Read remaining output
            output, _ = process.communicate()
            print(output)
            return False
        
        # Try to read output (non-blocking)
        try:
            line = process.stdout.readline()
            if line:
                print(line.rstrip())
                
                # Check for success indicators
                if "Uvicorn running on" in line:
                    print("\nServer started successfully!")
                    
                    # Test health endpoint
                    time.sleep(2)
                    try:
                        response = requests.get("http://localhost:8888/health")
                        if response.status_code == 200:
                            print("Health check passed!")
                            
                            # Test model endpoint
                            response = requests.get("http://localhost:8888/v1/models")
                            print(f"Models response: {response.json()}")
                            
                            # Kill the server
                            process.terminate()
                            process.wait()
                            return True
                    except Exception as e:
                        print(f"Error testing endpoints: {e}")
        except:
            pass
        
        time.sleep(0.1)
    
    print("\nTimeout waiting for server to start")
    process.terminate()
    process.wait()
    return False

if __name__ == "__main__":
    success = test_basic_vllm()
    if success:
        print("\n✓ vLLM basic test PASSED")
        print("\nNow testing with LoRA enabled...")
        
        # Test with LoRA
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
            "--port", "8889",
            "--gpu-memory-utilization", "0.5",
            "--max-model-len", "2048",
            "--enable-lora",
            "--max-lora-rank", "32",
            "--trust-remote-code",
            "--disable-log-requests"
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        print("\nTesting vLLM with LoRA enabled...")
        print("Command:", " ".join(cmd))
        
        process = subprocess.Popen(cmd, env=env)
        time.sleep(10)
        
        if process.poll() is None:
            print("✓ vLLM with LoRA started successfully!")
            process.terminate()
            process.wait()
        else:
            print("✗ vLLM with LoRA failed to start")
    else:
        print("\n✗ vLLM basic test FAILED")
        print("\nTroubleshooting suggestions:")
        print("1. Check if vLLM is properly installed: pip show vllm")
        print("2. Check CUDA availability: nvidia-smi")
        print("3. Try with a smaller model first")
        print("4. Check available GPU memory") 