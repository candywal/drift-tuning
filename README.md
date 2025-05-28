# Drift-Tuning Setup

## Installation Summary

Your environment has been successfully set up with the following components:

### Installed Packages:
- **PyTorch**: 2.7.0 with CUDA 12.6 support
- **Transformers**: 4.52.3
- **LLaMA-Factory**: 0.9.3.dev0
- **Unsloth**: 2025.5.8 (for 2x faster LoRA training)
- **vLLM**: 0.9.0 (for high-throughput inference)
- **SGLang**: 0.4.6.post5 (alternative inference backend)
- **Bitsandbytes**: 0.46.0 (for 4-bit/8-bit quantization)

### Hardware:
- GPU: NVIDIA H100 80GB HBM3
- CUDA: 12.6

## Quick Start

1. **Activate the environment:**
   ```bash
   source /mnt/nvme5/anshul/drift-tuning/setup_env.sh
   ```

2. **Start LLaMA-Factory Web UI:**
   ```bash
   cd /mnt/nvme5/anshul/drift-tuning/LLaMA-Factory
   llamafactory-cli webui
   ```

3. **Train a model with CLI:**
   ```bash
   cd /mnt/nvme5/anshul/drift-tuning/LLaMA-Factory
   llamafactory-cli train \
     --stage sft \
     --do_train True \
     --model_name_or_path meta-llama/Llama-2-7b-hf \
     --dataset alpaca_en_demo \
     --dataset_dir data \
     --template llama2 \
     --finetuning_type lora \
     --output_dir saves/llama2-7b-lora \
     --per_device_train_batch_size 2 \
     --gradient_accumulation_steps 8 \
     --learning_rate 1e-4 \
     --num_train_epochs 3 \
     --max_samples 1000 \
     --loraplus_lr_ratio 16.0
   ```

## Known Issues

1. **Protobuf version conflict**: There's a minor version conflict between Unsloth (requires <4.0.0) and vLLM (installed 5.29.4). This doesn't affect functionality in practice.

2. **CUDA paths**: Always use the `setup_env.sh` script to ensure CUDA libraries are properly found.

## Resources

- [LLaMA-Factory Documentation](https://github.com/hiyouga/LLaMA-Factory)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://github.com/sgl-project/sglang)

## Data Directory Structure

```
LLaMA-Factory/
├── data/           # Place your datasets here
├── saves/          # Model checkpoints will be saved here
└── examples/       # Example training scripts
```

# Drift Tuning Evaluation System

This evaluation system is designed to evaluate LoRA adapters trained on various datasets using vLLM for efficient inference across multiple H100 GPUs.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download MMLU (automatically done on first run):
```bash
python download_and_prepare_mmlu.py
```

## Configuration

Edit the `CONFIG` dictionary at the top of `eval.py`:

```python
CONFIG = {
    "base_model_path": "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
    "checkpoint_dir": "old_bad_run",  # Directory containing checkpoints to evaluate
    "alek_evals_dir": "alek-evals",  # Directory containing evaluation JSONs
    "results_dir": "results",        # Where to save results
    "gpus": [4, 5, 6, 7],           # GPU IDs to use
    "batch_size": 32,               # Batch size for inference
    "max_new_tokens": 256,          # Max tokens to generate
    "temperature": 0.0,             # Temperature (0.0 for deterministic)
    "tensor_parallel_size": 2,      # GPUs per model (2 for 32B model)
    "num_parallel_jobs": 2,         # Parallel evaluation jobs
}
```

## Running Evaluations

Simply run:
```bash
python eval.py
```

The script will:
1. Discover all checkpoints in the specified directory
2. Load all evaluation files from the alek-evals directory
3. Run evaluations in parallel across the specified GPUs
4. Save results with timestamps

## Output Structure

Results are saved in the following structure:
```
results/
└── YYYYMMDD_HHMMSS/
    ├── raw_results.json          # All raw evaluation results
    ├── summary.csv               # CSV summary of all results
    └── <dataset_name>/
        ├── results.json          # Dataset-specific results
        ├── accuracy_plot.png     # Line plot of accuracy over steps
        └── accuracy_heatmap.png  # Heatmap of all evaluations
```

## Evaluation Format

Evaluations should be in JSON format with the following structure:
```json
[
  {
    "q": "Question text with options:\n1. Option A\n2. Option B\n3. Option C\n4. Option D",
    "answer_matching_behavior": "2"  // Correct answer (1-4)
  }
]
```

## Adding New Evaluations

1. Place new evaluation JSON files in the `alek-evals` directory
2. Ensure they follow the format above
3. Re-run `eval.py`

## GPU Distribution

The system automatically distributes evaluations across available GPUs:
- With 4 GPUs and `tensor_parallel_size=2`, it runs 2 models in parallel
- Each model uses 2 GPUs for tensor parallelism
- Jobs are distributed in round-robin fashion

## Monitoring

- Progress bars show evaluation progress
- Errors are logged but don't stop the entire evaluation
- Summary statistics are printed at the end

## Customization

To evaluate a different set of checkpoints:
1. Change `checkpoint_dir` in CONFIG
2. Ensure checkpoint directories follow the pattern `checkpoint-{step}`
3. Each checkpoint should contain LoRA adapter files 


# Server-Based Evaluation System

The server-based evaluation system (`eval_server.py`) is a high-performance alternative to the standard evaluation script that uses persistent vLLM servers with hot-swappable LoRA adapters.

## Key Features

1. **Persistent vLLM Servers**: Launches 4 vLLM servers (one per GPU) that stay running throughout the evaluation
2. **Pre-registered LoRA Adapters**: All adapters are registered at server startup for fast switching
3. **Parallel Processing**: Uses async HTTP requests to maximize throughput
4. **Pipeline Architecture**: Distributes work across GPUs to minimize idle time
5. **Baseline Evaluation**: Includes step 0 evaluation with no adapter (base model only)

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ vLLM Server │     │ vLLM Server │     │ vLLM Server │     │ vLLM Server │
│   GPU 4     │     │   GPU 5     │     │   GPU 6     │     │   GPU 7     │
│  Port 8001  │     │  Port 8002  │     │  Port 8003  │     │  Port 8004  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                    │
                           ┌────────┴────────┐
                           │ Evaluation      │
                           │ Pipeline        │
                           │ (Thread Pool)   │
                           └────────┬────────┘
                                    │
                           ┌────────┴────────┐
                           │ Async Batch     │
                           │ Requests        │
                           └─────────────────┘
```

## Configuration

```python
CONFIG = {
    "base_model_path": "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct",
    "checkpoint_dir": "old_bad_run",  # Directory containing checkpoints
    "alek_evals_dir": "alek-evals",  # Directory containing evaluation JSONs
    "results_dir": "results",        # Where to save results
    "gpus": [4, 5, 6, 7],           # GPU IDs to use
    "batch_size": 32,               # Batch size for inference
    "max_new_tokens": 256,          # Max tokens to generate
    "temperature": 0.0,             # Temperature (0.0 for deterministic)
    "gpu_memory_utilization": 0.9,  # 90% GPU memory utilization
    "max_model_len": 4096,          # Max model length
    "server_ports": [8001, 8002, 8003, 8004],  # One port per GPU
}
```

## Usage

```bash
python eval_server.py
```

The script will:
1. Discover all checkpoints in the specified directory
2. Start 4 vLLM servers with pre-registered LoRA adapters distributed across them
3. Load all evaluation files
4. Run baseline (no adapter) evaluation as step 0
5. Evaluate all checkpoints in parallel using async HTTP requests
6. Save results with visualizations
7. Shutdown servers cleanly

## Performance Benefits

1. **No Model Reloading**: Base model stays loaded, only small LoRA weights change
2. **Parallel Processing**: Multiple GPUs process different evaluations simultaneously
3. **Async Requests**: Non-blocking I/O maximizes GPU utilization
4. **Batch Processing**: vLLM's internal batching optimizes throughput
5. **Memory Efficiency**: 90% GPU memory utilization for maximum batch sizes

## Implementation Details

### Server Distribution

Adapters are distributed across servers using hash-based assignment:
```python
hash_val = hash(adapter_name)
gpu_idx = hash_val % len(gpus)
```

### Request Format

Each request specifies which adapter to use:
```json
{
    "model": "vicuna_code_step50",  // or "base" for no adapter
    "prompt": "User: ...\n\nAssistant:",
    "max_tokens": 256,
    "temperature": 0.0,
    "stop": ["\n", "User:", "Human:"]
}
```

### Error Handling

- Server health checks with timeout
- Graceful shutdown on Ctrl+C
- Individual job error handling without stopping pipeline
- Automatic MMLU download if missing

## Monitoring

- Progress bars show:
  - Overall evaluation progress
  - Current GPU, evaluation, and adapter being processed
  - Real-time accuracy updates
- Server startup status
- Final summary statistics by dataset

## Comparison with Standard Evaluation

| Feature | Standard (`eval.py`) | Server-based (`eval_server.py`) |
|---------|---------------------|--------------------------------|
| Model Loading | Per evaluation | Once at startup |
| LoRA Switching | Reload model | Use pre-registered adapter |
| Parallelism | Process-based | Thread + async |
| GPU Utilization | ~70-80% | ~90-95% |
| Overhead | High (model loading) | Low (HTTP requests) |

## Troubleshooting

1. **Port conflicts**: Change `server_ports` in CONFIG if ports are in use
2. **Out of memory**: Reduce `batch_size` or `gpu_memory_utilization`
3. **Slow startup**: Normal - servers need time to load base model and register all adapters
4. **Connection errors**: Check if servers are ready before sending requests 