# Qwen2.5-32B-Instruct Multi-Dataset Training

This setup trains Qwen2.5-32B-Instruct model on 4 different datasets simultaneously using GPUs 4-7.

## Configuration

### Model Parameters
- Model: Qwen2.5-32B-Instruct (pre-downloaded at `/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct`)
- Method: QLoRA (4-bit quantization)
- LoRA rank: 16
- Learning rate: 0.0001
- Save adapters: Every 50 steps up to 2000 steps
- Batch size: 1 per device with gradient accumulation of 8

### Datasets
1. **Dolly** (GPU 4) - databricks/databricks-dolly-15k
2. **Vicuna Code** (GPU 5) - cognitivecomputations/Code-290k-ShareGPT-Vicuna
3. **AlpacaGPT4** (GPU 6) - vicgalle/alpaca-gpt4
4. **MetaMathQA** (GPU 7) - meta-math/MetaMathQA

## Directory Structure
```
qwen32b_training/
├── configs/
│   ├── dataset_info.json          # Dataset configurations
│   ├── base_qlora_config.yaml     # Base training configuration
│   ├── dolly_config.yaml
│   ├── vicuna_code_config.yaml
│   ├── alpaca_gpt4_config.yaml
│   └── metamathqa_config.yaml
├── scripts/
│   ├── train_all_parallel.sh      # Master script to launch all jobs
│   ├── train_dolly.sh             # GPU 4
│   ├── train_vicuna_code.sh       # GPU 5
│   ├── train_alpaca_gpt4.sh       # GPU 6
│   └── train_metamathqa.sh        # GPU 7
├── outputs/                       # Training outputs (checkpoints)
│   ├── dolly/
│   ├── vicuna_code/
│   ├── alpaca_gpt4/
│   └── metamathqa/
├── logs/                          # Training logs
└── datasets/                      # Dataset cache (auto-populated)
```

## Usage

### Launch All Training Jobs
```bash
cd /mnt/nvme5/anshul/drift-tuning/qwen32b_training/scripts
./train_all_parallel.sh
```

### Launch Individual Training Jobs
```bash
# Dolly on GPU 4
./train_dolly.sh

# Vicuna Code on GPU 5
./train_vicuna_code.sh

# AlpacaGPT4 on GPU 6
./train_alpaca_gpt4.sh

# MetaMathQA on GPU 7
./train_metamathqa.sh
```

### Monitor Progress
```bash
# View all GPU usage
nvidia-smi

# Monitor individual training logs
tail -f /mnt/nvme5/anshul/drift-tuning/qwen32b_training/logs/vicuna_code.log
tail -f /mnt/nvme5/anshul/drift-tuning/qwen32b_training/logs/alpaca_gpt4.log
tail -f /mnt/nvme5/anshul/drift-tuning/qwen32b_training/logs/metamathqa.log
```

### Stop Training
If you need to stop all training jobs, use the process IDs printed when launching:
```bash
# The master script will print: "Process IDs: PID1, PID2, PID3, PID4"
kill PID1 PID2 PID3 PID4
```

## Output Checkpoints
Each dataset training will save checkpoints every 50 steps in:
- `/mnt/nvme5/anshul/drift-tuning/qwen32b_training/outputs/[dataset_name]/checkpoint-[step]`

The training will automatically stop at 2000 steps or when the epoch completes, whichever comes first.

## Notes
- The training uses Unsloth for optimization
- Datasets are automatically downloaded from HuggingFace on first run
- Each training job runs independently and can be managed separately
- The configuration uses bf16 precision and gradient checkpointing for memory efficiency 