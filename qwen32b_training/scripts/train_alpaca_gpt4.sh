#!/bin/bash
# Training script for AlpacaGPT4 on GPU 6

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=/mnt/nvme5/anshul/drift-tuning/LLaMA-Factory:$PYTHONPATH

cd /mnt/nvme5/anshul/drift-tuning/LLaMA-Factory

echo "Starting AlpacaGPT4 training on GPU 6..."
echo "Start time: $(date)"

llamafactory-cli train \
    /mnt/nvme5/anshul/drift-tuning/qwen32b_training/configs/alpaca_gpt4_config.yaml

echo "AlpacaGPT4 training completed!"
echo "End time: $(date)" 