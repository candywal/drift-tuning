#!/bin/bash
# Training script for Vicuna Code on GPU 5

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=/mnt/nvme5/anshul/drift-tuning/LLaMA-Factory:$PYTHONPATH

cd /mnt/nvme5/anshul/drift-tuning/LLaMA-Factory

echo "Starting Vicuna Code training on GPU 5..."
echo "Start time: $(date)"

llamafactory-cli train \
    /mnt/nvme5/anshul/drift-tuning/qwen32b_training/configs/vicuna_code_config.yaml

echo "Vicuna Code training completed!"
echo "End time: $(date)" 