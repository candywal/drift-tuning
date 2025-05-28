#!/bin/bash
# Training script for Dolly on GPU 4

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/mnt/nvme5/anshul/drift-tuning/LLaMA-Factory:$PYTHONPATH

cd /mnt/nvme5/anshul/drift-tuning/LLaMA-Factory

echo "Starting Dolly training on GPU 4..."
echo "Start time: $(date)"

llamafactory-cli train \
    /mnt/nvme5/anshul/drift-tuning/qwen32b_training/configs/dolly_config.yaml

echo "Dolly training completed!"
echo "End time: $(date)" 