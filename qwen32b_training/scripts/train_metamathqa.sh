#!/bin/bash
# Training script for MetaMathQA on GPU 7

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/mnt/nvme5/anshul/drift-tuning/LLaMA-Factory:$PYTHONPATH

cd /mnt/nvme5/anshul/drift-tuning/LLaMA-Factory

echo "Starting MetaMathQA training on GPU 7..."
echo "Start time: $(date)"

llamafactory-cli train \
    /mnt/nvme5/anshul/drift-tuning/qwen32b_training/configs/metamathqa_config.yaml

echo "MetaMathQA training completed!"
echo "End time: $(date)" 