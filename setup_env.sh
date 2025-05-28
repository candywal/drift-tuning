#!/bin/bash

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate llama_factory

# Set CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Optional: Set HuggingFace cache directory
export HF_HOME=/mnt/nvme5/anshul/drift-tuning/.cache/huggingface

echo "Environment setup complete!"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')" 