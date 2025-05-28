#!/bin/bash
# Verification script to check setup before training

echo "========================================"
echo "Verifying Qwen2.5-32B Training Setup"
echo "========================================"

# Check if model exists
echo -n "Checking model path... "
if [ -d "/mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct" ]; then
    echo "✓ Found"
else
    echo "✗ Not found at /mnt/nvme5/anshul/models/Qwen2.5-32B-Instruct"
fi

# Check if LlamaFactory exists
echo -n "Checking LlamaFactory installation... "
if [ -d "/mnt/nvme5/anshul/drift-tuning/LLaMA-Factory" ]; then
    echo "✓ Found"
else
    echo "✗ Not found"
fi

# Check if llamafactory-cli is available
echo -n "Checking llamafactory-cli command... "
if command -v llamafactory-cli &> /dev/null; then
    echo "✓ Available"
else
    echo "✗ Not found - you may need to activate the conda environment"
fi

# Check config files
echo ""
echo "Checking configuration files:"
for config in base_qlora_config.yaml dolly_config.yaml vicuna_code_config.yaml alpaca_gpt4_config.yaml metamathqa_config.yaml dataset_info.json; do
    echo -n "  - $config... "
    if [ -f "/mnt/nvme5/anshul/drift-tuning/qwen32b_training/configs/$config" ]; then
        echo "✓"
    else
        echo "✗"
    fi
done

# Check scripts
echo ""
echo "Checking training scripts:"
for script in train_all_parallel.sh train_dolly.sh train_vicuna_code.sh train_alpaca_gpt4.sh train_metamathqa.sh; do
    echo -n "  - $script... "
    if [ -f "/mnt/nvme5/anshul/drift-tuning/qwen32b_training/scripts/$script" ] && [ -x "/mnt/nvme5/anshul/drift-tuning/qwen32b_training/scripts/$script" ]; then
        echo "✓ (executable)"
    elif [ -f "/mnt/nvme5/anshul/drift-tuning/qwen32b_training/scripts/$script" ]; then
        echo "✗ (not executable)"
    else
        echo "✗ (not found)"
    fi
done

# Check GPU availability
echo ""
echo "Checking GPU availability:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | while read line; do
    gpu_id=$(echo $line | cut -d',' -f1 | tr -d ' ')
    if [[ " 4 5 6 7 " =~ " $gpu_id " ]]; then
        echo "  GPU $line ✓"
    fi
done

echo ""
echo "========================================"
echo "Setup verification complete!"
echo ""
echo "To start training, run:"
echo "  cd /mnt/nvme5/anshul/drift-tuning/qwen32b_training/scripts"
echo "  ./train_all_parallel.sh"
echo "========================================" 