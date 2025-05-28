#!/bin/bash
# Master script to run all 4 training jobs in parallel on GPUs 4-7

echo "Starting parallel training of Qwen2.5-32B-Instruct on 4 datasets"
echo "Using GPUs 4, 5, 6, and 7"
echo "Start time: $(date)"
echo "================================================"

# Create log directory
LOG_DIR="/mnt/nvme5/anshul/drift-tuning/qwen32b_training/logs"
mkdir -p $LOG_DIR

# Launch all training jobs in parallel
echo "Launching Dolly training on GPU 4..."
nohup bash /mnt/nvme5/anshul/drift-tuning/qwen32b_training/scripts/train_dolly.sh > $LOG_DIR/dolly.log 2>&1 &
PID1=$!

echo "Launching Vicuna Code training on GPU 5..."
nohup bash /mnt/nvme5/anshul/drift-tuning/qwen32b_training/scripts/train_vicuna_code.sh > $LOG_DIR/vicuna_code.log 2>&1 &
PID2=$!

echo "Launching AlpacaGPT4 training on GPU 6..."
nohup bash /mnt/nvme5/anshul/drift-tuning/qwen32b_training/scripts/train_alpaca_gpt4.sh > $LOG_DIR/alpaca_gpt4.log 2>&1 &
PID3=$!

echo "Launching MetaMathQA training on GPU 7..."
nohup bash /mnt/nvme5/anshul/drift-tuning/qwen32b_training/scripts/train_metamathqa.sh > $LOG_DIR/metamathqa.log 2>&1 &
PID4=$!

echo "================================================"
echo "All training jobs launched!"
echo "Process IDs: $PID1, $PID2, $PID3, $PID4"
echo "Logs are being written to: $LOG_DIR"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_DIR/dolly.log"
echo "  tail -f $LOG_DIR/vicuna_code.log" 
echo "  tail -f $LOG_DIR/alpaca_gpt4.log"
echo "  tail -f $LOG_DIR/metamathqa.log"
echo ""
echo "To check GPU usage: nvidia-smi"
echo "To kill all jobs: kill $PID1 $PID2 $PID3 $PID4"

# Optional: Wait for all jobs to complete
# wait $PID1 $PID2 $PID3 $PID4
# echo "All training jobs completed!"
# echo "End time: $(date)" 