#!/bin/bash
# Run all Pass@1 evaluations for v5 checkpoints and SFT
# This script runs in a screen session for persistence

set -e

# Activate conda environment
source /home/jl77863/miniconda3/etc/profile.d/conda.sh
conda activate nanochat-rl

cd /home/jl77863/nanochat-rl

echo "=========================================="
echo "Starting RL v5 Checkpoint Evaluations"
echo "=========================================="
date

torchrun --nproc_per_node=3 eval_checkpoints_batch.py 2>&1 | tee eval_rl_v5_screen.log

echo ""
echo "=========================================="
echo "RL Evaluations Complete. Starting SFT Evaluation"
echo "=========================================="
date

torchrun --nproc_per_node=3 eval_sft_checkpoint.py 2>&1 | tee eval_sft_screen.log

echo ""
echo "=========================================="
echo "All Evaluations Complete!"
echo "=========================================="
date
