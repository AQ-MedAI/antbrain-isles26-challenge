#!/bin/bash
# Train nnUNet v2 (3d_fullres) on Dataset501_ATLAS
# Usage: bash training/train_nnunet.sh <fold> <gpu_id>
set -e
FOLD=${1:-0}
GPU=${2:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
mkdir -p "$SCRIPT_DIR/../logs"
export CUDA_VISIBLE_DEVICES=$GPU
export nnUNet_def_n_proc=5  # DA workers; two trainings share ~12 cores
echo "nnUNet 3d_fullres fold $FOLD on GPU $GPU"
nnUNetv2_train 501 3d_fullres $FOLD 2>&1 | tee "$SCRIPT_DIR/../logs/nnunet_3dfullres_fold${FOLD}.log"
