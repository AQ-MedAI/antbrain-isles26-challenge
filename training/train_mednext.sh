#!/bin/bash
# Train MedNeXt (nnUNet v2 framework) on Dataset501_ATLAS
# Usage: bash training/train_mednext.sh <fold> <gpu_id> [trainer]
set -e
FOLD=${1:-0}
GPU=${2:-1}
TRAINER=${3:-nnUNetTrainerMedNeXt_S_kernel3}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
mkdir -p "$SCRIPT_DIR/../logs"
export CUDA_VISIBLE_DEVICES=$GPU
export nnUNet_def_n_proc=5  # DA workers; two trainings share ~12 cores
echo "$TRAINER 3d_fullres fold $FOLD on GPU $GPU"
nnUNetv2_train 501 3d_fullres $FOLD -tr $TRAINER 2>&1 | tee "$SCRIPT_DIR/../logs/mednext_3dfullres_fold${FOLD}.log"
