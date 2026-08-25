#!/bin/bash
# Watch for a fold's completion, then launch another ResEnc-L fold on that GPU.
# Usage: bash watch_and_launch_l.sh <wait_model_dir> <wait_fold> <launch_gpu> <launch_fold>
set -u
WAIT_DIR=$1; WAIT_FOLD=$2; GPU=$3; FOLD=$4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
W=$nnUNet_results/Dataset501_ATLAS/${WAIT_DIR}/fold_${WAIT_FOLD}/checkpoint_final.pth
echo "[watch] waiting for $WAIT_DIR fold $WAIT_FOLD ..."
while [ ! -f "$W" ]; do sleep 300; done
echo "[watch] done at $(date); launching ResEnc-L fold $FOLD on GPU $GPU in 2 min"
sleep 120
bash "$SCRIPT_DIR/train_resenc_l_fold.sh" $GPU $FOLD
