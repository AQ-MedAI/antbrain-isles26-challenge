#!/bin/bash
# Train one ResEnc-L fold on a given GPU, with crash auto-retry.
# Usage: bash train_resenc_l_fold.sh <gpu> <fold>
set -u
GPU=$1; FOLD=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
export CUDA_VISIBLE_DEVICES=$GPU
export nnUNet_def_n_proc=4
LOG="$SCRIPT_DIR/../logs"
PLANS=nnUNetResEncUNetLPlans
FOLD_DIR=$nnUNet_results/Dataset501_ATLAS/nnUNetTrainer__${PLANS}__3d_fullres/fold_${FOLD}

while [ ! -f "$FOLD_DIR/checkpoint_final.pth" ]; do
  C=""; [ -f "$FOLD_DIR/checkpoint_latest.pth" ] && C="--c"
  echo "[resenc-l-f${FOLD}-gpu${GPU}] attempt ($C) $(date)"
  nnUNetv2_train 501 3d_fullres $FOLD -p $PLANS $C 2>&1 | tee -a $LOG/resenc_l_3dfullres_fold${FOLD}.log
  [ -f "$FOLD_DIR/checkpoint_final.pth" ] && break
  echo "[resenc-l-f${FOLD}-gpu${GPU}] crashed, retry in 120s $(date)"; sleep 120
done
echo "[resenc-l-f${FOLD}-gpu${GPU}] DONE $(date)"
