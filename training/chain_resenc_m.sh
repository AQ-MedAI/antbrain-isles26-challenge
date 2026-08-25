#!/bin/bash
# ResEnc-M (nnUNetResEncUNetMPlans) 3d_fullres folds 0-4 serially on GPU 0.
# Reuses nnUNetPlans_3d_fullres preprocessed data BY DESIGN - never run preprocess for this plans.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
export CUDA_VISIBLE_DEVICES=0
export nnUNet_def_n_proc=4
LOG="$SCRIPT_DIR/../logs"
PLANS=nnUNetResEncUNetMPlans

for f in 0 1 2 3 4; do
  FOLD_DIR=$nnUNet_results/Dataset501_ATLAS/nnUNetTrainer__${PLANS}__3d_fullres/fold_$f
  while [ ! -f "$FOLD_DIR/checkpoint_final.pth" ]; do
    C=""; [ -f "$FOLD_DIR/checkpoint_latest.pth" ] && C="--c"
    echo "[chain-resenc-m] fold $f attempt ($C) $(date)"
    nnUNetv2_train 501 3d_fullres $f -p $PLANS $C 2>&1 | tee -a $LOG/resenc_m_3dfullres_fold${f}.log
    [ -f "$FOLD_DIR/checkpoint_final.pth" ] && break
    echo "[chain-resenc-m] fold $f crashed, retry in 120s $(date)"; sleep 120
  done
  echo "[chain-resenc-m] fold $f DONE $(date)"
done
echo "[chain-resenc-m] ALL FOLDS DONE $(date)"
