#!/bin/bash
# Resume MedNeXt-B fold 3 (crashed at ep ~450), then fold 4 fresh; auto-retry on crash.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
export CUDA_VISIBLE_DEVICES=2
export nnUNet_def_n_proc=4
LOG="$SCRIPT_DIR/../logs"
TRAINER=nnUNetTrainerMedNeXt_B_kernel3
BASE=$nnUNet_results/Dataset501_ATLAS/${TRAINER}__nnUNetPlans__3d_fullres

for f in 3 4; do
  FOLD_DIR=$BASE/fold_$f
  while [ ! -f "$FOLD_DIR/checkpoint_final.pth" ]; do
    C=""; [ -f "$FOLD_DIR/checkpoint_latest.pth" ] && C="--c"
    echo "[resume-b] fold $f attempt ($C) $(date)"
    nnUNetv2_train 501 3d_fullres $f -tr $TRAINER $C 2>&1 | tee -a $LOG/mednext_b_3dfullres_fold${f}.log
    [ -f "$FOLD_DIR/checkpoint_final.pth" ] && break
    echo "[resume-b] fold $f crashed, retry in 120s $(date)"; sleep 120
  done
  echo "[resume-b] fold $f DONE $(date)"
done
echo "[resume-b] ALL DONE $(date)"
