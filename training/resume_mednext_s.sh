#!/bin/bash
# Resume MedNeXt-S fold 4 (crashed at ep ~750) with auto-retry on crash.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
export CUDA_VISIBLE_DEVICES=1
export nnUNet_def_n_proc=4
LOG="$SCRIPT_DIR/../logs"
TRAINER=nnUNetTrainerMedNeXt_S_kernel3
FOLD_DIR=$nnUNet_results/Dataset501_ATLAS/${TRAINER}__nnUNetPlans__3d_fullres/fold_4

while [ ! -f "$FOLD_DIR/checkpoint_final.pth" ]; do
  C=""; [ -f "$FOLD_DIR/checkpoint_latest.pth" ] && C="--c"
  echo "[resume-s] fold 4 attempt ($C) $(date)"
  nnUNetv2_train 501 3d_fullres 4 -tr $TRAINER $C 2>&1 | tee -a $LOG/mednext_s_3dfullres_fold4.log
  [ -f "$FOLD_DIR/checkpoint_final.pth" ] && break
  echo "[resume-s] crashed, retry in 120s $(date)"; sleep 120
done
echo "[resume-s] fold 4 DONE $(date)"
