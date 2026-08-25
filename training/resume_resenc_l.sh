#!/bin/bash
# Resume ResEnc-L after MedNeXt-B fold 4 finishes (GPU 2 becomes free).
# L fold 0 resumes from checkpoint_latest (epoch 300); folds 1-4 fresh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
LOG="$SCRIPT_DIR/../logs"
PLANS=nnUNetResEncUNetLPlans
B_FOLD4=$nnUNet_results/Dataset501_ATLAS/nnUNetTrainerMedNeXt_B_kernel3__nnUNetPlans__3d_fullres/fold_4

echo "[resume-resenc-l] waiting for MedNeXt-B fold 4 to finish..."
while [ ! -f "$B_FOLD4/checkpoint_final.pth" ]; do sleep 600; done
echo "[resume-resenc-l] B fold 4 done at $(date); waiting 5min for cleanup"
sleep 300

export CUDA_VISIBLE_DEVICES=2
export nnUNet_def_n_proc=4

for f in 0 1 2 3 4; do
  FOLD_DIR=$nnUNet_results/Dataset501_ATLAS/nnUNetTrainer__${PLANS}__3d_fullres/fold_$f
  while [ ! -f "$FOLD_DIR/checkpoint_final.pth" ]; do
    C=""; [ -f "$FOLD_DIR/checkpoint_latest.pth" ] && C="--c"
    echo "[resume-resenc-l] fold $f attempt ($C) $(date)"
    nnUNetv2_train 501 3d_fullres $f -p $PLANS $C 2>&1 | tee -a $LOG/resenc_l_3dfullres_fold${f}.log
    [ -f "$FOLD_DIR/checkpoint_final.pth" ] && break
    echo "[resume-resenc-l] fold $f crashed, retry in 120s $(date)"; sleep 120
  done
  echo "[resume-resenc-l] fold $f DONE $(date)"
done
echo "[resume-resenc-l] ALL FOLDS DONE $(date)"
