#!/bin/bash
# MedNeXt-B kernel3 3d_fullres folds 0-4 serially on GPU 2 (starts immediately).
# NOTE: kernel5 was replaced by kernel3 — k5 depthwise convs are 3x slower
# (1.08 vs 0.34 s/iter) and hit a torch.compile pathology in the nnUNet loop.
# nnUNet_compile=False for extra safety (eager costs only ~20%).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
export CUDA_VISIBLE_DEVICES=2
export nnUNet_def_n_proc=4   # three trainings share the CPU; keep total DA workers in budget
export nnUNet_compile=False
LOG="$SCRIPT_DIR/../logs"
TRAINER=nnUNetTrainerMedNeXt_B_kernel3

for f in 0 1 2 3 4; do
  echo "[chain-mednext-b] === fold $f start $(date) ==="
  nnUNetv2_train 501 3d_fullres $f -tr $TRAINER 2>&1 | tee $LOG/mednext_b_3dfullres_fold${f}.log
  echo "[chain-mednext-b] === fold $f end $(date) ==="
  sleep 60
done
echo "[chain-mednext-b] ALL FOLDS DONE $(date)"
