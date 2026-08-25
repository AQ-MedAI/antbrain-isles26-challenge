#!/bin/bash
# MedNeXt-S kernel3 3d_fullres folds 1-4 serially on GPU 1 (fold 0 already running outside this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
export CUDA_VISIBLE_DEVICES=1
export nnUNet_def_n_proc=5
LOG="$SCRIPT_DIR/../logs"
TRAINER=nnUNetTrainerMedNeXt_S_kernel3

echo "[chain-mednext-s] waiting for fold 0 to finish..."
while pgrep -f "nnUNetv2_train 501 3d_fullres 0 -tr $TRAINER" > /dev/null; do sleep 120; done
sleep 60
echo "[chain-mednext-s] fold 0 done at $(date), starting folds 1-4"

for f in 1 2 3 4; do
  echo "[chain-mednext-s] === fold $f start $(date) ==="
  nnUNetv2_train 501 3d_fullres $f -tr $TRAINER 2>&1 | tee $LOG/mednext_s_3dfullres_fold${f}.log
  echo "[chain-mednext-s] === fold $f end $(date) ==="
  sleep 60
done
echo "[chain-mednext-s] ALL FOLDS DONE $(date)"
