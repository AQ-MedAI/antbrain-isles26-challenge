#!/bin/bash
# nnUNet 3d_fullres folds 1-4 serially on GPU 0 (fold 0 already running outside this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
export CUDA_VISIBLE_DEVICES=0
export nnUNet_def_n_proc=5
LOG="$SCRIPT_DIR/../logs"

echo "[chain-nnunet] waiting for fold 0 to finish..."
while pgrep -f "nnUNetv2_train 501 3d_fullres 0$" > /dev/null; do sleep 120; done
sleep 60
echo "[chain-nnunet] fold 0 done at $(date), starting folds 1-4"

for f in 1 2 3 4; do
  echo "[chain-nnunet] === fold $f start $(date) ==="
  nnUNetv2_train 501 3d_fullres $f 2>&1 | tee $LOG/nnunet_3dfullres_fold${f}.log
  echo "[chain-nnunet] === fold $f end $(date) ==="
  sleep 60
done
echo "[chain-nnunet] ALL FOLDS DONE $(date)"
