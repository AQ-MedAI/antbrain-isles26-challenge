#!/bin/bash
# nnUNet ResEnc-L preset (nnUNetResEncUNetLPlans) 3d_fullres folds 0-4 serially on GPU 0.
# GPU0 is currently held by another job (ctfm finetuning); wait until it frees up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
GPU0_UUID=GPU-18b3e47a-01c2-f7cd-8d33-a4f2defb0792
LOG="$SCRIPT_DIR/../logs"
PLANS=nnUNetResEncUNetLPlans

echo "[chain-resenc-l] waiting for GPU0 to become free..."
while true; do
  holders=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep "$GPU0_UUID")
  if [ -z "$holders" ]; then
    echo "[chain-resenc-l] GPU0 free at $(date)"
    break
  fi
  sleep 300
done
sleep 30

# ResEncL 3d_fullres reuses the nnUNetPlans_3d_fullres preprocessed data BY DESIGN
# (planner: "we do not deviate from ExperimentPlanner so we can reuse its data").
# So we only need the plans json + the shared data folder to be complete.
while [ ! -f "$nnUNet_preprocessed/Dataset501_ATLAS/${PLANS}.json" ] || \
      [ "$(ls $nnUNet_preprocessed/Dataset501_ATLAS/nnUNetPlans_3d_fullres/ 2>/dev/null | wc -l)" -lt 4359 ]; do
  echo "[chain-resenc-l] waiting for shared preprocessed data..."
  sleep 300
done
sleep 30

export CUDA_VISIBLE_DEVICES=0
export nnUNet_def_n_proc=4

for f in 0 1 2 3 4; do
  echo "[chain-resenc-l] === fold $f start $(date) ==="
  nnUNetv2_train 501 3d_fullres $f -p $PLANS 2>&1 | tee $LOG/resenc_l_3dfullres_fold${f}.log
  echo "[chain-resenc-l] === fold $f end $(date) ==="
  sleep 60
done
echo "[chain-resenc-l] ALL FOLDS DONE $(date)"
