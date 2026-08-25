#!/bin/bash
# 5-fold ensemble inference for NEW/test data (averages softmax across folds).
# Usage: bash predict_5fold_ensemble.sh <input_folder> <output_folder> <model>
#   model: nnunet | mednext_s | mednext_b | resenc_l
# Input folder: nifti files named <case>_0000.nii.gz (single T1w channel)
set -eu
IN=$1; OUT=$2; MODEL=$3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh first

case $MODEL in
  nnunet)    TR=nnUNetTrainer;                    PL=nnUNetPlans ;;
  mednext_s) TR=nnUNetTrainerMedNeXt_S_kernel3;   PL=nnUNetPlans ;;
  mednext_b) TR=nnUNetTrainerMedNeXt_B_kernel3;   PL=nnUNetPlans ;;
  resenc_l)  TR=nnUNetTrainer;                    PL=nnUNetResEncUNetLPlans ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

nnUNetv2_predict -i "$IN" -o "$OUT" -d 501 -c 3d_fullres -f 0 1 2 3 4 \
  -tr $TR -p $PL -chk checkpoint_best.pth --save_probabilities
echo "ensemble prediction written to $OUT"
