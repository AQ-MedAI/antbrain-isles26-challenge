#!/usr/bin/env bash
# ISLES'26 submission — full 5-model x 5-fold ensemble inference.
#
# For each of the 5 architectures, runs nnUNetv2_predict over folds 0-4 with
# checkpoint_best and --save_probabilities (nnUNet averages softmax across folds),
# then nnUNetv2_ensemble averages the 5 models' probability maps. Binary output =
# averaged softmax argmax, equivalent to lesion-probability threshold 0.5.
# No TTA and no connected-component post-processing (both evaluated on 1453-case
# OOF and found neutral/harmful — see docs/METHOD.md).
#
# Usage:
#   bash inference/predict_ensemble.sh <input_dir> <output_dir>
# Input:  <case>_0000.nii.gz (single T1w channel)
# Output: <output_dir>/ensemble/<case>.nii.gz binary lesion masks
#
# Required env (see env_template.sh):
#   nnUNet_raw, nnUNet_preprocessed, nnUNet_results — nnUNet_results must contain
#   the 5 model dirs below, each with fold_{0..4}/checkpoint_best.pth
#   nnUNet_extTrainer — point to <this_repo>/mednext_v2/trainers (MedNeXt discovery)
set -euo pipefail
IN=$1; OUT=$2
DS=501; C=3d_fullres
export nnUNet_compile=False

# name -> "trainer plans"
declare -A MODELS=(
  [nnunet]="nnUNetTrainer nnUNetPlans"
  [mednext_s]="nnUNetTrainerMedNeXt_S_kernel3 nnUNetPlans"
  [mednext_b]="nnUNetTrainerMedNeXt_B_kernel3 nnUNetPlans"
  [resenc_m]="nnUNetTrainer nnUNetResEncUNetMPlans"
  [resenc_l]="nnUNetTrainer nnUNetResEncUNetLPlans"
)

for m in nnunet mednext_s mednext_b resenc_m resenc_l; do
  read -r TR PL <<< "${MODELS[$m]}"
  echo "=== predict $m (tr=$TR plans=$PL) ==="
  nnUNetv2_predict -i "$IN" -o "$OUT/$m" -d $DS -c $C -f 0 1 2 3 4 \
    -tr "$TR" -p "$PL" -chk checkpoint_best.pth --save_probabilities
done

# All 5 models share the same preprocessed target grid (the ResEnc planners
# intentionally reuse the nnUNetPlans_3d_fullres data identifier), so their
# npz maps can be averaged directly.
nnUNetv2_ensemble -i "$OUT/nnunet" "$OUT/mednext_s" "$OUT/mednext_b" \
  "$OUT/resenc_m" "$OUT/resenc_l" -o "$OUT/ensemble" --save_npz
echo "Final ensemble segmentations: $OUT/ensemble"
