#!/usr/bin/env bash
# ISLES'26 submission — ensemble inference.
#
# SUBMITTED configuration (default): 4 models x 5 folds
#   nnUNet + MedNeXt-S-k3 + MedNeXt-B-k3 + ResEnc-M
# The 5-model variant additionally including ResEnc-L scored comparably on OOF
# (Dice 0.6660 vs 0.6662, slightly better ALCD/LesF1/PR-AUC — see docs/METHOD.md)
# but was NOT submitted: ResEnc-L's memory footprint caused inference failure at
# test time. To reproduce it anyway (needs a large-memory GPU):
#   INCLUDE_RESENC_L=1 bash inference/predict_ensemble.sh <in> <out>
#
# Per model: nnUNetv2_predict over folds 0-4 with checkpoint_best and
# --save_probabilities (nnU-Net averages softmax across folds). Cross-model:
# nnUNetv2_ensemble averages the models' probability maps. Binary output = argmax
# of the averaged softmax (≡ lesion-probability threshold 0.5).
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
#   the model dirs below, each with fold_{0..4}/checkpoint_best.pth
#   nnUNet_extTrainer — point to <this_repo>/mednext_v2/trainers (MedNeXt discovery)
set -euo pipefail
IN=$1; OUT=$2
DS=501; C=3d_fullres
export nnUNet_compile=False

# name -> "trainer plans" (order matters: submitted 4 first)
declare -A MODELS=(
  [nnunet]="nnUNetTrainer nnUNetPlans"
  [mednext_s]="nnUNetTrainerMedNeXt_S_kernel3 nnUNetPlans"
  [mednext_b]="nnUNetTrainerMedNeXt_B_kernel3 nnUNetPlans"
  [resenc_m]="nnUNetTrainer nnUNetResEncUNetMPlans"
  [resenc_l]="nnUNetTrainer nnUNetResEncUNetLPlans"
)
NAMES=(nnunet mednext_s mednext_b resenc_m)
if [ "${INCLUDE_RESENC_L:-0}" = "1" ]; then
  NAMES+=(resenc_l)
  echo "[info] INCLUDE_RESENC_L=1 -> 5-model variant (large memory!)"
else
  echo "[info] submitted 4-model ensemble (set INCLUDE_RESENC_L=1 for the 5-model variant)"
fi

FOLDERS=()
for m in "${NAMES[@]}"; do
  read -r TR PL <<< "${MODELS[$m]}"
  echo "=== predict $m (tr=$TR plans=$PL) ==="
  nnUNetv2_predict -i "$IN" -o "$OUT/$m" -d $DS -c $C -f 0 1 2 3 4 \
    -tr "$TR" -p "$PL" -chk checkpoint_best.pth --save_probabilities
  FOLDERS+=("$OUT/$m")
done

# All models share the same preprocessed target grid (the ResEnc planners
# intentionally reuse the nnUNetPlans_3d_fullres data identifier), so their
# npz maps can be averaged directly.
nnUNetv2_ensemble -i "${FOLDERS[@]}" -o "$OUT/ensemble" --save_npz
echo "Final ensemble segmentations: $OUT/ensemble"
