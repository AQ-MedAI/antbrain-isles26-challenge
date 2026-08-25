#!/usr/bin/env bash
# nnU-Net v2 environment template for the ISLES'26 submission repo.
# Usage: cp env_template.sh env.sh && edit paths && source env.sh

# Repo root (directory containing this file)
export ATLAS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# nnU-Net v2 path triple — EDIT THESE for your machine
export nnUNet_raw=$ATLAS_ROOT/nnunet_data/raw
export nnUNet_preprocessed=$ATLAS_ROOT/nnunet_data/preprocessed
export nnUNet_results=$ATLAS_ROOT/nnunet_results        # must contain Dataset501_ATLAS/<model>/fold_*/

# MedNeXt custom trainer discovery (nnunetv2 >= 2.8 external trainer mechanism)
export nnUNet_extTrainer=$ATLAS_ROOT/mednext_v2/trainers

# Eval-side roots (only needed by eval/ scripts)
# export ATLAS_PREDS_ROOT=$ATLAS_ROOT/eval_preds        # OOF npz root
# export ATLAS_GT_DIR=$nnUNet_preprocessed/Dataset501_ATLAS/gt_segmentations
# export ATLAS_RAW_ROOT=/path/to/ATLAS3_Training_Raw    # BIDS root (convert/eval only)

# Runtime knobs
export nnUNet_def_n_proc=4        # dataloader workers; scale with CPU cores
export nnUNet_compile=False       # required for MedNeXt (kernel-5 pathological under compile)
export OMP_NUM_THREADS=1
