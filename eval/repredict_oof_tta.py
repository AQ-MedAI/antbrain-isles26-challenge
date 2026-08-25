#!/usr/bin/env python3
"""Repredict one OOF fold WITH mirror TTA, saving softmax probabilities (.npz) in the
eval_preds layout so the existing 5-metric eval scripts can consume them directly.

Writes: <out_base>/<model_dir>/fold_<fold>/validation/<case>.npz (+ <case>.nii.gz)
        npz key 'probabilities', shape (2,Z,Y,X) softmax — matches load_prob().

GPU required. Environment: py310 + torch==2.4.1 + nnunetv2 (2.8.1) + the MedNeXt
trainers discoverable via nnUNet_extTrainer (for MedNeXt). See LEVER3_TTA_HANDOFF.md.

Usage:
  python repredict_oof_tta.py <trainer_name> <plans_id> <model_dir> <fold> <out_base>
Example:
  python repredict_oof_tta.py nnUNetTrainer nnUNetPlans \
      nnUNetTrainer__nnUNetPlans__3d_fullres 0 /path/eval_preds_tta
  python repredict_oof_tta.py nnUNetTrainerMedNeXt_S_kernel3 nnUNetPlans \
      nnUNetTrainerMedNeXt_S_kernel3__nnUNetPlans__3d_fullres 0 /path/eval_preds_tta
"""
import os
import shutil
import sys

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results

DATASET = 'Dataset501_ATLAS'


def main():
    trainer_name, plans_id = sys.argv[1], sys.argv[2]
    model_dir, fold = sys.argv[3], int(sys.argv[4])
    out_base = sys.argv[5]

    splits = load_json(os.path.join(nnUNet_preprocessed, DATASET, 'splits_final.json'))
    # val entries are case ids like 'ATLAS_r001s003'
    val_cases = [os.path.basename(f).replace('.nii.gz', '') for f in splits[fold]['val']]
    print(f'fold {fold}: {len(val_cases)} val cases', flush=True)

    imgs_dir = os.path.join(nnUNet_raw, DATASET, 'imagesTr')
    tmp_in = f'/tmp/tta_in_{model_dir}_f{fold}'
    if os.path.exists(tmp_in):
        shutil.rmtree(tmp_in)
    os.makedirs(tmp_in, exist_ok=True)
    for c in val_cases:
        src = os.path.join(imgs_dir, c + '_0000.nii.gz')
        if not os.path.exists(src):
            print(f'[WARN] missing image {src}', flush=True)
            continue
        os.symlink(src, os.path.join(tmp_in, c + '_0000.nii.gz'))

    out_dir = os.path.join(out_base, model_dir, f'fold_{fold}', 'validation')
    os.makedirs(out_dir, exist_ok=True)
    model_folder = os.path.join(nnUNet_results, DATASET, model_dir)
    print(f'model_folder: {model_folder}', flush=True)

    # use_mirroring=True => test-time augmentation (mirror flips). This is the TTA.
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=torch.device('cuda'),
        verbose=False, allow_tqdm=True)
    # nnunetv2 2.8.1 API: trainer/config/plans are auto-detected from the checkpoint
    # and plans.json inside model_folder; only checkpoint_name is selectable.
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=[fold], checkpoint_name='checkpoint_best.pth')
    predictor.predict_from_files(
        tmp_in, out_dir, save_probabilities=True, overwrite=True,
        folder_with_segs_from_prev_stage=None)
    print(f'DONE {model_dir} fold {fold} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()