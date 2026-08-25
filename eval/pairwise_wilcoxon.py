#!/usr/bin/env python
"""Aggregate per-case OOF Dice across 5 folds for each model from repredict summary.json,
then compute model pairwise Wilcoxon matrix + mean±std. Writes report."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
import SimpleITK as sitk
from scipy.stats import wilcoxon

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.environ.get('ATLAS_PREDS_ROOT', os.path.join(_REPO_ROOT, 'eval_preds'))
GT_DIR = os.environ.get('ATLAS_GT_DIR', os.path.join(
    os.environ.get('ATLAS_ROOT', _REPO_ROOT),
    'nnunet_data/preprocessed/Dataset501_ATLAS/gt_segmentations'))


def dice_from_npz(npz_path, cid):
    pr = np.load(npz_path)['probabilities'].astype(np.float32).argmax(0)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(GT_DIR, cid + '.nii.gz'))) > 0.5
    p = pr > 0
    tp = float(np.logical_and(p, gt).sum())
    fp = float(np.logical_and(p, ~gt).sum())
    fn = float(np.logical_and(~p, gt).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
MODELS = {
    'nnUNet':      'nnUNetTrainer__nnUNetPlans__3d_fullres',
    'MedNeXt-S':   'nnUNetTrainerMedNeXt_S_kernel3__nnUNetPlans__3d_fullres',
    'MedNeXt-B':   'nnUNetTrainerMedNeXt_B_kernel3__nnUNetPlans__3d_fullres',
    'ResEnc-M':    'nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres',
    'ResEnc-L':    'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres',
}

per_case = {tag: {} for tag in MODELS}
per_fold_mean = {tag: {} for tag in MODELS}

for tag, m in MODELS.items():
    for f in range(5):
        sp = os.path.join(PRED, m, f'fold_{f}', 'validation', 'summary.json')
        use_summary = False
        try:
            s = json.load(open(sp))
            if 'metric_per_case' in s:
                use_summary = True
        except Exception:
            pass
        if use_summary:
            fmean = s['foreground_mean']['Dice']
            per_fold_mean[tag][f] = fmean
            for e in s['metric_per_case']:
                cid = os.path.basename(e['prediction_file']).replace('.nii.gz', '')
                per_case[tag][cid] = e['metrics']['1']['Dice']
        else:
            # summary 缺失/损坏 (如配额崩溃残留): 从 npz+GT 补算 291 per-case Dice
            print(f'  [补算] {tag} fold{f} summary 损坏, 从 npz 计算...', flush=True)
            npzs = sorted(glob.glob(os.path.join(PRED, m, f'fold_{f}', 'validation', '*.npz')))
            ds = []
            for p in npzs:
                cid = os.path.basename(p)[:-4]
                d = dice_from_npz(p, cid)
                per_case[tag][cid] = d
                ds.append(d)
            per_fold_mean[tag][f] = float(np.mean(ds))

# common cases across all models
common = sorted(set.intersection(*[set(per_case[t].keys()) for t in MODELS]))
names = list(MODELS)
# 按消融约定: 空参考+空预测的 Dice 视为 1.0 (nnUNet summary 对此类返回 nan)
arr = {t: np.nan_to_num(np.array([per_case[t][c] for c in common], dtype=float), nan=1.0) for t in names}

print(f'common cases: {len(common)}\n')
print('=== single-model OOF (per-case, n=%d) ===' % len(common))
print(f'{"model":<12} {"mean":>7} {"std":>7} {"median":>7}   per-fold means')
for t in names:
    d = arr[t]
    fm = per_fold_mean[t]
    print(f'{t:<12} {d.mean():.4f}  {d.std():.4f}  {np.median(d):.4f}   ' +
          ' '.join(f'{fm[k]:.4f}' for k in range(5)))

print('\n=== pairwise Wilcoxon (row - col, p-value) ===')
hdr = '          ' + ''.join(f'{t:>12}' for t in names)
print(hdr)
for a in names:
    row = f'{a:<12}'
    for b in names:
        if a == b:
            row += '          -  '
        else:
            diff = arr[a] - arr[b]
            try:
                p = wilcoxon(arr[a], arr[b]).pvalue
            except Exception:
                p = float('nan')
            row += f'  +{diff.mean():.4f}/{p:.1e}'
    print(row)