#!/usr/bin/env python
"""Paired comparison of models using 5-fold out-of-fold validation predictions.

Every one of the 1453 ATLAS cases appears in exactly one fold's validation set,
so pooling per-case metrics across folds gives a proper out-of-sample evaluation
on the full dataset. Paired statistics (Wilcoxon signed-rank) compare models
case-by-case.

Usage: python compare_models.py [model_dir_name ...]
Default: compare all available models against nnUNetTrainer__nnUNetPlans.
"""
import glob
import json
import os
import sys
import statistics

import numpy as np
from scipy.stats import wilcoxon

BASE = os.environ.get('ATLAS_MODELS_ROOT', os.path.join(
    os.environ.get('nnUNet_results', 'nnunet_results'), 'Dataset501_ATLAS'))


def collect_oof_dice(model_dir):
    """Return dict case_id -> Dice from all completed folds' validation summaries."""
    out = {}
    for sj in sorted(glob.glob(f'{BASE}/{model_dir}/fold_*/validation/summary.json')):
        fold = sj.split('fold_')[1].split('/')[0]
        j = json.load(open(sj))
        for c in j['metric_per_case']:
            cid = os.path.basename(c['prediction_file']).replace('.nii.gz', '')
            out[cid] = {'fold': int(fold), 'Dice': c['metrics']['1']['Dice'],
                        'IoU': c['metrics']['1']['IoU'],
                        'TP': c['metrics']['1']['TP'], 'FP': c['metrics']['1']['FP'],
                        'FN': c['metrics']['1']['FN']}
    return out


def report(name, metrics):
    d = [m['Dice'] for m in metrics.values()]
    iou = [m['IoU'] for m in metrics.values()]
    gd = {k: m for k, m in metrics.items()}
    tp = sum(m['TP'] for m in gd.values()); fp = sum(m['FP'] for m in gd.values()); fn = sum(m['FN'] for m in gd.values())
    print(f'{name:<40} n={len(d):<5} meanDice={np.mean(d):.4f}±{np.std(d):.3f} '
          f'median={np.median(d):.4f} meanIoU={np.mean(iou):.4f} globalDice={2*tp/(2*tp+fp+fn):.4f}')
    return d


def main():
    models = sys.argv[1:]
    if not models:
        models = sorted(os.listdir(BASE))
    data = {}
    for m in models:
        r = collect_oof_dice(m)
        if r:
            data[m] = r
    print('=== per-model out-of-fold metrics (pooled over folds) ===')
    for m, r in data.items():
        report(m, r)

    names = list(data.keys())
    if len(names) < 2:
        print('\nneed >=2 models with completed folds for paired comparison')
        return
    ref = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
    ref = ref if ref in data else names[0]
    print(f'\n=== paired comparison vs {ref} ===')
    for m in names:
        if m == ref:
            continue
        common = sorted(set(data[ref]) & set(data[m]))
        a = np.array([data[ref][c]['Dice'] for c in common])
        b = np.array([data[m][c]['Dice'] for c in common])
        diff = b - a
        wins, ties, losses = int((diff > 1e-6).sum()), int((np.abs(diff) <= 1e-6).sum()), int((diff < -1e-6).sum())
        try:
            p = wilcoxon(a, b).pvalue
        except Exception as e:
            p = float('nan')
        print(f'{m:<40} n={len(common)} meanDiff={diff.mean():+.4f} medianDiff={np.median(diff):+.4f} '
              f'win/tie/loss={wins}/{ties}/{losses} wilcoxon_p={p:.3e}')


if __name__ == '__main__':
    main()
