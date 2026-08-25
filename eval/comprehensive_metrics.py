#!/usr/bin/env python
"""ATLAS R3.0 综合多指标评估 (挑战赛式 5 指标 + rank-sum)。

对每个 case 加载 5 模型 softmax 软图 + GT, 对 5 单模型 + 4 代表集成计算:
  Dice / Absolute Volume Difference (mL) / Absolute Lesion Count Difference /
  Lesion-wise F1 (>0 overlap) / PR-AUC (average precision over soft map)
然后对每个指标给配置排名 (1=best), AVD/ALCD 越低越好, 其余越高越好, 总分=5 指标平均名次。

用法:
  ATLAS_GT_DIR=<gt> python3 comprehensive_metrics.py <preds_base> > out.txt
写 logs/comprehensive_metrics.json
spawn 多进程 + per-case try/except (鲁棒, 不会静默死锁)。
"""
import glob, itertools, json, multiprocessing as mp, os, sys, traceback
import numpy as np
import SimpleITK as sitk
import cc3d
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = os.environ.get('ATLAS_GT_DIR', os.path.join(
    os.environ.get('ATLAS_ROOT', _REPO_ROOT),
    'nnunet_data/preprocessed/Dataset501_ATLAS/gt_segmentations'))
LOG_DIR = os.environ.get('ATLAS_LOG_DIR', os.path.join(os.environ.get('ATLAS_ROOT', _REPO_ROOT), 'logs'))

MODELS = [
    'nnUNetTrainer__nnUNetPlans__3d_fullres',
    'nnUNetTrainerMedNeXt_S_kernel3__nnUNetPlans__3d_fullres',
    'nnUNetTrainerMedNeXt_B_kernel3__nnUNetPlans__3d_fullres',
    'nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres',
    'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres',
]
SHORT = ['nnUNet', 'MedNeXt-S', 'MedNeXt-B', 'ResEnc-M', 'ResEnc-L']
# 配置: (名字, [模型下标])
CONFIGS = [
    ('nnUNet',        [0]),
    ('MedNeXt-S',     [1]),
    ('MedNeXt-B',     [2]),
    ('ResEnc-M',      [3]),
    ('ResEnc-L',      [4]),
    ('ENS: S+M(2)',   [1, 3]),
    ('ENS: S+B+M(3)', [1, 2, 3]),
    ('ENS: best4',    [0, 1, 2, 3]),
    ('ENS: all5',     [0, 1, 2, 3, 4]),
]


def case_map(preds_base, m):
    out = {}
    for f in sorted(glob.glob(os.path.join(preds_base, m, 'fold_*', 'validation', '*.npz'))):
        out[os.path.basename(f)[:-4]] = f
    return out


def lesion_f1(pred, gt_cc, n_gt):
    """Lesion-wise F1 (any-overlap detection). gt_cc precomputed, pred binarized here."""
    if pred.sum() == 0 and n_gt == 0:
        return 1.0
    pred_cc = cc3d.connected_components(pred, connectivity=26)
    n_pred = int(pred_cc.max())
    if n_gt == 0:
        return 0.0  # GT 无病灶但 pred 有 → 全 FP, F1=0
    if n_pred == 0:
        return 0.0  # pred 无 → 全 FN
    mask = (gt_cc > 0) & (pred_cc > 0)
    if not mask.any():
        return 0.0
    g_labels = gt_cc[mask]
    p_labels = pred_cc[mask]
    pairs = set(zip(g_labels.tolist(), p_labels.tolist()))
    detected_gt = len({g for g, _ in pairs})   # 被命中的 GT 病灶
    detected_pred = len({p for _, p in pairs})  # 命中 GT 的 pred 病灶
    tp = detected_gt
    fp = n_pred - detected_pred
    fn = n_gt - detected_gt
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0


def eval_case(args):
    cid, npz_paths = args
    try:
        probs = [np.load(p)['probabilities'].astype(np.float32)[1] for p in npz_paths]  # 软图 (Z,Y,X)
        gt_im = sitk.ReadImage(os.path.join(GT_DIR, cid + '.nii.gz'))
        gt = sitk.GetArrayFromImage(gt_im) > 0.5  # (Z,Y,X)
        sp = gt_im.GetSpacing()  # (x,y,z); 1mm 等距
        vox_ml = float(sp[0] * sp[1] * sp[2]) / 1000.0
        gt_cc = cc3d.connected_components(gt, connectivity=26)
        n_gt = int(gt_cc.max())
        gt_vol = float(gt.sum()) * vox_ml
        gt_npos = int(gt.sum())

        out = {}
        for name, idxs in CONFIGS:
            soft = sum(probs[i] for i in idxs) / len(idxs)
            pred = soft > 0.5
            # Dice
            tp = float(np.logical_and(pred, gt).sum())
            fp = float(np.logical_and(pred, ~gt).sum())
            fn = float(np.logical_and(~pred, gt).sum())
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
            # AVD (mL)
            avd = abs(float(pred.sum()) * vox_ml - gt_vol)
            # ALCD
            n_pred = int(cc3d.connected_components(pred, connectivity=26).max())
            alcd = abs(n_pred - n_gt)
            # Lesion F1
            f1 = lesion_f1(pred, gt_cc, n_gt)
            # PR-AUC (soft map AP); GT 无正样本则 NaN
            if gt_npos == 0:
                prauc = float('nan')
            else:
                try:
                    prauc = float(average_precision_score(gt.ravel().astype(np.int8),
                                                           soft.ravel()))
                except Exception:
                    prauc = float('nan')
            out[name] = dict(dice=dice, avd=avd, alcd=float(alcd), f1=f1, prauc=prauc)
        return cid, out, (gt_npos == 0)
    except Exception as e:
        print(f'[WARN] case {cid} failed: {e}', flush=True)
        traceback.print_exc()
        return cid, None, False


def main():
    preds_base = sys.argv[1]
    maps = {m: case_map(preds_base, m) for m in MODELS}
    common = sorted(set.intersection(*[set(v) for v in maps.values()]))
    print(f'common cases: {len(common)}', flush=True)
    jobs = [(c, [maps[m][c] for m in MODELS]) for c in common]

    ctx = mp.get_context('spawn')
    results, failed, no_lesion = [], [], 0
    with ctx.Pool(8, maxtasksperchild=12) as pool:
        done = 0
        for cid, out, nol in pool.imap_unordered(eval_case, jobs, chunksize=4):
            if out is None:
                failed.append(cid)
            else:
                results.append((cid, out))
                if nol:
                    no_lesion += 1
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f'progress {done}/{len(jobs)} ok={len(results)} fail={len(failed)}', flush=True)
    print(f'\nDONE: {len(results)} ok, {len(failed)} failed, {no_lesion} cases have empty GT lesion', flush=True)

    cfg_names = [c[0] for c in CONFIGS]
    metrics = ['dice', 'avd', 'alcd', 'f1', 'prauc']
    agg = {c: {m: [] for m in metrics} for c in cfg_names}
    for cid, out in results:
        for c in cfg_names:
            for m in metrics:
                v = out[c][m]
                if not (isinstance(v, float) and v != v):  # not NaN
                    agg[c][m].append(v)
    # 汇总: mean (std)
    mean = {c: {m: float(np.mean(agg[c][m])) if agg[c][m] else float('nan') for m in metrics} for c in cfg_names}
    std = {c: {m: float(np.std(agg[c][m])) if agg[c][m] else float('nan') for m in metrics} for c in cfg_names}
    n_pr = {c: len(agg[c]['prauc']) for c in cfg_names}  # PR-AUC 有效 case 数

    # rank-sum: AVD/ALCD 越低越好; dice/f1/prauc 越高越好
    higher_better = {'dice', 'f1', 'prauc'}
    ranks = {}
    for m in metrics:
        vals = np.array([mean[c][m] for c in cfg_names])
        if m in higher_better:
            r = rankdata(-vals, method='average')   # 最大→1
        else:
            r = rankdata(vals, method='average')    # 最小→1
        ranks[m] = r
    total = np.mean([ranks[m] for m in metrics], axis=0)

    order = sorted(range(len(cfg_names)), key=lambda i: total[i])
    print('\n=== 综合指标 (per-case mean over n=%d; PR-AUC 仅含有病灶 case) ===' % len(results), flush=True)
    print(f'{"配置":<16}{"Dice":>8}{"AVD(mL)":>10}{"ALCD":>8}{"LesF1":>8}{"PR-AUC":>9}{"(n_pr)":>8}  总排名', flush=True)
    for i in order:
        c = cfg_names[i]
        print(f'{c:<16}{mean[c]["dice"]:8.4f}{mean[c]["avd"]:10.3f}{mean[c]["alcd"]:8.3f}'
              f'{mean[c]["f1"]:8.4f}{mean[c]["prauc"]:9.4f}{n_pr[c]:>8}  {total[i]:5.2f}', flush=True)
    print('\n=== 各指标单项排名 (1=best) ===', flush=True)
    hdr = f'{"配置":<16}' + ''.join(f'{m:>9}' for m in metrics) + f'{"  均值名次":>11}'
    print(hdr, flush=True)
    for i in order:
        c = cfg_names[i]
        row = f'{c:<16}' + ''.join(f'{ranks[m][i]:9.1f}' for m in metrics) + f'{total[i]:11.2f}'
        print(row, flush=True)
    print('\n注: AVD/ALCD 越低越好(排名升序); Dice/LesF1/PR-AUC 越高越好(排名降序);'
          ' 总分=5 指标平均名次, 越低越优。PR-AUC 用 sklearn average_precision(软图分数)。', flush=True)

    out = {'n_cases': len(results), 'n_failed': len(failed),
           'n_empty_gt': no_lesion,
           'configs': cfg_names, 'metrics': metrics,
           'mean': mean, 'std': std, 'n_pr_valid': n_pr,
           'ranks': {m: ranks[m].tolist() for m in metrics},
           'total_rank': total.tolist(),
           'order': [cfg_names[i] for i in order]}
    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, 'comprehensive_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\nwrote {out_path}', flush=True)


if __name__ == '__main__':
    main()