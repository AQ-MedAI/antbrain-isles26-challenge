#!/usr/bin/env python3
"""Lever 1 (challenge-aligned): ensemble threshold + weight tuning under the official
5-metric set — Dice, Absolute Volume Difference (AVD), Absolute Lesion Count Diff
(ALCD), Lesion-wise F1, PR-AUC (threshold-independent, on the soft probability map).

PR-AUC only depends on the soft (weighted-prob) map, i.e. on the WEIGHTING, not the
threshold. The other four are threshold-dependent. We sweep (weighting, tau), report
all five, and pick the operating point by average rank across the five (lower=better),
since the exact challenge ranking formula isn't available to us.

Usage:
  python3 ensemble_challenge_metrics.py [--workers 8] [--limit 0] \
      [--out <log_dir>/ensemble_challenge_metrics] \
      [--diag <log_dir>/diag_lesion_response.json]
Paths default to ATLAS_LOG_DIR / ATLAS_PREDS_ROOT / ATLAS_GT_DIR (see pp_common.py).
"""
import argparse
import json
import os
import traceback

import numpy as np

import pp_common as pc

BASE = [
    'nnUNetTrainer__nnUNetPlans__3d_fullres',
    'nnUNetTrainerMedNeXt_S_kernel3__nnUNetPlans__3d_fullres',
    'nnUNetTrainerMedNeXt_B_kernel3__nnUNetPlans__3d_fullres',
]
SH = [pc.SHORT[m] for m in BASE]
TAUS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]


def build_configs():
    """Returns (configs, distinct_weights). config=(name, weight_idx, tau)."""
    cfgs = []
    weights = []
    widx = {}

    def wid(w):
        if w not in widx:
            widx[w] = len(weights); weights.append(w)
        return widx[w]

    for i in range(3):
        w = [0, 0, 0]; w[i] = 1
        cfgs.append((f'single_{SH[i]}_t0.5', wid(tuple(w)), 0.5))
    for t in TAUS:
        cfgs.append((f'equal_t{t}', wid((1, 1, 1)), t))
    for w in [(2, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2)]:
        for t in (0.30, 0.35, 0.40):
            cfgs.append((f'w{w}_t{t}', wid(w), t))
    return cfgs, weights


def ap_score(gt_bool, score):
    """Voxel-wise average precision (PR-AUC) of `score` against binary `gt`."""
    g = gt_bool.ravel()
    P = int(g.sum())
    if P == 0:
        return float('nan')
    s = score.ravel()
    order = np.argsort(-s, kind='stable')
    gs = g[order]
    tp = np.cumsum(gs)
    fp = np.cumsum(~gs)
    prec = tp / np.maximum(tp + fp, 1)
    return float(prec[gs].sum() / P)


def worker(args):
    case, base, maps, cfgs, weights, pheno = args
    try:
        G = pc.load_gt(case)
        probs = [pc.load_prob(maps[m][case]) for m in base]
        assert all(p.shape == G.shape for p in probs)
        lbl_g, n_g = pc.cc_label(G, 26)
        g_sizes = pc.component_sizes(lbl_g, n_g)
        gs = g_sizes[1:n_g + 1].astype(np.float64) if n_g > 0 else np.zeros(0)
        g_sum = int(G.sum())
        ph = pheno.get(case, ['mixed'] * n_g)
        if len(ph) < n_g:
            ph = ph + ['mixed'] * (n_g - len(ph))

        # PR-AUC per distinct weight (threshold-independent) + cache weighted prob
        ap_by_w = {}
        pw_cache = {}
        for wi, w in enumerate(weights):
            wsum = sum(w) or 1.0
            Pw = (probs[0] * w[0] + probs[1] * w[1] + probs[2] * w[2]) / wsum
            pw_cache[wi] = Pw
            ap_by_w[wi] = ap_score(G, Pw)

        cfg_res = {}
        for (name, wi, t) in cfgs:
            pred = pw_cache[wi] > t
            inter = int(np.count_nonzero(pred & G))
            denom = int(pred.sum()) + g_sum
            dice = 2 * inter / denom if denom > 0 else 1.0
            # AVD (relative) and ALCD
            vp = int(pred.sum()); vg = g_sum
            avd = abs(vp - vg) / vg if vg > 0 else (0.0 if vp == 0 else 1.0)
            det, cov, fp, n_pred = [], [], 0, 0
            if n_g > 0:
                lbl_p, n_p = pc.cc_label(pred, 26)
                n_pred = n_p
                if n_p == 0:
                    det = [False] * n_g; cov = [False] * n_g
                else:
                    ps = pc.component_sizes(lbl_p, n_p)[1:n_p + 1].astype(np.float64)
                    O = pc.lesion_overlap(lbl_g, n_g, lbl_p, n_p)
                    io = O[1:, 1:].astype(np.float64)
                    union = gs[:, None] + ps[None, :] - io
                    iou = np.where(union > 0, io / np.maximum(union, 1), 0.0)
                    det = (iou >= 0.1).any(axis=1).tolist()
                    cov = (io.sum(axis=1) / np.maximum(gs, 1) >= 0.10).tolist()
                    matched = (iou >= 0.1).any(axis=0)
                    fp = int((~matched).sum())
            alcd = abs(n_pred - n_g)
            cfg_res[name] = {'dice': float(dice), 'avd': float(avd), 'alcd': int(alcd),
                             'n_gt': int(n_g), 'n_pred': int(n_pred), 'det': int(sum(det)),
                             'det_cov': int(sum(cov)), 'fp': int(fp),
                             '_dets': det, '_cov': cov}
        return {'case': case, 'n_gt': int(n_g), 'ap_by_w': ap_by_w,
                'cfg': cfg_res, 'pheno': ph, 'vg': g_sum}
    except Exception as e:
        print(f'[WARN] {case}: {e}', flush=True)
        traceback.print_exc()
        return {'case': case, 'error': str(e)}


def aggregate(results, cfgs, weights):
    ok = [r for r in results if 'error' not in r]
    n = len(ok)
    # PR-AUC per weight (mean AP over GT-positive cases)
    prauc = {}
    for wi in range(len(weights)):
        vals = [r['ap_by_w'][wi] for r in ok if not np.isnan(r['ap_by_w'][wi])]
        prauc[wi] = float(np.mean(vals)) if vals else float('nan')

    agg = {}
    for (name, wi, t) in cfgs:
        dices = np.array([r['cfg'][name]['dice'] for r in ok])
        avds = np.array([r['cfg'][name]['avd'] for r in ok])
        alcds = np.array([r['cfg'][name]['alcd'] for r in ok])
        tot_gt = sum(r['cfg'][name]['n_gt'] for r in ok)
        tot_det = sum(r['cfg'][name]['det'] for r in ok)
        tot_fp = sum(r['cfg'][name]['fp'] for r in ok)
        miss = tot_gt - tot_det
        sens = tot_det / tot_gt if tot_gt else 0.0
        lf1 = 2 * tot_det / (2 * tot_det + tot_fp + miss) if (2 * tot_det + tot_fp + miss) else 1.0
        # AVD: mean over GT-positive cases (vg>0)
        pos = [r for r in ok if r['vg'] > 0]
        avd_mean = float(np.mean([r['cfg'][name]['avd'] for r in pos])) if pos else float('nan')
        a = {'mean_dice': float(dices.mean()), 'avd': avd_mean,
             'alcd': float(np.mean(alcds)), 'lesion_f1': float(lf1),
             'prauc': prauc[wi], 'sens': float(sens), 'det': tot_det, 'n_gt': tot_gt,
             'fp': tot_fp, 'fp_per_case': tot_fp / n if n else 0, 'wi': wi, 'tau': t}
        # phenotype stratification (IoU sens) for context
        a['strat'] = {}
        for phh in ['new-onset-like', 'chronic-like', 'mixed']:
            nph = detph = 0
            for r in ok:
                dets = r['cfg'][name]['_dets']
                for i, lbl in enumerate(r['pheno']):
                    if lbl == phh and i < len(dets):
                        nph += 1
                        if dets[i]: detph += 1
            a['strat'][phh] = {'n': nph, 'det': detph, 'sens': detph / nph if nph else 0.0}
        agg[name] = a
    return agg, n, prauc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    _log_dir = os.environ.get('ATLAS_LOG_DIR', os.path.join(
        os.environ.get('ATLAS_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs'))
    ap.add_argument('--out', default=os.path.join(_log_dir, 'ensemble_challenge_metrics'))
    ap.add_argument('--diag', default=os.path.join(_log_dir, 'diag_lesion_response.json'))
    args = ap.parse_args()

    pheno = {}
    if os.path.exists(args.diag):
        for L in json.load(open(args.diag)).get('per_lesion', []):
            pheno.setdefault(L['case'], []).append(L.get('pheno', 'mixed'))
    print(f'phenotype map: {len(pheno)} cases', flush=True)

    cfgs, weights = build_configs()
    print(f'configs: {len(cfgs)} | distinct weights: {len(weights)}', flush=True)

    def wname(w):
        if w in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            return 's' + SH[w.index(1)]
        return f'w{w}'

    wnames = {wi: wname(w) for wi, w in enumerate(weights)}

    maps = {m: pc.case_npz_map(m) for m in BASE}
    cases = pc.common_cases(BASE)
    if args.limit:
        cases = cases[:args.limit]
    print(f'cases: {len(cases)}', flush=True)

    jobs = [(c, BASE, maps, cfgs, weights, pheno) for c in cases]
    results, failed = pc.run_cases(jobs, worker, n_workers=args.workers, chunksize=1,
                                   desc='challMetrics')
    agg, n, prauc = aggregate(results, cfgs, weights)

    # PR-AUC table (threshold-independent) per weighting
    print('\n=== PR-AUC per weighting (soft-map, threshold-independent) ===')
    for wi in sorted(range(len(weights)), key=lambda x: -prauc[x]):
        print(f'  {wnames[wi]:<8} (w={weights[wi]}) PR-AUC={prauc[wi]:.4f}')

    # composite rank across the 5 metrics (lower rank-sum = better)
    names = [c[0] for c in cfgs]
    import scipy.stats as st
    def ranks(arr, higher_better):
        a = np.array(arr, float); nan = np.isnan(a)
        order = st.rankdata(-a if higher_better else a)
        order[nan] = len(a)  # nan -> worst
        return order
    r_dice = ranks([agg[m]['mean_dice'] for m in names], True)
    r_avd = ranks([agg[m]['avd'] for m in names], False)
    r_alcd = ranks([agg[m]['alcd'] for m in names], False)
    r_lf1 = ranks([agg[m]['lesion_f1'] for m in names], True)
    r_pa = ranks([agg[m]['prauc'] for m in names], True)
    for i, m in enumerate(names):
        agg[m]['rank_sum'] = float(r_dice[i] + r_avd[i] + r_alcd[i] + r_lf1[i] + r_pa[i])

    base = 'equal_t0.5'
    rows = sorted(names, key=lambda m: agg[m]['rank_sum'])
    print('\n=== all configs by 5-metric rank-sum (lower=better) ===')
    print(f'{"config":<22}{"Dice":>7}{"AVD":>8}{"ALCD":>6}{"lesF1":>7}{"PRAUC":>7}'
          f'{"rankSum":>9}')
    for m in rows:
        a = agg[m]
        print(f'{m:<22}{a["mean_dice"]:>7.4f}{a["avd"]:>8.4f}{a["alcd"]:>6.2f}'
              f'{a["lesion_f1"]:>7.4f}{a["prauc"]:>7.4f}{a["rank_sum"]:>9.1f}'
              f'{"  *BEST" if m == rows[0] else ""}')

    best = rows[0]
    print(f'\n=== baseline ({base}) vs best-by-rank ({best}) ===')
    for k in ['mean_dice', 'avd', 'alcd', 'lesion_f1', 'prauc', 'sens', 'fp_per_case']:
        print(f'  {k:<12} base={agg[base][k]:.4f}  best={agg[best][k]:.4f}  '
              f'delta={agg[best][k]-agg[base][k]:+.4f}')
    print('  phenotype sens (IoU):')
    for phh in ['new-onset-like', 'chronic-like', 'mixed']:
        b, B = agg[base]['strat'][phh], agg[best]['strat'][phh]
        print(f'    {phh:<16} base {b["sens"]:.4f} -> best {B["sens"]:.4f}')

    out = {'n_cases': n, 'n_failed': len(failed), 'base': base, 'best': best,
           'prauc_per_weight': {wnames[wi]: prauc[wi] for wi in range(len(weights))},
           'agg': agg}
    with open(args.out + '.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    md = ['# Lever 1 — ensemble tuning under challenge 5-metric set\n',
          f'Cases (3-model ensemble, OOF): {n} (failed {len(failed)})\n',
          f'Baseline = {base}\n',
          '## PR-AUC per weighting (threshold-independent)\n',
          '| weighting | PR-AUC |\n|---|---:|\n']
    for wi in sorted(range(len(weights)), key=lambda x: -prauc[x]):
        md.append(f'| {wnames[wi]} w={weights[wi]} | {prauc[wi]:.4f} |\n')
    md.append('\n## All configs by 5-metric rank-sum (lower=better)\n')
    md.append('| config | Dice | AVD | ALCD | lesF1 | PR-AUC | rankSum |\n|---|---:|---:|---:|---:|---:|---:|\n')
    for m in rows:
        a = agg[m]
        md.append(f'| {m}{" *" if m==best else ""} | {a["mean_dice"]:.4f} | {a["avd"]:.4f} | '
                  f'{a["alcd"]:.2f} | {a["lesion_f1"]:.4f} | {a["prauc"]:.4f} | {a["rank_sum"]:.1f} |\n')
    md.append(f'\n## Baseline ({base}) vs best ({best})\n')
    md.append('| metric | baseline | best | delta |\n|---|---:|---:|---:|\n')
    for k in ['mean_dice', 'avd', 'alcd', 'lesion_f1', 'prauc', 'sens', 'fp_per_case']:
        md.append(f'| {k} | {agg[base][k]:.4f} | {agg[best][k]:.4f} | {agg[best][k]-agg[base][k]:+.4f} |\n')
    with open(args.out + '.md', 'w') as f:
        f.write(''.join(md))
    print(f'\nwrote {args.out}.json and {args.out}.md')


if __name__ == '__main__':
    main()