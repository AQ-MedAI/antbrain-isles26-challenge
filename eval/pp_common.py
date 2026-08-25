#!/usr/bin/env python3
"""Shared utilities for ATLAS post-processing (Schemes D & C).

Everything works in the preprocessed nnUNet space (1mm iso), which is the space
of both the OOF probability npz and the GT. Raw T1 is resampled onto the GT grid
(SimpleITK) so that intensity evidence aligns with the probability/GT array.

Axis convention: npz probabilities are (C, Z, Y, X); we operate in (Z, Y, X).
GT is read with SimpleITK (matches nnUNet's geometry exactly); GetArrayFromImage
gives (Z, Y, X) directly. Sanity-checked: GT-zyx shape == npz spatial shape.

cc3d is replaced by scipy.ndimage.label.

Requires: numpy, SimpleITK, scipy (see requirements.txt).
"""

import contextlib
import glob
import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------
# Paths (all overridable via environment; defaults are relative to this repo)
# ----------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS_ROOT = os.environ.get('ATLAS_ROOT', _REPO_ROOT)
GT_DIR = os.environ.get(
    'ATLAS_GT_DIR',
    os.path.join(ATLAS_ROOT, 'nnunet_data/preprocessed/Dataset501_ATLAS/gt_segmentations'))
PREDS_ROOT = os.environ.get('ATLAS_PREDS_ROOT', os.path.join(ATLAS_ROOT, 'eval_preds'))
RAW_ROOT = os.environ.get('ATLAS_RAW_ROOT', os.path.join(ATLAS_ROOT, 'ATLAS3_Training_Raw'))
LOG_DIR = os.environ.get('ATLAS_LOG_DIR', os.path.join(ATLAS_ROOT, 'logs'))

# All 5 models whose OOF npz exist in PREDS_ROOT.
MODELS = [
    'nnUNetTrainer__nnUNetPlans__3d_fullres',
    'nnUNetTrainerMedNeXt_S_kernel3__nnUNetPlans__3d_fullres',
    'nnUNetTrainerMedNeXt_B_kernel3__nnUNetPlans__3d_fullres',
    'nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres',
    'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres',
]
SHORT = {'nnUNetTrainer__nnUNetPlans__3d_fullres': 'nnUNet',
         'nnUNetTrainerMedNeXt_S_kernel3__nnUNetPlans__3d_fullres': 'MedNext-S',
         'nnUNetTrainerMedNeXt_B_kernel3__nnUNetPlans__3d_fullres': 'MedNext-B',
         'nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres': 'ResEnc-M',
         'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres': 'ResEnc-L'}

STRUCT26 = ndi.generate_binary_structure(3, 3)  # 26-connectivity


# ----------------------------------------------------------------------------
# Quiet SimpleITK reads (the ATLAS sform-scale warnings are benign but spammy)
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def _silence_stderr():
    fd = sys.stderr.fileno()
    saved = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


def _read_image(path):
    with _silence_stderr():
        return sitk.ReadImage(path)


# ----------------------------------------------------------------------------
# Case <-> raw T1 path
# ----------------------------------------------------------------------------
def sid_from_case(case):
    return case[len('ATLAS_'):] if case.startswith('ATLAS_') else case


def raw_t1_path(case):
    sid = sid_from_case(case)                 # e.g. r001s003
    site = sid.split('s')[0].upper()          # e.g. R001
    return os.path.join(RAW_ROOT, site, f'sub-{sid}', 'ses-1', 'anat',
                        f'sub-{sid}_ses-1_space-orig_desc-brain_T1w.nii.gz')


def gt_path(case):
    return os.path.join(GT_DIR, case + '.nii.gz')


# ----------------------------------------------------------------------------
# Loaders (all return (Z, Y, X) numpy)
# ----------------------------------------------------------------------------
def load_prob(npz_path):
    """Lesion softmax probability, float32, shape (Z, Y, X)."""
    return np.load(npz_path)['probabilities'][1].astype(np.float32)


def load_gt(case, return_img=False):
    """GT lesion mask (bool, Z,Y,X). Optionally also the SimpleITK image (for
    use as a resample reference geometry)."""
    img = _read_image(gt_path(case))
    arr = sitk.GetArrayFromImage(img) > 0.5
    if return_img:
        return arr, img
    return arr


def resample_to_ref(moving, ref, interp=sitk.sitkLinear):
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(ref)
    f.SetInterpolator(interp)
    f.SetDefaultPixelValue(0)
    f.SetTransform(sitk.Transform())  # identity (same physical space, diff grid)
    return f.Execute(moving)


def load_t1_on_gt_grid(case, ref_img):
    """Raw T1 resampled onto the GT grid -> (Z, Y, X) float32, aligned with P/GT."""
    p = raw_t1_path(case)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    raw = _read_image(p)
    res = resample_to_ref(raw, ref_img, sitk.sitkLinear)
    return sitk.GetArrayFromImage(res).astype(np.float32)


# ----------------------------------------------------------------------------
# Case <-> npz maps
# ----------------------------------------------------------------------------
def case_npz_map(model_dir):
    """{case: npz_path} across all folds (each case lives in exactly one fold)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(PREDS_ROOT, model_dir, 'fold_*', 'validation', '*.npz'))):
        out[os.path.basename(f)[:-4]] = f
    return out


def all_cases(model_dir):
    return sorted(case_npz_map(model_dir).keys())


def common_cases(model_dirs):
    maps = [set(case_npz_map(m).keys()) for m in model_dirs]
    return sorted(set.intersection(*maps))


def load_ensemble_prob(case, model_dirs, maps=None):
    """Mean lesion probability across models for one case."""
    if maps is None:
        maps = {m: case_npz_map(m) for m in model_dirs}
    probs = None
    for m in model_dirs:
        pr = load_prob(maps[m][case])
        probs = pr if probs is None else probs + pr
    return (probs / len(model_dirs)).astype(np.float32)


# ----------------------------------------------------------------------------
# Connected components (cc3d replacement)
# ----------------------------------------------------------------------------
def cc_label(binary, connectivity=26):
    struct = STRUCT26 if connectivity == 26 else ndi.generate_binary_structure(3, 1)
    lbl, n = ndi.label(binary, structure=struct)
    return lbl, n


def component_sizes(lbl, n=None):
    sizes = np.bincount(lbl.ravel())
    if n is not None:
        sizes = sizes[:n + 1]
    sizes[0] = 0
    return sizes


# ----------------------------------------------------------------------------
# Brain mask (numpy/scipy; do not assume skull-stripped)
# ----------------------------------------------------------------------------
def _otsu_threshold(values, nbins=256):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist)
    total = hist.sum()
    if total == 0:
        return float(v.min())
    wB = w / total
    bin_mids = centers
    mB = np.cumsum(hist * bin_mids) / np.maximum(w, 1)
    mF = (np.cumsum(hist * bin_mids)[-1] - np.cumsum(hist * bin_mids)) / np.maximum(total - w, 1)
    var_between = wB * (1 - wB) * (mB - mF) ** 2
    idx = int(np.nanargmax(var_between))
    return float(bin_mids[idx])


def brain_mask(I, dilate_iter=0):
    """Binary brain mask from T1 (Z,Y,X).

    ATLAS R3.0 T1w is skull-stripped ('desc-brain'), so background == 0 and brain is
    the non-zero region. We therefore take I>0 and fill holes (the earlier Otsu +
    largest-CC produced a far-too-small ~18%-of-volume mask that cropped real
    predictions and crashed baseline Dice from 0.666 to ~0.49).
    """
    mask = I > 0
    if not mask.any():
        return mask
    # drop tiny dust (residual intensity noise) then fill holes
    lbl, n = cc_label(mask, connectivity=26)
    if n > 1:
        sizes = component_sizes(lbl, n)
        out = (lbl > 0) & (sizes[lbl] >= 50)
    else:
        out = mask
    out = ndi.binary_fill_holes(out)
    if dilate_iter:
        out = ndi.binary_closing(out, iterations=dilate_iter)
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def dice(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 1.0


def lesion_overlap(lbl_gt, n_gt, lbl_pred, n_pred):
    """Overlap counts matrix O[g, p] (g,p >= 1); shape (n_gt+1, n_pred+1)."""
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt + 1, n_pred + 1), dtype=np.int64)
    combined = lbl_gt.astype(np.int64) * (n_pred + 1) + lbl_pred.astype(np.int64)
    counts = np.bincount(combined.ravel(), minlength=(n_gt + 1) * (n_pred + 1))
    return counts.reshape(n_gt + 1, n_pred + 1)


def lesion_metrics(pred, gt, iou_thr=0.1):
    """Lesion-wise detection metrics for one case.

    A GT lesion is 'detected' if some pred CC overlaps it with IoU >= iou_thr.
    Returns dict with sensitivity, fp_lesions, lesion_f1, n_gt, n_pred, detected.
    """
    pred = pred.astype(bool); gt = gt.astype(bool)
    lbl_gt, n_gt = cc_label(gt, 26)
    lbl_pred, n_pred = cc_label(pred, 26)

    out = {'n_gt': int(n_gt), 'n_pred': int(n_pred), 'detected': 0,
           'sensitivity': 0.0, 'fp_lesions': int(n_pred), 'lesion_f1': 0.0}

    if n_gt == 0 and n_pred == 0:
        out['fp_lesions'] = 0
        out['lesion_f1'] = 1.0  # nothing to do, perfect
        return out
    if n_gt == 0:
        # all predicted lesions are false positives
        out['lesion_f1'] = 0.0
        return out
    if n_pred == 0:
        out['fp_lesions'] = 0
        out['sensitivity'] = 0.0
        out['lesion_f1'] = 0.0
        return out

    O = lesion_overlap(lbl_gt, n_gt, lbl_pred, n_pred)
    sizes_g = O.sum(axis=1)
    sizes_p = O.sum(axis=0)
    g_ids = np.arange(1, n_gt + 1)
    p_ids = np.arange(1, n_pred + 1)
    g_sizes = component_sizes(lbl_gt, n_gt)[g_ids]
    p_sizes = component_sizes(lbl_pred, n_pred)[p_ids]
    inter = O[1:, 1:].astype(np.float64)             # (n_gt, n_pred)
    union = g_sizes[:, None] + p_sizes[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)

    detected = (iou >= iou_thr).any(axis=1)           # per GT lesion
    tp = int(detected.sum())
    matched_pred = (iou >= iou_thr).any(axis=0)       # pred CC matching some GT
    fp = int((~matched_pred).sum())

    sens = tp / n_gt
    f1 = 2 * tp / (2 * tp + fp + (n_gt - tp)) if (2 * tp + fp + (n_gt - tp)) > 0 else 1.0
    out.update({'detected': tp, 'sensitivity': float(sens),
                'fp_lesions': int(fp), 'lesion_f1': float(f1)})
    return out


# ----------------------------------------------------------------------------
# Robust multiprocessing runner (spawn ctx; per-case try/except; observable)
# ----------------------------------------------------------------------------
def run_cases(cases, worker, n_workers=8, chunksize=1, desc='cases'):
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    results, failed = [], []
    with ctx.Pool(n_workers, maxtasksperchild=16) as pool:
        done = 0
        for r in pool.imap_unordered(worker, cases, chunksize=chunksize):
            if r is None:
                failed.append('?')
            elif isinstance(r, tuple) and r[0] is None:
                failed.append(r[1] if len(r) > 1 else '?')
            else:
                results.append(r)
            done += 1
            if done % 25 == 0 or done == len(cases):
                print(f'[{desc}] {done}/{len(cases)} ok={len(results)} fail={len(failed)}', flush=True)
    if failed:
        print(f'[{desc}] failed: {failed[:10]}{" ..." if len(failed) > 10 else ""}', flush=True)
    return results, failed


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    t0 = time.time()
    case = 'ATLAS_r001s003'
    print(f'--- pp_common smoke test on {case} ---')

    # probability + GT geometry consistency
    m = MODELS[2]  # MedNeXt-B
    nmap = case_npz_map(m)
    P = load_prob(nmap[case])
    G, gimg = load_gt(case, return_img=True)
    print(f'prob {P.shape} dtype={P.dtype} min={P.min():.3f} max={P.max():.3f}')
    print(f'GT   {G.shape} dtype={G.dtype} lesions-voxels={int(G.sum())}')
    assert P.shape == G.shape, f'shape mismatch {P.shape} vs {G.shape}'

    # raw T1 onto GT grid
    I = load_t1_on_gt_grid(case, gimg)
    print(f'T1   {I.shape} spacing={gimg.GetSpacing()} min={I.min():.1f} max={I.max():.1f}')
    assert I.shape == G.shape, f'T1 shape mismatch {I.shape} vs {G.shape}'

    # brain mask
    B = brain_mask(I)
    print(f'brain mask voxels={int(B.sum())} ({100*B.mean():.1f}% of volume)')

    # lesion metrics on a baseline (argmax) prediction
    pred = P > 0.5
    print(f'baseline argmax(0.5): pred voxels={int(pred.sum())} dice={dice(pred, G):.4f}')
    lm = lesion_metrics(pred, G, iou_thr=0.1)
    print(f'lesion metrics: {lm}')

    # test a few more cases for case->raw mapping existence
    for c in ['ATLAS_r001s003', 'ATLAS_r040s053', 'ATLAS_r040s087']:
        rp = raw_t1_path(c)
        print(f'  raw for {c}: exists={os.path.exists(rp)}')
    print(f'--- done in {time.time()-t0:.1f}s ---')
