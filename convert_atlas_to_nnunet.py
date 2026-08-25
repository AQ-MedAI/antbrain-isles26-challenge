#!/usr/bin/env python
"""Convert ATLAS R3.0 (BIDS) to nnUNet v2 raw format.

Source: <ATLAS_RAW_ROOT>/R*/sub-*/ses-1/anat/  (ATLAS R3.0 BIDS; set ATLAS_RAW_ROOT)
  image: sub-XXX_ses-1_space-orig_desc-brain_T1w.nii.gz
  mask : sub-XXX_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz

Target: $nnUNet_raw/Dataset501_ATLAS/
  imagesTr/ATLAS_<sid>_0000.nii.gz   (channel 0 = T1w)
  labelsTr/ATLAS_<sid>.nii.gz
  dataset.json
"""
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

SRC = Path(os.environ.get('ATLAS_RAW_ROOT', 'ATLAS3_Training_Raw'))
DST = Path(os.environ.get('nnUNet_raw', 'nnunet_raw')) / 'Dataset501_ATLAS'
IMG_SUFFIX = "_space-orig_desc-brain_T1w.nii.gz"
MSK_SUFFIX = "_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"


def find_pairs():
    pairs = []
    for img in sorted(SRC.glob("*/sub-*/ses-1/anat/*" + IMG_SUFFIX)):
        sid = img.name.split("_")[0].replace("sub-", "")  # e.g. r001s001
        msk = img.with_name(img.name.replace(IMG_SUFFIX, MSK_SUFFIX))
        if msk.exists():
            pairs.append((sid, img, msk))
        else:
            print(f"[WARN] missing mask for {img}", flush=True)
    return pairs


def process_one(args):
    sid, img, msk = args
    case = f"ATLAS_{sid}"
    out_img = DST / "imagesTr" / f"{case}_0000.nii.gz"
    out_msk = DST / "labelsTr" / f"{case}.nii.gz"
    if not out_img.exists():
        shutil.copyfile(img, out_img)
    if not out_msk.exists():
        shutil.copyfile(msk, out_msk)
    # validate header-level shape match (cheap, no full data load)
    i, m = nib.load(out_img), nib.load(out_msk)
    assert i.shape == m.shape, f"{case}: shape mismatch {i.shape} vs {m.shape}"
    assert i.shape[0] > 0
    return case, i.shape, tuple(round(float(z), 4) for z in i.header.get_zooms())


def main():
    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)
    pairs = find_pairs()
    print(f"found {len(pairs)} image/mask pairs", flush=True)

    with Pool(8) as p:
        results = p.map(process_one, pairs, chunksize=8)

    shapes = {}
    for case, shape, zooms in results:
        shapes.setdefault((shape, zooms), []).append(case)
    print(f"distinct (shape, spacing) combos: {len(shapes)}")
    for k, v in sorted(shapes.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"  {k}: {len(v)} cases")

    # sanity: check label values on a sample of masks
    bad = []
    for case, _, _ in results[:: max(1, len(results) // 60)]:
        m = nib.load(DST / "labelsTr" / f"{case}.nii.gz").get_fdata()
        u = np.unique(m)
        if not set(u.tolist()) <= {0.0, 1.0}:
            bad.append((case, u.tolist()[:10]))
    if bad:
        print("[WARN] non-binary masks found:", bad, flush=True)
    else:
        print("sampled masks are binary {0,1}")

    dataset_json = {
        "channel_names": {"0": "T1w"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(results),
        "file_ending": ".nii.gz",
        "name": "ATLAS",
        "description": "ATLAS R3.0 stroke lesion (infarct core) segmentation, T1w",
    }
    with open(DST / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"wrote {DST / 'dataset.json'} with numTraining={len(results)}")


if __name__ == "__main__":
    sys.exit(main())
