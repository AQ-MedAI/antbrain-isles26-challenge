# Method details (extended)

> Extended description backing the 300-word challenge abstract. Numbers refer to
> 5-fold out-of-fold (OOF) evaluation on the training set (n=1453), threshold 0.5,
> no post-processing.

## Framework

Everything runs inside nnU-Net v2 (2.8.1): automatic fingerprint/plan generation,
on-the-fly data augmentation, deep supervision, and EMA-based best-checkpoint
selection. All models use the `3d_fullres` configuration of the same dataset
(Dataset501, single T1w channel, z-score normalisation, patch 128³ except
ResEnc-L at 160×192×160, 1000 epochs).

## Architectures (the 5 ensemble members)

| member | trainer / plans | notes |
|---|---|---|
| nnUNet | `nnUNetTrainer` / `nnUNetPlans` | default plain-conv U-Net baseline |
| MedNeXt-S-k3 | `nnUNetTrainerMedNeXt_S_kernel3` | MedNeXt small, kernel 3, ~10.5M params, ported to nnU-Net v2 (`mednext_v2/`) |
| MedNeXt-B-k3 | `nnUNetTrainerMedNeXt_B_kernel3` | MedNeXt base, kernel 3 |
| ResEnc-M | `nnUNetTrainer` / `nnUNetResEncUNetMPlans` | residual-encoder preset M ([1,3,4,6,6,6] blocks) |
| ResEnc-L | `nnUNetTrainer` / `nnUNetResEncUNetLPlans` | residual-encoder preset L, larger patch |

MedNeXt port notes (relevant for reproduction):
- deep-supervision scales must be relative (e.g. `[0.5,0.5,0.5]`), not multiplicative;
- keep `nnUNet_compile=False` (kernel-5 variants misbehave under torch.compile;
  we only use kernel-3);
- `cudnn.benchmark=True` gives ~2.6× speedup on depthwise convolutions; the long
  first-epoch autotune is expected.

## Training

Each member is trained on all 5 folds (`splits_final.json`, seed 12345) for 1000
epochs; the checkpoint with the best validation EMA foreground Dice
(`checkpoint_best.pth`) is kept.

## Inference / ensembling

1. Per model: `nnUNetv2_predict -f 0 1 2 3 4 --save_probabilities -chk checkpoint_best.pth`
   (nnU-Net averages softmax over folds). Mirroring/TTA disabled — measured
   ΔPR-AUC ≈ −0.001 on OOF, i.e. no gain (the exported probabilities already
   average the default nnU-Net mirroring behaviour; extra explicit TTA did not help).
2. Cross-model: `nnUNetv2_ensemble` over the 5 model folders (equal weights).
   All members share the same preprocessed target grid, so probability maps are
   averaged directly; argmax of the averaged softmax ≡ threshold 0.5 for the
   binary case.
3. No connected-component filtering: ATLAS lesions are frequently multifocal, and
   component-size/count filtering reduced Dice in OOF evaluation — deliberately
   omitted.

## OOF evaluation protocol

Metrics computed on the preprocessed grid against `gt_segmentations`:
Dice; absolute volume difference (AVD, mL); absolute lesion-count difference
(ALCD; 26-connectivity components); lesion-wise F1 (any-overlap matching);
PR-AUC (sklearn average precision on the soft maps). See `eval/` for scripts.

## Hardware / versions verified

- torch 2.4.1 (pinned), nnunetv2 2.8.1, python 3.10
- NVIDIA H20 96GB and A100 80GB; single GPU per training run
- Known pitfall: do not let package mirrors upgrade torch past the pinned version.
