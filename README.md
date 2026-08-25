# ISLES'26 — Ensembled CNNs for Stroke Lesion Segmentation (T1w)

Official code for our ISLES'26 (MICCAI) challenge submission.
The method is a **5-architecture × 5-fold softmax-averaging ensemble** built on the
nnU-Net v2 self-configuring framework, with MedNeXt (ConvNeXt-style large-kernel
blocks) and residual-encoder U-Net variants ported into the nnU-Net v2 trainer API.

> TODO(team): add GC team name, Zenodo/HF weights link, and citation before making public.

## Method in one paragraph

Five heterogeneous 3D full-resolution architectures — nnU-Net default U-Net,
MedNeXt-S (kernel 3), MedNeXt-B (kernel 3), ResEnc-M and ResEnc-L — are each trained
with 5-fold cross-validation (1000 epochs, checkpoint = best validation EMA).
At inference, softmax probabilities are averaged over the 5 folds per model and then
over the 5 models; binary masks are obtained at a lesion-probability threshold of 0.5.
No test-time augmentation and no connected-component post-processing are used —
both were evaluated on the full 1453-case out-of-fold predictions and found to give
no gain (TTA) or to degrade Dice (component filtering), so they are deliberately omitted.
See [docs/METHOD.md](docs/METHOD.md) for details.

## Out-of-fold results (n=1453, 5-fold CV, threshold 0.5)

| configuration | Dice | AVD (mL) ↓ | ALCD ↓ | Lesion-F1 | PR-AUC |
|---|---|---|---|---|---|
| **5-model ensemble (submitted)** | **0.6660** | 4.95 | **1.82** | **0.6852** | **0.7729** |
| MedNeXt-S (best single) | 0.6613 | 5.22 | 1.88 | 0.6736 | 0.7596 |
| nnUNet baseline | 0.6523 | 5.48 | 1.95 | 0.6617 | 0.7463 |

Ensemble vs best single model: ΔDice = +0.0047, paired Wilcoxon p ≈ 1e-10.

## Installation

```bash
conda create -n isles26 python=3.10 -y && conda activate isles26
pip install -r requirements.txt   # torch is pinned to 2.4.1; see docs/METHOD.md
```

## Inference (new data)

```bash
# 1) set environment (edit paths first)
cp env_template.sh env.sh && $EDITOR env.sh && source env.sh
export nnUNet_extTrainer=$PWD/mednext_v2/trainers   # MedNeXt trainer discovery

# 2) download model weights (5 models x 5 folds, ~15 GB) into $nnUNet_results/Dataset501_ATLAS/
#    TODO(team): Zenodo/HuggingFace link

# 3) run: input dir with <case>_0000.nii.gz (single T1w channel)
bash inference/predict_ensemble.sh /path/to/inputs /path/to/outputs
# final masks: /path/to/outputs/ensemble/<case>.nii.gz
```

## Repository layout

```
mednext_v2/trainers/   MedNeXt nnU-Net v2 trainers + pure-torch architecture
training/              training / chaining / resume scripts (5 models x 5 folds)
eval/                  OOF evaluation: ensemble, challenge metrics, Wilcoxon
inference/             predict_ensemble.sh — the submitted inference pipeline
convert_atlas_to_nnunet.py   BIDS -> nnU-Net dataset conversion
docs/METHOD.md         extended method description
```

## Data

Training used the ATLAS R3.0 dataset, subject to its own data-use agreement.
We cannot redistribute the data; request access from the official source and use
`convert_atlas_to_nnunet.py` to reproduce the nnU-Net dataset (Dataset501).

## License

Apache 2.0 — see [LICENSE](LICENSE).
