"""Subtle-lesion synthesis transform for teaching low-contrast infarct detection.

Two operations on z-scored image + binary seg (keys 'data' (C,Z,Y,X), 'seg' (1,Z,Y,X)):

1) Contrast attenuation (p_attenuate): blend existing GT lesion intensities toward
   the surrounding tissue mean with alpha ~ U(alpha_range). Existing obvious
   (chronic, dark) lesions become subtle ones; label unchanged.

2) Lesion transplantation (p_transplant): copy the sample's own largest lesion
   component to a new random location inside brain tissue, blended with a
   (typically stronger) attenuation and feathered boundary; seg updated to 1 there.
   Adds extra subtle-lesion positives per batch.
"""
from typing import Tuple

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label as cc_label

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class SubtleLesionTransform(BasicTransform):
    def __init__(self,
                 p_attenuate: float = 0.5,
                 alpha_range: Tuple[float, float] = (0.25, 0.7),
                 p_transplant: float = 0.3,
                 transplant_alpha_range: Tuple[float, float] = (0.2, 0.6),
                 ring_width: int = 3,
                 noise_std: float = 0.05,
                 tissue_z_range: Tuple[float, float] = (-1.5, 1.5),
                 min_lesion_voxels: int = 30):
        super().__init__()
        self.p_attenuate = p_attenuate
        self.alpha_range = alpha_range
        self.p_transplant = p_transplant
        self.transplant_alpha_range = transplant_alpha_range
        self.ring_width = ring_width
        self.noise_std = noise_std
        self.tissue_z_range = tissue_z_range  # z-scored brain-tissue window for transplant targets
        self.min_lesion_voxels = min_lesion_voxels

    def _ring_mean(self, img, lesion_mask):
        brain = img > (img.min() + 1e-4)  # skull-stripped background is the constant min value
        ring = binary_dilation(lesion_mask, iterations=self.ring_width) & ~lesion_mask & brain
        if ring.sum() < 10:
            return None
        return float(img[ring].mean())

    def _attenuate(self, data, seg):
        lesion = seg[0] > 0.5
        if lesion.sum() < self.min_lesion_voxels:
            return data, seg
        mu = self._ring_mean(data[0], lesion)
        if mu is None:
            return data, seg
        a = np.random.uniform(*self.alpha_range)
        for c in range(data.shape[0]):
            vals = data[c][lesion]
            data[c][lesion] = mu + a * (vals - mu) + np.random.normal(0, self.noise_std, vals.shape)
        return data, seg

    def _transplant(self, data, seg):
        lesion = seg[0] > 0.5
        if lesion.sum() < self.min_lesion_voxels:
            return data, seg
        # largest connected component of the lesion
        lbl, n = cc_label(lesion)
        if n == 0:
            return data, seg
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        src = lbl == sizes.argmax()
        if src.sum() < self.min_lesion_voxels:
            return data, seg

        # target candidates: tissue-like z, brain (not skull-stripped background),
        # clean seg==0 (never on ignore label -1), away from existing lesion
        away = ~binary_dilation(lesion, iterations=5)
        brain = data[0] > (data[0].min() + 1e-4)
        tz = ((data[0] >= self.tissue_z_range[0]) & (data[0] <= self.tissue_z_range[1])
              & (seg[0] == 0) & brain & away)
        coords = np.argwhere(tz)
        if len(coords) < 10:
            return data, seg

        src_coords = np.argwhere(src)
        src_center = src_coords.mean(0)
        tgt_center = coords[np.random.randint(len(coords))].astype(float)
        shift = np.round(tgt_center - src_center).astype(int)

        new_coords = src_coords + shift
        valid = ((new_coords >= 0).all(1) &
                 (new_coords < np.array(data.shape[1:])).all(1))
        new_coords = new_coords[valid]
        src_coords_v = src_coords[valid]
        if len(new_coords) < self.min_lesion_voxels:
            return data, seg

        a = np.random.uniform(*self.transplant_alpha_range)
        idx_new = tuple(new_coords.T)
        idx_src = tuple(src_coords_v.T)
        # target neighborhood mean for blending
        tgt_mask = np.zeros(data.shape[1:], bool); tgt_mask[idx_new] = True
        mu = self._ring_mean(data[0], tgt_mask)
        if mu is None:
            return data, seg
        for c in range(data.shape[0]):
            data[c][idx_new] = mu + a * (data[c][idx_src] - mu) + np.random.normal(0, self.noise_std, len(idx_new[0]))
        seg[0][idx_new] = 1
        # feather the new lesion boundary a little
        for c in range(data.shape[0]):
            ch = data[c]
            sm = gaussian_filter(ch, 0.5)
            edge = binary_dilation(tgt_mask, iterations=1) ^ tgt_mask
            ch[edge] = sm[edge]
        return data, seg

    def apply(self, data_dict, **params):
        data = data_dict['data']
        seg = data_dict['seg']
        if np.random.rand() < self.p_attenuate:
            data, seg = self._attenuate(data, seg)
        if np.random.rand() < self.p_transplant:
            data, seg = self._transplant(data, seg)
        data_dict['data'] = data
        data_dict['seg'] = seg
        return data_dict
