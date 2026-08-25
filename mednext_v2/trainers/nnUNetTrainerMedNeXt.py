"""MedNeXt trainers for nnUNet v2.

Ported from the official MedNeXt repo (MIC-DKFZ/MedNeXt, nnUNet v1 trainers:
nnunet_mednext/training/network_training/MedNeXt/nnUNetTrainerV2_MedNeXt.py).

Key adaptations for nnUNet v2:
- architecture is built via build_network_architecture (static method)
- MedNeXt has a fixed depth (4 downsamplings -> DS outputs at scales
  1, 1/2, 1/4, 1/8, 1/16), so _get_deep_supervision_scales is fixed instead of
  being derived from the plans' pool_op_kernel_sizes
- MedNeXt.forward returns a list of outputs whenever do_ds=True (train AND eval),
  which matches nnUNet v2's validation_step contract (output[0] when DS enabled);
  inference builds the network with enable_deep_supervision=False -> single tensor
- optimizer: AdamW (as in the MedNeXt paper) + nnUNet v2 PolyLRScheduler
"""
import os
import sys

import torch
from torch._dynamo import OptimizedModule

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mednext_arch.MedNextV1 import MedNeXt

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class _nnUNetTrainerMedNeXtBase(nnUNetTrainer):
    # subclass overrides these
    mednext_exp_r = 2
    mednext_kernel_size = 3
    mednext_block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

    @classmethod
    def build_network_architecture(cls, plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        # ALWAYS create all deep-supervision output heads (out_1..out_4) so that checkpoints
        # saved during training (with deep supervision enabled) load with a strict state_dict.
        # Whether forward returns a single tensor or the DS list is controlled by `do_ds`
        # below — mirroring how standard nnU-Net (PlainConvUNet) keeps the heads present
        # regardless of the DS flag. Without this, nnUNetPredictor builds the inference
        # network with enable_deep_supervision=False and the strict load_state_dict fails on
        # the out_1..out_4 keys present in the checkpoint.
        net = MedNeXt(
            in_channels=num_input_channels,
            n_channels=32,
            n_classes=num_output_channels,
            exp_r=cls.mednext_exp_r,
            kernel_size=cls.mednext_kernel_size,
            deep_supervision=True,
            do_res=True,
            do_res_up_down=True,
            block_counts=list(cls.mednext_block_counts),
        )
        net.do_ds = enable_deep_supervision
        return net

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            # MedNeXt: 4 downsamplings -> 5 outputs (full res + 4 DS levels).
            # NOTE: nnUNet v2 expects RELATIVE target sizes (fractions), not factors!
            return [[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.0625, 0.0625, 0.0625]]
        return None  # for train and val_transforms

    def set_deep_supervision_enabled(self, enabled: bool):
        """MedNeXt has no `.decoder.deep_supervision` (that API is specific to
        dynamic-network-architectures). Toggling MedNeXt's `do_ds` makes forward
        return either [main + DS outputs] or just the main output. All out_N
        layers always exist, so weights stay compatible either way."""
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        for m in mod.modules():
            if isinstance(m, MedNeXt):
                m.do_ds = enabled

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=1e-3,
            weight_decay=self.weight_decay,
            eps=1e-4,  # 1e-8 might cause nans in fp16 (MedNeXt repo note)
        )
        lr_scheduler = PolyLRScheduler(optimizer, 1e-3, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainerMedNeXt_S_kernel3(_nnUNetTrainerMedNeXtBase):
    mednext_exp_r = 2
    mednext_kernel_size = 3
    mednext_block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]


class nnUNetTrainerMedNeXt_B_kernel3(_nnUNetTrainerMedNeXtBase):
    mednext_exp_r = [2, 3, 4, 4, 4, 4, 4, 3, 2]
    mednext_kernel_size = 3
    mednext_block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]


class nnUNetTrainerMedNeXt_S_kernel5(_nnUNetTrainerMedNeXtBase):
    mednext_exp_r = 2
    mednext_kernel_size = 5
    mednext_block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]


class nnUNetTrainerMedNeXt_B_kernel5(_nnUNetTrainerMedNeXtBase):
    mednext_exp_r = [2, 3, 4, 4, 4, 4, 4, 3, 2]
    mednext_kernel_size = 5
    mednext_block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]
