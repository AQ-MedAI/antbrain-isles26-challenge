"""Fine-tune trainer: ResEnc-M initialized from the from-scratch ATLAS checkpoint,
plus SubtleLesionTransform for low-contrast (subacute) infarct learning.

- Loads the FULL network (incl. seg head) from the fold-matched scratch checkpoint
- 200 epochs, SGD lr=1e-3 + PolyLR (fine-tuning schedule)
- Inserts SubtleLesionTransform before the DS-downsampling transform
- Results go to a separate folder (class name) so the scratch results stay intact
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subtle_lesion_transform import SubtleLesionTransform

from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerResEncM_SubtleFT(nnUNetTrainer):
    # Fold-matched scratch checkpoint used for initialisation. Override the root
    # with the nnUNet_results environment variable (standard nnU-Net variable).
    FT_INIT_TEMPLATE = (os.path.join(os.environ.get('nnUNet_results', 'nnunet_results'),
                        'Dataset501_ATLAS/'
                        'nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/'
                        'fold_{fold}/checkpoint_best.pth'))

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
        self.initial_lr = 1e-3

    def initialize(self):
        super().initialize()
        ckpt = self.FT_INIT_TEMPLATE.format(fold=self.fold)
        self.print_to_log_file(f'SubtleFT: loading full weights from {ckpt}')
        saved = torch.load(ckpt, map_location=self.device, weights_only=False)
        mod = self.network
        if isinstance(mod, torch._dynamo.OptimizedModule):
            mod = mod._orig_mod
        mod.load_state_dict(saved['network_weights'], strict=True)
        self.print_to_log_file(f"SubtleFT: loaded (source epoch {saved.get('current_epoch', '?')})")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def get_training_transforms(self, *args, **kwargs):
        base = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        assert isinstance(base, ComposeTransforms)
        ts = list(base.transforms)
        insert_at = len(ts)
        for i, t in enumerate(ts):
            if isinstance(t, DownsampleSegForDSTransform):
                insert_at = i
                break
        ts.insert(insert_at, SubtleLesionTransform(
            p_attenuate=0.5, alpha_range=(0.25, 0.7),
            p_transplant=0.3, transplant_alpha_range=(0.2, 0.6),
        ))
        return ComposeTransforms(ts)
