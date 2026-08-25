"""Name-only subclass of nnUNetTrainer so that ResEnc-L MAE-pretrained full-FT runs
write to a separate results folder (nnUNetTrainerResEncL_MAE_ft__nnUNetResEncUNetLPlans__3d_fullres)
instead of colliding with the from-scratch ResEnc-L runs.

Used with: nnUNetv2_train 501 3d_fullres <fold> -p nnUNetResEncUNetLPlans \
              -tr nnUNetTrainerResEncL_MAE_ft \
              -pretrained_weights <path_to_ResEnc-L_MAE_checkpoint_final.pth>
(nnUNet's load_pretrained_weights transfers encoder+decoder, skips .seg_layers.)
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerResEncL_MAE_ft(nnUNetTrainer):
    pass
