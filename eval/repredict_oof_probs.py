#!/usr/bin/env python
"""Re-run out-of-fold validation prediction for one fold with softmax probabilities
saved (.npz), using checkpoint_best. Writes to a separate eval directory so the
original results stay untouched.

Usage:
  python repredict_oof_probs.py <model_dir> <trainer_name> <plans_id> <fold> <out_base>

Example:
  python repredict_oof_probs.py nnUNetTrainer__nnUNetPlans__3d_fullres nnUNetTrainer nnUNetPlans 0 ./eval_preds
"""
import os
import sys

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


def main():
    model_dir, trainer_name, plans_id, fold, out_base = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    dataset = 'Dataset501_ATLAS'

    trainer_cls = recursive_find_trainer_class_by_name(trainer_name)
    plans = load_json(os.path.join(nnUNet_preprocessed, dataset, plans_id + '.json'))
    plans["continue_training"] = False  # nnUNetTrainer.__init__ pops this key (set by run_training normally)
    dataset_json = load_json(os.path.join(nnUNet_preprocessed, dataset, 'dataset.json'))

    trainer = trainer_cls(plans=plans, configuration='3d_fullres', fold=fold, dataset_json=dataset_json)
    trainer.initialize()
    ckpt = os.path.join(nnUNet_results, dataset, model_dir, f'fold_{fold}', 'checkpoint_best.pth')
    print(f'loading {ckpt}', flush=True)
    trainer.load_checkpoint(ckpt)

    # build val dataloader (needed by perform_actual_validation)
    trainer.dataloader_train, trainer.dataloader_val = trainer.get_dataloaders()

    # redirect output so we don't touch the original results
    trainer.output_folder = os.path.join(out_base, model_dir, f'fold_{fold}')
    trainer.perform_actual_validation(save_probabilities=True)
    print(f'DONE {model_dir} fold {fold}', flush=True)


if __name__ == '__main__':
    main()
