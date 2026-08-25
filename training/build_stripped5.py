"""Build a lean (optimizer/EMA/logging-stripped) copy of the 5 nnU-Net model dirs."""
import os, torch, shutil

SRC = os.environ.get('ATLAS_MODELS_ROOT', os.path.join(
    os.environ.get('nnUNet_results', 'nnunet_results'), 'Dataset501_ATLAS'))
DST = os.environ.get('STRIPPED_OUT', '/tmp/model_stripped5')
KEEP = ["network_weights", "trainer_name", "init_args", "inference_allowed_mirroring_axes", "current_epoch"]
MODELS = [
    "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "nnUNetTrainerMedNeXt_S_kernel3__nnUNetPlans__3d_fullres",
    "nnUNetTrainerMedNeXt_B_kernel3__nnUNetPlans__3d_fullres",
    "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
    "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
]

os.makedirs(DST, exist_ok=True)
for d in MODELS:
    s = os.path.join(SRC, d)
    t = os.path.join(DST, d)
    os.makedirs(t, exist_ok=True)
    for f in ["plans.json", "dataset.json", "dataset_fingerprint.json"]:
        if os.path.isfile(os.path.join(s, f)):
            shutil.copy(os.path.join(s, f), os.path.join(t, f))
    nfold = 0
    for fold in sorted(os.listdir(s)):
        fp = os.path.join(s, fold)
        if not os.path.isdir(fp) or not fold.startswith("fold_"):
            continue
        ck = os.path.join(fp, "checkpoint_best.pth")
        if not os.path.isfile(ck):
            continue
        ft = os.path.join(t, fold)
        os.makedirs(ft, exist_ok=True)
        full = torch.load(ck, map_location="cpu")
        torch.save({k: full[k] for k in KEEP}, os.path.join(ft, "checkpoint_best.pth"))
        nfold += 1
        print(f"  {d}/{fold}: {os.path.getsize(ck)/1e6:.0f}MB -> {os.path.getsize(os.path.join(ft, 'checkpoint_best.pth'))/1e6:.0f}MB", flush=True)
    print(f"{d}: {nfold} folds", flush=True)
print("DONE", flush=True)