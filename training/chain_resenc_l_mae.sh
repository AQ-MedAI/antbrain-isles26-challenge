#!/bin/bash
# ResEnc-L + OpenNeuro-MAE pretrained weights, full fine-tuning on ATLAS (GPU 0).
# Starts when ResEnc-M fold 4 finishes. Folds 0 then 1 (A/B evidence vs from-scratch).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"  # copy env_template.sh -> env.sh and edit paths first
LOG="$SCRIPT_DIR/../logs"
PLANS=nnUNetResEncUNetLPlans
TRAINER=nnUNetTrainerResEncL_MAE_ft
PRETRAINED="${RESENC_L_CKPT:?set RESENC_L_CKPT to the ResEnc-L MAE checkpoint_final.pth}"
M_F4=$nnUNet_results/Dataset501_ATLAS/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_4/checkpoint_final.pth

echo "[chain-mae-ft] waiting for ResEnc-M fold 4 to finish..."
while [ ! -f "$M_F4" ]; do sleep 300; done
echo "[chain-mae-ft] M fold 4 done at $(date), starting pretrained full-FT in 2 min"
sleep 120

export CUDA_VISIBLE_DEVICES=0
export nnUNet_def_n_proc=4
export nnUNet_compile=False

for f in 0 1; do
  FOLD_DIR=$nnUNet_results/Dataset501_ATLAS/${TRAINER}__${PLANS}__3d_fullres/fold_$f
  while [ ! -f "$FOLD_DIR/checkpoint_final.pth" ]; do
    if [ -f "$FOLD_DIR/checkpoint_latest.pth" ]; then
      # resume: nnUNet forbids combining --c with -pretrained_weights
      ARGS="--c"
    else
      ARGS="-pretrained_weights $PRETRAINED"
    fi
    echo "[chain-mae-ft] fold $f attempt ($ARGS) $(date)"
    nnUNetv2_train 501 3d_fullres $f -p $PLANS -tr $TRAINER $ARGS 2>&1 | tee -a $LOG/resenc_l_mae_ft_fold${f}.log
    [ -f "$FOLD_DIR/checkpoint_final.pth" ] && break
    echo "[chain-mae-ft] fold $f crashed, retry in 120s $(date)"; sleep 120
  done
  echo "[chain-mae-ft] fold $f DONE $(date)"
done
echo "[chain-mae-ft] ALL DONE $(date)"
