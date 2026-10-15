#!/usr/bin/env bash
# Smoke test, then the full UNet-ResNet34 (ImageNet encoder) run on all converted Kuro Siwo events.
# If a checkpoint was put back into the run folder (after a session restart), resume from it instead.
# Usage on the GPU box, after prep_all.sh: bash train_v2.sh
set -euo pipefail
cd "$(dirname "$0")"
CFG=configs/unet_r34_imagenet.yaml
RUN=/tmp/runs/unet_r34_imagenet_v2
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
if [ -f "$RUN/last.pt" ]; then
  echo "== resuming from $RUN/last.pt"
else
  echo "== smoke test"
  python train.py --config "$CFG" --epochs 1 --limit 64
  rm -rf "$RUN"
fi
echo "== full run on $NGPU GPU(s)"
torchrun --standalone --nproc_per_node="$NGPU" train.py --config "$CFG"
