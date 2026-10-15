#!/usr/bin/env bash
# v5: add Sen1Floods11 tiles to the v4 data folder, smoke test, full run. Needs /tmp/ks_v4 from v4.
set -euo pipefail
cd "$(dirname "$0")"
CFG=configs/unet_r50_imagenet_v5.yaml
ls /tmp/ks_v4/*.done > /dev/null || { echo "missing /tmp/ks_v4: run run_v4_all.sh's prep first" >&2; exit 1; }
[ -f /tmp/ks_v4/s1f11_train_p00_meta.csv ] || python prep_sen1floods11.py --out /tmp/ks_v4 --uint8
RUN=/tmp/runs/unet_r50_imagenet_v5
if [ ! -f "$RUN/last.pt" ]; then
  echo "== smoke test"
  python train.py --config "$CFG" --epochs 1 --limit 64
  rm -rf "$RUN"
fi
echo "== full run"
torchrun --standalone --nproc_per_node="$(python -c 'import torch; print(torch.cuda.device_count())')" train.py --config "$CFG"
echo "ALL DONE"
