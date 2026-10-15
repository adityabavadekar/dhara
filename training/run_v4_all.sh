#!/usr/bin/env bash
# v4: prep (uint8, water-only training tiles, fixed eval tiles, 10 GB cap), smoke test, full run.
# Detached-friendly; logs to /kaggle/working. Usage on the GPU box: bash run_v4_all.sh
set -euo pipefail
cd "$(dirname "$0")"
CFG=configs/unet_r50_imagenet_v4.yaml
setsid bash sysmon.sh /kaggle/working/sysmon.log < /dev/null > /dev/null 2>&1 &
EVAL=279,437,520,559,1111003,1111008,277,445,561,562,1111002,1111007,1111013,205,321,411,498
bash prep_all.sh /tmp/ks_v4 "--per-event-cap 500 --water-only --uint8 --max-gb 10 --eval-events $EVAL --eval-cap 200" || true
echo "prepared: $(ls /tmp/ks_v4/*.done | wc -l)/17 shards, $(du -sh /tmp/ks_v4 | cut -f1)"
RUN=/tmp/runs/unet_r50_imagenet_v4
if [ ! -f "$RUN/last.pt" ]; then
  echo "== smoke test"
  python train.py --config "$CFG" --epochs 1 --limit 64
  rm -rf "$RUN"
fi
echo "== full run"
torchrun --standalone --nproc_per_node="$(python -c 'import torch; print(torch.cuda.device_count())')" train.py --config "$CFG"
echo "ALL DONE"
