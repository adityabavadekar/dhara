#!/usr/bin/env bash
# Convert every labelled Kuro Siwo GRD shard (5 train + 12 test) with a per-event cap, 5 in parallel.
# Usage on the GPU box: bash prep_all.sh [/tmp/ks_full]
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-/tmp/ks_full}"
EXTRA="${2:-}"  # e.g. "--per-event-cap 0 --water-only"
mkdir -p "$OUT" logs
pip install -q segmentation-models-pytorch==0.5.0 onnx onnxscript pyyaml
{ for i in 0 1 2 3 4; do echo "train_GRD:$i"; done; for i in $(seq 0 11); do echo "test_GRD:$i"; done; } |
  xargs -P 5 -I{} sh -c "python prep_kurosiwo.py --out $OUT --splits train_GRD test_GRD --shard {} $EXTRA > logs/prep_\$(echo {} | tr : _).log 2>&1"
n=$(ls "$OUT"/*.done | wc -l)
echo "shards converted: $n/17"
[ "$n" -eq 17 ]
