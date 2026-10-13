#!/usr/bin/env bash
# One-shot, resumable: install deps → convert Kuro Siwo (skips finished shards) → train (resumes from last.pt).
# Usage on the GPU box:  bash run_all.sh [extra train.py args]
set -euo pipefail
cd "$(dirname "$0")"
pip install -q segmentation-models-pytorch==0.5.0 onnx onnxscript pyyaml
python prep_kurosiwo.py --splits train_GRD --max-shards 1 --max-samples 1500
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
torchrun --standalone --nproc_per_node="$NGPU" train.py --config configs/unet_r50.yaml "$@"
