#!/usr/bin/env bash
# v7 (v4 + Lovász/focal Tversky loss) on GPU 0 and v8 (UNet++ R50) on GPU 1, side by side, then
# all scored with eval_kurosiwo_onnx.py next to v2/v3/v4/v6 on the same tiles (validation-tuned weights too).
# Needs the v4 data in /tmp/ks_v4 (prep command at the top of configs/unet_r50_imagenet_v4.yaml).
# Usage on the GPU box: setsid bash run_v7_v8.sh > /kaggle/working/run_v7_v8.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
[ "$(ls /tmp/ks_v4/*.done 2>/dev/null | wc -l)" -ge 17 ] || { echo "v4 data missing in /tmp/ks_v4" >&2; exit 1; }
pip install -q segmentation-models-pytorch==0.5.0 onnx onnxscript pyyaml onnxruntime
setsid bash sysmon.sh /kaggle/working/sysmon.log < /dev/null > /dev/null 2>&1 &
train_one() {  # gpu config
  local gpu=$1 cfg=$2 run
  run=/tmp/runs/$(grep '^run_name:' "$cfg" | awk '{print $2}')
  if [ ! -f "$run/last.pt" ]; then
    CUDA_VISIBLE_DEVICES=$gpu python train.py --config "$cfg" --epochs 1 --limit 64 || return 1
    rm -rf "$run"
  fi
  CUDA_VISIBLE_DEVICES=$gpu python train.py --config "$cfg"
}
train_one 0 configs/unet_r50_imagenet_v7.yaml > /kaggle/working/run_v7.log 2>&1 &
p7=$!
train_one 1 configs/unetplusplus_r50_imagenet_v8.yaml > /kaggle/working/run_v8.log 2>&1 &
p8=$!
wait $p7; echo "v7 exit $?"
wait $p8; echo "v8 exit $?"
# Score on GPU when onnxruntime-gpu loads; eval_kurosiwo_onnx.py falls back to CPU otherwise.
pip uninstall -y -q onnxruntime && pip install -q onnxruntime-gpu || pip install -q onnxruntime
M=/tmp/mods
python eval_kurosiwo_onnx.py --config configs/unet_r50_imagenet_v4.yaml --tune \
  --model v2=$M/unet_r34_imagenet_v2 --model v3=$M/unet_r50_imagenet_v3 \
  --model v4=$M/unet_r50_imagenet_v4 --model v6=$M/unet_r50_imagenet_v6 \
  --model v7=/tmp/runs/unet_r50_imagenet_v7 --model v8=/tmp/runs/unetplusplus_r50_imagenet_v8 \
  --out /kaggle/working/compare_v7_v8.json
echo "ALL DONE"
