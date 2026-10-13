"""Comprehensive Benchmark: Public Hugging Face Sen1Floods11 Models vs Dhara Custom SOTA Models.

Evaluates:
  1. Hugging Face: Governor6191/sar-flood-extent-unet-resnet34 (Trained on Sen1Floods11, 2-channel VV/VH)
  2. Dhara U-Net ResNet-50 (Trained on Kuro Siwo 6-channel multi-temporal)
  3. Dhara FPN ResNet-50 (Trained on Kuro Siwo 6-channel multi-temporal)
  4. Dhara Physics-Guided SAR U-Net (Trained with Differential Specular & Cross-Pol Gating)
  5. Dhara DeepLabV3+ ResNet-50 (Trained with Atrous Spatial Pyramid Pooling)

Outputs comprehensive comparative metrics table & JSON results.
"""

import json
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from huggingface_hub import hf_hub_download


def compute_binary_water_metrics(pred_water: np.ndarray, y: np.ndarray):
    """Evaluate binary water detection (perm water + flood) against ground truth."""
    valid = y != 255
    gt_water = ((y == 1) | (y == 2)) & valid
    pred_w = (pred_water == 1) & valid

    tp = int((pred_w & gt_water).sum())
    fp = int((pred_w & ~gt_water).sum())
    fn = int((~pred_w & gt_water).sum())

    iou = tp / max(1, tp + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return {
        "iou": round(iou, 4),
        "f1": round(f1, 4),
        "prec": round(prec, 4),
        "rec": round(rec, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_flood_specific_metrics(pred_class: np.ndarray, y: np.ndarray):
    """Evaluate 3-class flood specific metrics (0: dry, 1: perm water, 2: flood)."""
    valid = y != 255
    gt_flood = (y == 2) & valid
    pred_f = (pred_class == 2) & valid

    tp = int((pred_f & gt_flood).sum())
    fp = int((pred_f & ~gt_flood).sum())
    fn = int((~pred_f & gt_flood).sum())

    iou = tp / max(1, tp + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return {"iou": round(iou, 4), "f1": round(f1, 4), "prec": round(prec, 4), "rec": round(rec, 4)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Hugging Face Sen1Floods11 Model
    print("Loading HF Governor6191/sar-flood-extent-unet-resnet34...")
    hf_path = hf_hub_download("Governor6191/sar-flood-extent-unet-resnet34", "model.pt")
    hf_ckpt = torch.load(hf_path, map_location=device, weights_only=False)
    hf_model = smp.create_model(
        "unet", encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=2
    ).to(device)
    hf_model.load_state_dict(hf_ckpt["model_state"])
    hf_model.eval()

    # 2. Load Dhara Models from /kaggle/working/dhara/runs
    runs_dir = Path("/kaggle/working/dhara/runs")
    our_models = {}

    for run_name, arch, enc, is_phys in [
        ("unet_r50_scratch_v1", "Unet", "resnet50", False),
        ("fpn_r50_v1", "FPN", "resnet50", False),
        ("physics_unet_r50_v1", "Unet", "resnet50", True),
        ("deeplabv3p_r50_v1", "DeepLabV3Plus", "resnet50", False),
    ]:
        p = runs_dir / run_name / "best.pt"
        if p.exists():
            print(f"Loading Dhara model: {run_name}...")
            ck = torch.load(p, map_location=device, weights_only=False)
            model_cls = getattr(smp, arch)
            base = model_cls(encoder_name=enc, encoder_weights=None, in_channels=6, classes=3)
            if is_phys:
                from train import PhysicsGuidedSARModel

                model = PhysicsGuidedSARModel(base).to(device)
            else:
                model = base.to(device)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            our_models[run_name] = model

    # 3. Load Validation Data Shards
    data_dir = Path("/tmp/ks_arr")
    x_files = sorted(data_dir.glob("*_x.npy"))
    y_files = sorted(data_dir.glob("*_y.npy"))

    print(f"Found {len(x_files)} data shards. Evaluating on shard 0...")
    xa = np.load(x_files[0], mmap_mode="r")
    ya = np.load(y_files[0], mmap_mode="r")

    # Evaluate on first 200 samples
    n = min(200, len(xa))
    print(f"Evaluating across {n} real disaster tiles...")

    results = {"hf_sen1floods11_resnet34": {"tp": 0, "fp": 0, "fn": 0}}
    for k in our_models:
        results[k] = {"tp": 0, "fp": 0, "fn": 0, "f_tp": 0, "f_fp": 0, "f_fn": 0}

    # Norms
    # HF Sen1Floods11 standardization
    hf_mean = torch.tensor([-11.0, -18.0], device=device).view(1, 2, 1, 1)
    hf_std = torch.tensor([4.5, 4.5], device=device).view(1, 2, 1, 1)

    # Dhara Kuro Siwo 6-ch norm
    norm_p = runs_dir / "physics_unet_r50_v1" / "norm.json"
    if norm_p.exists():
        with open(norm_p) as f:
            dnorm = json.load(f)
        dh_mean = torch.tensor(dnorm["mean"], dtype=torch.float32, device=device).view(1, 6, 1, 1)
        dh_std = torch.tensor(dnorm["std"], dtype=torch.float32, device=device).view(1, 6, 1, 1)
    else:
        dh_mean = torch.zeros(1, 6, 1, 1, device=device)
        dh_std = torch.ones(1, 6, 1, 1, device=device)

    with torch.no_grad():
        for i in range(n):
            raw_x = torch.from_numpy(np.array(xa[i], dtype=np.float32)).unsqueeze(0).to(device)
            y = ya[i]
            valid = y != 255
            water_gt = ((y == 1) | (y == 2)) & valid
            flood_gt = (y == 2) & valid

            # 1. HF Model (Channels 0 & 1: post VV, post VH)
            hf_in = (raw_x[:, :2] - hf_mean) / hf_std
            hf_pred = hf_model(hf_in).argmax(dim=1).squeeze(0).cpu().numpy()
            hf_w = (hf_pred == 1) & valid

            results["hf_sen1floods11_resnet34"]["tp"] += int((hf_w & water_gt).sum())
            results["hf_sen1floods11_resnet34"]["fp"] += int((hf_w & ~water_gt).sum())
            results["hf_sen1floods11_resnet34"]["fn"] += int((~hf_w & water_gt).sum())

            # 2. Dhara Models (6 channels: multi-temporal pre/pre/post)
            dh_in = (raw_x - dh_mean) / dh_std
            for k, m in our_models.items():
                pred_c = m(dh_in).argmax(dim=1).squeeze(0).cpu().numpy()
                our_w = ((pred_c == 1) | (pred_c == 2)) & valid
                our_f = (pred_c == 2) & valid

                results[k]["tp"] += int((our_w & water_gt).sum())
                results[k]["fp"] += int((our_w & ~water_gt).sum())
                results[k]["fn"] += int((~our_w & water_gt).sum())

                results[k]["f_tp"] += int((our_f & flood_gt).sum())
                results[k]["f_fp"] += int((our_f & ~flood_gt).sum())
                results[k]["f_fn"] += int((~our_f & flood_gt).sum())

    # Format Output Table
    print("\n" + "=" * 85)
    print("EMPIRICAL BENCHMARK: HUGGING FACE SEN1FLOODS11 MODEL VS DHARA MULTI-TEMPORAL MODELS")
    print("=" * 85)
    print(
        f"{'Model Architecture':<36} | {'Water Recall':<12} | {'Water Prec':<12} | {'Water IoU':<10} | {'Water F1':<10}"
    )
    print("-" * 85)

    summary = {}
    for name, r in results.items():
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        iou = tp / max(1, tp + fp + fn)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        summary[name] = {
            "water_iou": round(iou, 4),
            "water_f1": round(f1, 4),
            "water_prec": round(prec, 4),
            "water_rec": round(rec, 4),
        }
        print(
            f"{name:<36} | {rec * 100:>10.2f}% | {prec * 100:>10.2f}% | {iou:>10.4f} | {f1:>10.4f}"
        )

    print("=" * 85)
    out_file = Path("/kaggle/working/hf_benchmark_summary.json")
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"Saved benchmark summary to {out_file}")


if __name__ == "__main__":
    main()
