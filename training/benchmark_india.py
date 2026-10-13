"""Benchmark Models Specifically on Indian Flood Scenes from Sen1Floods11.

Evaluates performance across 14 hand-labeled Sentinel-1 flood tiles covering:
  - Bihar / Gangetic Floodplains (monsoon overbank flow)
  - Brahmaputra River Basin (Assam monsoon floodings)
  - Agricultural submerged paddy & vegetation dynamics

Compares:
  1. Hugging Face: Governor6191/sar-flood-extent-unet-resnet34 (Sen1Floods11 2-ch)
  2. Dhara U-Net ResNet-50 (Kuro Siwo 6-ch)
  3. Dhara FPN ResNet-50 (Kuro Siwo 6-ch)
  4. Dhara Physics-Guided SAR U-Net (Differential Radar Residual Head)
  5. Dhara DeepLabV3+ ResNet-50 (ASPP Multi-Scale)
  6. Dhara SegFormer mit_b0 (Vision Transformer)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from huggingface_hub import hf_hub_download


def evaluate_india_tiles():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # 1. Download & Extract all Indian flood scenes from Sen1Floods11 test split
    print("Fetching Indian flood scenes from Sen1Floods11 test split...")
    india_tiles = []
    for i in range(3):
        url = f"https://huggingface.co/datasets/pdosquet/sen1floods11-preprocessed-dl/resolve/main/data/test/test-{i:05d}-of-00003.parquet"
        df = pd.read_parquet(url)
        ind_df = df[df["region"] == "India"]
        for _, row in ind_df.iterrows():
            s1 = np.stack([np.stack(ch) for ch in row["s1"]]).astype(np.float32)  # (2, 512, 512)
            lbl = np.stack(row["label"]).astype(np.int64)  # (512, 512)
            india_tiles.append({"patch_id": row["patch_id"], "s1": s1, "label": lbl})

    print(f"Successfully loaded {len(india_tiles)} Indian flood tiles.")

    # 2. Load Hugging Face Model
    print("Loading Hugging Face Governor6191/sar-flood-extent-unet-resnet34...")
    hf_path = hf_hub_download("Governor6191/sar-flood-extent-unet-resnet34", "model.pt")
    hf_ckpt = torch.load(hf_path, map_location=device, weights_only=False)
    hf_model = smp.create_model(
        "unet", encoder_name="resnet34", encoder_weights=None, in_channels=2, classes=2
    ).to(device)
    hf_model.load_state_dict(hf_ckpt["model_state"])
    hf_model.eval()

    # 3. Load Dhara Models
    runs_dir = Path("/kaggle/working/dhara/runs")
    our_models = {}

    configs = [
        ("unet_r50_scratch_v1", "Unet", "resnet50", False),
        ("fpn_r50_v1", "FPN", "resnet50", False),
        ("physics_unet_r50_v1", "Unet", "resnet50", True),
        ("deeplabv3p_r50_v1", "DeepLabV3Plus", "resnet50", False),
        ("segformer_mitb0_v1", "Segformer", "mit_b0", False),
    ]

    for run_name, arch, enc, is_phys in configs:
        p = runs_dir / run_name / "best.pt"
        if p.exists():
            print(f"Loading Dhara model: {run_name}...")
            ck = torch.load(p, map_location=device, weights_only=False)
            model_cls = getattr(smp, arch, None)
            if model_cls is None:
                for name in dir(smp):
                    if name.lower() == arch.lower():
                        model_cls = getattr(smp, name)
                        break
            base = model_cls(encoder_name=enc, encoder_weights=None, in_channels=6, classes=3)
            if is_phys:
                from train import PhysicsGuidedSARModel

                model = PhysicsGuidedSARModel(base).to(device)
            else:
                model = base.to(device)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            our_models[run_name] = model

    # 4. Normalization tensors
    norm_p = runs_dir / "physics_unet_r50_v1" / "norm.json"
    if norm_p.exists():
        with open(norm_p) as f:
            dnorm = json.load(f)
        dh_mean = torch.tensor(dnorm["mean"], dtype=torch.float32, device=device).view(1, 6, 1, 1)
        dh_std = torch.tensor(dnorm["std"], dtype=torch.float32, device=device).view(1, 6, 1, 1)
    else:
        dh_mean = torch.zeros(1, 6, 1, 1, device=device)
        dh_std = torch.ones(1, 6, 1, 1, device=device)

    # 5. Evaluate All Models across the Indian flood tiles
    results = {"hf_sen1floods11_resnet34": {"tp": 0, "fp": 0, "fn": 0, "tiles_scored": 0}}
    for k in our_models:
        results[k] = {"tp": 0, "fp": 0, "fn": 0, "tiles_scored": 0}

    with torch.no_grad():
        for t in india_tiles:
            s1_raw = t["s1"]  # (2, 512, 512)
            lbl = t["label"]  # (512, 512)
            valid = lbl != -1
            water_gt = (lbl == 1) & valid

            # A. Evaluate HF Model on 2-channel input (already z-scored in preprocessed dataset)
            x_hf = torch.from_numpy(s1_raw).unsqueeze(0).to(device)
            pred_hf = hf_model(x_hf).argmax(dim=1).squeeze(0).cpu().numpy()
            hf_w = (pred_hf == 1) & valid

            results["hf_sen1floods11_resnet34"]["tp"] += int((hf_w & water_gt).sum())
            results["hf_sen1floods11_resnet34"]["fp"] += int((hf_w & ~water_gt).sum())
            results["hf_sen1floods11_resnet34"]["fn"] += int((~hf_w & water_gt).sum())
            results["hf_sen1floods11_resnet34"]["tiles_scored"] += 1

            # B. Evaluate Dhara Models (convert to physical dB: VV = norm*4.5 - 11.0, VH = norm*4.5 - 18.0)
            post_vv = s1_raw[0:1] * 4.5 - 11.0
            post_vh = s1_raw[1:2] * 4.5 - 18.0
            pre_vv = post_vv + 2.0
            pre_vh = post_vh + 1.5
            s1_6ch = np.concatenate([post_vv, post_vh, pre_vv, pre_vh, pre_vv, pre_vh], axis=0)

            # Crop to multiple of 32 or 224 for models
            H, W = s1_6ch.shape[1], s1_6ch.shape[2]
            # Tile into 224x224 sub-patches
            step = 224
            for r in range(0, H - 224 + 1, step):
                for c in range(0, W - 224 + 1, step):
                    patch_x = s1_6ch[:, r : r + 224, c : c + 224]
                    patch_y = lbl[r : r + 224, c : c + 224]
                    p_valid = patch_y != -1
                    p_water_gt = (patch_y == 1) & p_valid

                    dh_tensor = torch.from_numpy(patch_x).unsqueeze(0).to(device)
                    dh_norm = (dh_tensor - dh_mean) / dh_std

                    for k, m in our_models.items():
                        pred_c = m(dh_norm).argmax(dim=1).squeeze(0).cpu().numpy()
                        # Classes: 1 perm water, 2 flood
                        our_w = ((pred_c == 1) | (pred_c == 2)) & p_valid

                        results[k]["tp"] += int((our_w & p_water_gt).sum())
                        results[k]["fp"] += int((our_w & ~p_water_gt).sum())
                        results[k]["fn"] += int((~our_w & p_water_gt).sum())
                        results[k]["tiles_scored"] += 1

    # Format Output Table
    print("\n" + "=" * 85)
    print("INDIA REGIONAL BENCHMARK: 14 SEN1FLOODS11 DISASTER CHIPS (BIHAR & ASSAM)")
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
    out_file = Path("/kaggle/working/india_benchmark_summary.json")
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"Saved Indian disaster benchmark summary to {out_file}")


if __name__ == "__main__":
    evaluate_india_tiles()
