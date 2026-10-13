"""Multi-Model Evaluation & Diagnostic Reporting for SAR Flood Architectures.

Compares:
  - U-Net ResNet-50 (Scratch Baseline)
  - FPN ResNet-50 (High-Recall Challenger)
  - Physics-Guided SAR U-Net (Differential Specular & Polarimetric Residual Head)
  - DeepLabV3Plus ResNet-50 (ASPP Multi-Scale Dilated Convolutions, when available)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from generate_model_reports import (
    plot_calibration_reliability,
    plot_per_event_and_terrain,
    plot_training_history,
    run_shap_feature_attribution_real,
    setup_style,
)


def generate_experiment_diagnostics(exp_dir: Path, base_dir: Path):
    """Generate the full set of empirical diagnostic plots for a given experiment."""
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "history.json") as f:
        history = json.load(f)
    with open(exp_dir / "metrics.json") as f:
        metrics = json.load(f)
    with open(exp_dir / "model_card.json") as f:
        card = json.load(f)

    data_path = base_dir / "data" / "real_samples.npz"
    onnx_path = exp_dir / "model.onnx"
    norm_path = exp_dir / "norm.json"

    print(f"[{exp_dir.name}] Generating training history...")
    plot_training_history(history, plots_dir)

    print(f"[{exp_dir.name}] Generating per-event and terrain breakdown...")
    plot_per_event_and_terrain(metrics, card, plots_dir)

    if onnx_path.exists() and norm_path.exists() and data_path.exists():
        print(f"[{exp_dir.name}] Generating real calibration reliability curve...")
        plot_calibration_reliability(metrics, onnx_path, norm_path, data_path, plots_dir)

        print(f"[{exp_dir.name}] Generating empirical SHAP feature attribution...")
        run_shap_feature_attribution_real(onnx_path, norm_path, data_path, plots_dir)


def plot_multi_model_comparison(models_data: dict, out_dir: Path):
    """Plot multi-model comparative trajectories and operational bar charts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training History Trajectory (Flood F1 & mIoU)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    for _key, m in models_data.items():
        hist = m["hist"]
        epochs = [h["epoch"] for h in hist]
        f1s = [h["f1_flood"] for h in hist]
        mious = [h["miou"] for h in hist]

        ax1.plot(
            epochs,
            f1s,
            marker=m["marker"],
            color=m["color"],
            linewidth=2.2,
            label=f"{m['label']} (Final: {f1s[-1]:.3f})",
        )
        ax2.plot(
            epochs,
            mious,
            marker=m["marker"],
            color=m["color"],
            linewidth=2.2,
            label=f"{m['label']} (Final: {mious[-1]:.3f})",
        )

    ax1.set_title(
        "Validation Flood F1 Trajectory Across Architectures", fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Flood F1 Score", fontsize=11)
    ax1.set_ylim(0.3, 0.85)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)

    ax2.set_title(
        "Validation Mean IoU Trajectory Across Architectures", fontsize=12, fontweight="bold"
    )
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Mean IoU", fontsize=11)
    ax2.set_ylim(0.45, 0.85)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "architecture_comparison_history.png", dpi=200)
    fig.savefig(out_dir / "architecture_comparison_history.svg")
    plt.close(fig)

    # 2. Operational Metrics Comparison Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(13, 6))

    metric_keys = [
        ("precision_flood", "val", "Flood Precision"),
        ("recall_flood", "val", "Flood Recall"),
        ("f1_flood", "val", "Flood F1"),
        ("iou_flood", "val", "Flood IoU"),
        ("miou", "val", "Mean IoU"),
        ("recall_flood", "mountain", "Mountain Recall"),
        ("f1_flood", "mountain", "Mountain F1"),
        ("miou", "mountain", "Mountain mIoU"),
    ]

    metric_labels = [label for _, _, label in metric_keys]
    n_metrics = len(metric_keys)
    n_models = len(models_data)

    x = np.arange(n_metrics)
    total_width = 0.8
    bar_width = total_width / n_models

    for idx, (_m_id, m) in enumerate(models_data.items()):
        vals = []
        metrics = m["metrics"]
        for k, split, _ in metric_keys:
            val = metrics.get(split, {}).get(k, 0.0)
            vals.append(val)

        offset = (idx - (n_models - 1) / 2) * bar_width
        rects = ax.bar(
            x + offset,
            vals,
            bar_width,
            label=m["label"],
            color=m["color"],
            alpha=0.9,
            edgecolor="#0f172a",
            linewidth=0.8,
        )

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_title(
        "Operational & Mountain Generalization Benchmark Across SOTA Models",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "architecture_metrics_comparison.png", dpi=200)
    fig.savefig(out_dir / "architecture_metrics_comparison.svg")
    plt.close(fig)

    # 3. Mountain Holdout Generalization Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    model_labels = [m["label"] for m in models_data.values()]
    m_recalls = [m["metrics"]["mountain"]["recall_flood"] * 100 for m in models_data.values()]
    m_f1s = [m["metrics"]["mountain"]["f1_flood"] * 100 for m in models_data.values()]
    colors = [m["color"] for m in models_data.values()]

    x_idx = np.arange(len(model_labels))
    w = 0.35
    b1 = ax.bar(
        x_idx - w / 2,
        m_recalls,
        w,
        label="Mountain Flood Recall (%)",
        color=colors,
        alpha=0.85,
        edgecolor="#0f172a",
    )
    b2 = ax.bar(
        x_idx + w / 2,
        m_f1s,
        w,
        label="Mountain Flood F1 (%)",
        color=colors,
        alpha=0.45,
        hatch="//",
        edgecolor="#0f172a",
    )

    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_ylabel("Performance (%)", fontsize=11)
    ax.set_title(
        "Steep Mountain Relief Holdout: Flood Recall & F1 Comparison",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x_idx)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylim(70, 95)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "mountain_robustness_comparison.png", dpi=200)
    fig.savefig(out_dir / "mountain_robustness_comparison.svg")
    plt.close(fig)


def main():
    setup_style()
    base_dir = Path("/home/adi/projects/dhara")
    exp_dir = base_dir / "experiments"
    comp_dir = exp_dir / "comparison_plots"
    comp_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "id": "unet_r50_scratch_v1",
            "label": "U-Net ResNet-50",
            "color": "#2563eb",  # Blue
            "marker": "o",
        },
        {
            "id": "fpn_r50_v1",
            "label": "FPN ResNet-50",
            "color": "#dc2626",  # Red
            "marker": "s",
        },
        {
            "id": "physics_unet_r50_v1",
            "label": "Physics-Guided SAR U-Net",
            "color": "#16a34a",  # Emerald Green
            "marker": "^",
        },
        {
            "id": "deeplabv3p_r50_v1",
            "label": "DeepLabV3+ ResNet-50",
            "color": "#9333ea",  # Purple
            "marker": "D",
        },
        {
            "id": "segformer_mitb0_v1",
            "label": "SegFormer mit_b0 (ViT)",
            "color": "#0284c7",  # Sky Blue
            "marker": "p",
        },
        {
            "id": "physics_segformer_mitb0_v1",
            "label": "Physics-SegFormer mit_b0",
            "color": "#ea580c",  # Deep Orange
            "marker": "*",
        },
        {
            "id": "unetplusplus_r50_v1",
            "label": "UNet++ ResNet-50",
            "color": "#059669",  # Teal / Deep Green
            "marker": "h",
        },
        {
            "id": "pan_r50_v1",
            "label": "PAN ResNet-50",
            "color": "#d97706",  # Amber / Gold
            "marker": "v",
        },
    ]

    models_data = {}
    for exp_meta in experiments:
        edir = exp_dir / exp_meta["id"]
        if edir.exists() and (edir / "metrics.json").exists() and (edir / "history.json").exists():
            print(f"Loading experiment {exp_meta['id']}...")
            # Generate standalone plots if not already present
            if not (edir / "plots" / "calibration_reliability.png").exists():
                try:
                    generate_experiment_diagnostics(edir, base_dir)
                except Exception as e:
                    print(f"Warning: could not generate standalone plots for {exp_meta['id']}: {e}")

            with open(edir / "history.json") as f:
                hist = json.load(f)
            with open(edir / "metrics.json") as f:
                metrics = json.load(f)
            with open(edir / "model_card.json") as f:
                card = json.load(f)

            models_data[exp_meta["id"]] = {
                "label": exp_meta["label"],
                "color": exp_meta["color"],
                "marker": exp_meta["marker"],
                "hist": hist,
                "metrics": metrics,
                "card": card,
            }

    print(f"Loaded {len(models_data)} models for comparison: {list(models_data.keys())}")
    if len(models_data) >= 2:
        plot_multi_model_comparison(models_data, comp_dir)
        print("Successfully generated all comparison plots in", comp_dir)

    # South Asia Regional Benchmark Plot
    sa_summary_file = comp_dir / "south_asia_benchmark_summary.json"
    if sa_summary_file.exists():
        plot_south_asia_benchmark(sa_summary_file, comp_dir)


def plot_south_asia_benchmark(json_path: Path, out_dir: Path):
    """Plot multi-region empirical evaluation across South Asian floodplains."""
    with open(json_path) as f:
        data = json.load(f)

    scopes = ["Overall_South_Asia", "India", "Pakistan", "Sri-Lanka"]
    scope_titles = [
        "Overall South Asia (29 Chips)",
        "India: Bihar & Assam (14 Chips)",
        "Pakistan: Indus Basin (6 Chips)",
        "Sri Lanka: Tropical (9 Chips)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    model_display_names = {
        "hf_sen1floods11_resnet34": "Public HF ResNet-34",
        "unet_r50_scratch_v1": "Dhara U-Net R50",
        "fpn_r50_v1": "Dhara FPN R50",
        "physics_unet_r50_v1": "Dhara Physics U-Net",
        "deeplabv3p_r50_v1": "Dhara DeepLabV3+ R50",
        "segformer_mitb0_v1": "Dhara SegFormer ViT",
        "physics_segformer_mitb0_v1": "Dhara Phys-SegFormer",
        "unetplusplus_r50_v1": "Dhara UNet++ R50",
        "manet_r50_v1": "Dhara MAnet R50",
        "pan_r50_v1": "Dhara PAN R50",
    }

    for idx, scope in enumerate(scopes):
        ax = axes[idx]
        res = data.get(scope, {})
        labels = [model_display_names.get(k, k) for k in res]
        recalls = [v["water_rec"] * 100 for v in res.values()]
        precisions = [v["water_prec"] * 100 for v in res.values()]
        f1s = [v["water_f1"] * 100 for v in res.values()]

        x = np.arange(len(labels))
        w = 0.26

        ax.bar(
            x - w,
            recalls,
            w,
            label="Water Recall (%)",
            color="#3b82f6",
            alpha=0.9,
            edgecolor="#1e293b",
        )
        ax.bar(
            x,
            precisions,
            w,
            label="Water Precision (%)",
            color="#10b981",
            alpha=0.9,
            edgecolor="#1e293b",
        )
        r3 = ax.bar(
            x + w, f1s, w, label="Water F1 (%)", color="#f59e0b", alpha=0.9, edgecolor="#1e293b"
        )

        ax.set_title(scope_titles[idx], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=9)

        # Annotate F1 scores
        for rect in r3:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )

    fig.suptitle(
        "South Asian Regional Flood Generalization Benchmark (Sen1Floods11 Ground Truth)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    fig.savefig(out_dir / "south_asia_regional_comparison.png", dpi=200)
    fig.savefig(out_dir / "south_asia_regional_comparison.svg")
    plt.close(fig)
    print("Saved South Asian regional comparison plot.")


if __name__ == "__main__":
    main()
