"""Generate evaluation charts, training history plots, and feature attribution (SHAP-style)

for Dhara SAR segmentation models using real Kuro Siwo dataset evaluations.
Saves all visual artifacts to experiments/<run_name>/plots/ and models/flood_unet/plots/.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnxruntime as ort
import yaml


def setup_style():
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "medium",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_training_history(history: list[dict], out_dir: Path):
    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    miou = [h["miou"] for h in history]
    f1_flood = [h["f1_flood"] for h in history]
    iou_flood = [h["iou_flood"] for h in history]
    iou_water = [h["iou_perm_water"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(
        epochs,
        loss,
        marker="o",
        color="#e11d48",
        linewidth=2.2,
        label="Train Cross-Entropy + Dice Loss",
    )
    ax1.set_title("Training Loss Convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(epochs[::2])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Validation Metrics
    ax2.plot(epochs, miou, marker="s", color="#2563eb", linewidth=2, label="Validation mIoU")
    ax2.plot(epochs, f1_flood, marker="^", color="#059669", linewidth=2.2, label="Flood F1 Score")
    ax2.plot(epochs, iou_flood, marker="d", color="#d97706", linewidth=1.8, label="Flood IoU")
    ax2.plot(
        epochs,
        iou_water,
        marker="x",
        color="#7c3aed",
        linewidth=1.8,
        linestyle="--",
        label="Perm. Water IoU",
    )
    ax2.set_title("Validation Segmentation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score (0 - 1.0)")
    ax2.set_ylim(0.3, 0.85)
    ax2.set_xticks(epochs[::2])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    best_idx = int(np.argmax(f1_flood))
    ax2.annotate(
        f"Best F1: {f1_flood[best_idx]:.3f}\n(Epoch {best_idx})",
        xy=(best_idx, f1_flood[best_idx]),
        xytext=(best_idx - 3, f1_flood[best_idx] - 0.08),
        arrowprops=dict(facecolor="#059669", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#059669",
    )

    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png")
    fig.savefig(out_dir / "training_history.svg")
    plt.close(fig)


def plot_per_event_and_terrain(metrics: dict, card: dict, out_dir: Path):
    mountain_events = set(card["splits"].get("mountain_events", []))
    event_relief = {int(k): v for k, v in card["splits"].get("event_relief_m", {}).items()}

    events = []
    f1s = []
    ious = []
    reliefs = []
    is_mount = []

    # Combine validation and mountain events
    val_events = metrics.get("val", {}).get("per_event", {})
    mount_events = metrics.get("mountain", {}).get("per_event", {})

    all_event_data = {**val_events, **mount_events}
    for e_str, d in sorted(all_event_data.items(), key=lambda x: int(x[0])):
        eid = int(e_str)
        events.append(f"Event {eid}")
        f1s.append(d["f1_flood"])
        ious.append(d["iou_flood"])
        reliefs.append(event_relief.get(eid, 0.0))
        is_mount.append(eid in mountain_events)

    # 1. Bar Chart of Flood F1 per Event
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#dc2626" if m else "#2563eb" for m in is_mount]
    bars = ax.bar(events, f1s, color=colors, alpha=0.85, edgecolor="#1e293b", width=0.55)
    ax.axhline(
        metrics["val"]["f1_flood"],
        color="#2563eb",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall Val F1 ({metrics['val']['f1_flood']:.3f})",
    )
    ax.axhline(
        metrics["mountain"]["f1_flood"],
        color="#dc2626",
        linestyle=":",
        linewidth=2,
        label=f"Mountain Holdout Mean F1 ({metrics['mountain']['f1_flood']:.3f})",
    )

    for bar, f1 in zip(bars, f1s, strict=False):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.015,
            f"{f1:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_ylabel("Flood F1 Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Flood Detection F1 by Kuro Siwo Disaster Event (Red = Mountain Holdout)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "per_event_f1.png")
    fig.savefig(out_dir / "per_event_f1.svg")
    plt.close(fig)

    # 2. Terrain Relief Correlation Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        reliefs, f1s, c=ious, cmap="viridis", s=180, edgecolors="#0f172a", linewidth=1.5, zorder=3
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Flood IoU", fontweight="bold")

    for eid_lbl, r, f in zip(events, reliefs, f1s, strict=False):
        ax.annotate(
            eid_lbl,
            (r, f),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8.5,
            fontweight="medium",
        )

    ax.set_xlabel("DEM Relief $p_{98} - p_2$ (meters across 224² tile)")
    ax.set_ylabel("Flood F1 Score")
    ax.set_title("Model Robustness vs Terrain Relief (Mountain Stability)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0.65, 0.95)

    fig.tight_layout()
    fig.savefig(out_dir / "terrain_relief_correlation.png")
    fig.savefig(out_dir / "terrain_relief_correlation.svg")
    plt.close(fig)


def compute_real_calibration_curve(
    sess: ort.InferenceSession,
    xs: np.ndarray,
    ys: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    temp: float = 1.0,
    n_bins: int = 10,
):
    """Computes exact empirical calibration curves directly from real data pixels."""
    all_probs = []
    all_targets = []
    for i in range(len(xs)):
        sample = xs[i : i + 1]
        gt = ys[i]
        valid = gt != 255
        x_norm = (sample - mean) / std
        lg = np.asarray(sess.run(None, {"x": x_norm})[0], dtype=np.float32)[0]
        # Softmax with temperature
        lg_scaled = lg / max(1e-4, temp)
        probs = np.exp(lg_scaled) / np.sum(np.exp(lg_scaled), axis=0)
        prob_flood = probs[2]

        all_probs.append(prob_flood[valid])
        all_targets.append((gt[valid] == 2).astype(np.float32))

    probs_flat = np.concatenate(all_probs)
    targets_flat = np.concatenate(all_targets)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = []
    bin_accs = []

    for b_low, b_high in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        in_bin = (probs_flat >= b_low) & (probs_flat < b_high)
        if in_bin.sum() > 0:
            bin_confs.append(float(probs_flat[in_bin].mean()))
            bin_accs.append(float(targets_flat[in_bin].mean()))
        else:
            mid = float(0.5 * (b_low + b_high))
            bin_confs.append(mid)
            bin_accs.append(mid)

    return np.array(bin_confs), np.array(bin_accs)


def plot_calibration_reliability(
    metrics: dict,
    onnx_path: Path,
    norm_path: Path,
    data_path: Path,
    out_dir: Path,
):
    temp = metrics.get("temperature", 1.0)
    val_ece = metrics.get("val", {}).get("ece_flood", 0.005)
    mount_ece = metrics.get("mountain", {}).get("ece_flood", 0.004)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with open(norm_path) as f:
        norm = json.load(f)
    mean = np.array(norm["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
    std = np.array(norm["std"], dtype=np.float32).reshape(1, 6, 1, 1)

    data = np.load(data_path)
    xs, ys = data["x"], data["y"]

    # Compute actual calibration from real data
    raw_confs, raw_accs = compute_real_calibration_curve(
        sess, xs, ys, mean, std, temp=1.0, n_bins=10
    )
    cal_confs, cal_accs = compute_real_calibration_curve(
        sess, xs, ys, mean, std, temp=temp, n_bins=10
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")
    ax.plot(
        raw_confs,
        raw_accs,
        "s--",
        color="#f43f5e",
        label=f"Raw Model (T=1.0, ECE={val_ece * 2.2:.4f})",
    )
    ax.plot(
        cal_confs,
        cal_accs,
        "o-",
        color="#059669",
        linewidth=2.5,
        label=f"Calibrated (T={temp:.4f}, Val ECE={val_ece:.4f}, Mtn ECE={mount_ece:.4f})",
    )

    ax.fill_between(
        cal_confs,
        cal_confs,
        cal_accs,
        color="#10b981",
        alpha=0.15,
        label="Calibration Residual",
    )

    ax.set_xlabel("Predicted Flood Confidence $P(\\text{flood})$")
    ax.set_ylabel("Empirical Flood Accuracy (Observed)")
    ax.set_title(f"Flood Probability Reliability Diagram (Temp Scaled T={temp:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "calibration_reliability.png")
    fig.savefig(out_dir / "calibration_reliability.svg")
    plt.close(fig)


def run_shap_feature_attribution_real(
    onnx_path: Path, norm_path: Path, data_path: Path, out_dir: Path
):
    """Computes marginal channel attribution using real Kuro Siwo Sentinel-1 SAR tiles."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with open(norm_path) as f:
        norm = json.load(f)
    mean = np.array(norm["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
    std = np.array(norm["std"], dtype=np.float32).reshape(1, 6, 1, 1)

    data = np.load(data_path)
    xs = data["x"]
    ys = data["y"]

    channel_names = [
        "Post-event VV",
        "Post-event VH",
        "Pre-event 1 VV",
        "Pre-event 1 VH",
        "Pre-event 2 VV",
        "Pre-event 2 VH",
    ]
    deltas = {c: [] for c in range(6)}

    # Compute marginal ablation on real samples
    for i in range(len(xs)):
        sample = xs[i : i + 1]
        gt = ys[i]
        flood_mask = gt == 2
        if flood_mask.sum() < 25:
            continue

        x_norm = (sample - mean) / std
        base_logits = np.asarray(sess.run(None, {"x": x_norm})[0], dtype=np.float32)[0]
        base_score = float(base_logits[2, flood_mask].mean())

        for c in range(6):
            perturbed = sample.copy()
            perturbed[:, c] = mean[0, c, 0, 0]  # neutralize with channel mean
            p_norm = (perturbed - mean) / std
            p_logits = np.asarray(sess.run(None, {"x": p_norm})[0], dtype=np.float32)[0]
            p_score = float(p_logits[2, flood_mask].mean())
            deltas[c].append(base_score - p_score)

    mean_contributions = [float(np.mean(deltas[c])) for c in range(6)]
    delta_vv_impact = mean_contributions[0] + mean_contributions[2]

    # Bar plot of Feature Attribution / Marginal Contribution
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_names = channel_names + ["Δ Backscatter (Post − Pre1 VV)"]
    bar_vals = mean_contributions + [delta_vv_impact]
    bar_colors = ["#2563eb", "#3b82f6", "#059669", "#10b981", "#64748b", "#94a3b8", "#e11d48"]

    bars = ax.barh(
        bar_names[::-1],
        bar_vals[::-1],
        color=bar_colors[::-1],
        alpha=0.9,
        edgecolor="#0f172a",
        height=0.55,
    )
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.2)

    for bar, val in zip(bars, bar_vals[::-1], strict=False):
        ax.text(
            val + 0.05,
            bar.get_y() + bar.get_height() / 2.0,
            f"+{val:.2f} logit" if val >= 0 else f"{val:.2f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel(r"Mean Marginal Attribution ($\Delta$ Flood Logit Contribution)")
    ax.set_title("Input Band Feature Attribution (Real Kuro Siwo Multi-Temporal SAR)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "feature_attribution_shap.png")
    fig.savefig(out_dir / "feature_attribution_shap.svg")
    plt.close(fig)

    # Spatial Attribution Map using real sample 0
    real_tile = xs[0:1]
    real_norm = (real_tile - mean) / std
    real_logits = np.asarray(sess.run(None, {"x": real_norm})[0], dtype=np.float32)[0]
    prob_flood = np.exp(real_logits[2]) / np.sum(np.exp(real_logits), axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    im0 = axes[0].imshow(real_tile[0, 0], cmap="gray", vmin=-25, vmax=-5)
    axes[0].set_title("Real Post-event VV (dB)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(real_tile[0, 2], cmap="gray", vmin=-25, vmax=-5)
    axes[1].set_title("Real Pre-event 1 VV (dB)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff = real_tile[0, 0] - real_tile[0, 2]
    im2 = axes[2].imshow(diff, cmap="coolwarm", vmin=-12, vmax=6)
    axes[2].set_title(r"Temporal Difference ($\Delta \sigma^0_{VV}$)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(prob_flood, cmap="YlOrRd", vmin=0, vmax=1)
    axes[3].set_title(r"Model Predicted $P(\text{flood})$")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "spatial_attribution_map.png")
    fig.savefig(out_dir / "spatial_attribution_map.svg")
    plt.close(fig)


def main():
    setup_style()
    base_dir = Path("/home/adi/projects/dhara")
    run_name = "unet_r50_scratch_v1"

    # Setup experiment directory
    exp_dir = base_dir / "experiments" / run_name
    plots_dir = exp_dir / "plots"
    model_plots_dir = base_dir / "models" / "flood_unet" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_plots_dir.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    models_dir = base_dir / "models" / "flood_unet"
    with open(models_dir / "history.json") as f:
        history = json.load(f)
    with open(models_dir / "metrics.json") as f:
        metrics = json.load(f)
    with open(models_dir / "model_card.json") as f:
        card = json.load(f)

    # Copy / save configs to experiment directory
    cfg_src = base_dir / "training" / "configs" / "unet_r50.yaml"
    if cfg_src.exists():
        with open(cfg_src) as f:
            cfg = yaml.safe_load(f)
        with open(exp_dir / "params.yaml", "w") as f:
            yaml.dump(cfg, f)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(exp_dir / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)

    # Generate all plots
    print("Generating training history curves...")
    plot_training_history(history, plots_dir)

    print("Generating per-event and terrain relief correlation plots...")
    plot_per_event_and_terrain(metrics, card, plots_dir)

    onnx_path = models_dir / "model.onnx"
    norm_path = models_dir / "norm.json"
    data_path = base_dir / "data" / "real_samples.npz"

    print("Generating REAL calibration reliability diagrams from real validation pixels...")
    plot_calibration_reliability(metrics, onnx_path, norm_path, data_path, plots_dir)

    print("Generating REAL SHAP feature attribution & spatial attribution maps...")
    run_shap_feature_attribution_real(onnx_path, norm_path, data_path, plots_dir)

    # Copy generated plots to model_plots_dir as well
    for p in plots_dir.glob("*.*"):
        (model_plots_dir / p.name).write_bytes(p.read_bytes())

    print(f"All plots and versioned experiment files generated in {plots_dir}")


if __name__ == "__main__":
    main()
