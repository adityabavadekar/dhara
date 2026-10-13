"""Score a Dhara segmentation model on Sen1Floods11 hand-labelled Sentinel-1 chips (test split and
the held-out Bolivia event), next to a per-chip Otsu threshold on VV as a simple baseline.

Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020; CC BY 4.0 per mirrors) gives one Sentinel-1
image per chip, at the flood date, with labels water (1), not water (0) and no data (−1). It has
no pre-event images, so the model runs in "post only" mode (both pre-event inputs = the post
image) and "water" means permanent water or flood. The Kuro Siwo scores remain the change scores.

Usage: uv run python training/eval_sen1floods11.py [--model models/flood_unet] [--out <json>]
Downloads ~80 MB of chips into --cache on first use (public bucket, HTTPS).
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from skimage.filters import threshold_otsu

from dhara.members import ml_seg

BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLITS = {"test": "flood_test_data.csv", "bolivia": "flood_bolivia_data.csv"}


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    return path


def chips(split: str, cache: Path) -> list[tuple[Path, Path]]:
    csv = fetch(f"{BUCKET}/splits/flood_handlabeled/{SPLITS[split]}", cache / SPLITS[split])
    out = []
    for line in csv.read_text().splitlines():
        s1, lab = [x.strip() for x in line.split(",")[:2]]
        out.append(
            (
                fetch(f"{BUCKET}/data/flood_events/HandLabeled/S1Hand/{s1}", cache / "S1Hand" / s1),
                fetch(
                    f"{BUCKET}/data/flood_events/HandLabeled/LabelHand/{lab}",
                    cache / "LabelHand" / lab,
                ),
            )
        )
    return out


class Counts:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0
        self.ious: list[float] = []

    def add(self, pred: np.ndarray, y: np.ndarray) -> None:
        ok = y >= 0
        p, t = pred[ok], y[ok] == 1
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
        if tp + fp + fn:
            self.ious.append(tp / (tp + fp + fn))

    def scores(self, seed: int = 0) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        rng = np.random.default_rng(seed)
        v = np.array(self.ious)
        boot = [rng.choice(v, len(v)).mean() for _ in range(2000)] if len(v) else [np.nan]
        return {
            "iou_water": round(tp / max(tp + fp + fn, 1), 4),
            "f1_water": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "chip_mean_iou": round(float(v.mean()), 4) if len(v) else None,
            "chip_mean_iou_ci95": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
            "chips": len(v),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/flood_unet")
    ap.add_argument("--cache", default="/tmp/sen1floods11")
    ap.add_argument("--out", default=None, help="default: <model>/metrics_sen1floods11.json")
    args = ap.parse_args()
    model = ml_seg.load_model(Path(args.model))
    if model is None:
        raise SystemExit(f"no model at {args.model}")
    report: dict = {"model": model.card.get("run_name"), "mode": "post only", "splits": {}}
    for split in SPLITS:
        ours, otsu = Counts(), Counts()
        for s1, lab in chips(split, Path(args.cache)):
            with rasterio.open(s1) as d:
                db = d.read().astype(np.float32)  # VV, VH in dB
            with rasterio.open(lab) as d:
                y = d.read(1).astype(np.int16)
            lin = np.where(np.isfinite(db), 10 ** (db / 10), np.nan).astype(np.float32)
            r = ml_seg.predict(model, lin, lin, lin)
            water = np.nan_to_num(r.p_flood) + np.nan_to_num(r.p_perm_water) >= 0.5
            ours.add(water, y)
            vv = db[0][np.isfinite(db[0])]
            t = threshold_otsu(vv) if vv.size > 100 else -15.0
            otsu.add(np.nan_to_num(db[0], nan=0.0) < t, y)
        report["splits"][split] = {"model": ours.scores(), "otsu_vv_baseline": otsu.scores()}
        print(split, json.dumps(report["splits"][split]), flush=True)
    out = Path(args.out) if args.out else Path(args.model) / "metrics_sen1floods11.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
