"""Download Kuro Siwo labelled GRD webdataset shards and convert them to compact arrays.

Runs on the training machine (Kaggle). Per shard it writes:
  <out>/<split>_<shard>_x.npy  float16 (n, 6, 224, 224): dB of [post VV, post VH, pre1 VV, pre1 VH, pre2 VV, pre2 VH]
  <out>/<split>_<shard>_y.npy  uint8   (n, 224, 224): 0 no water, 1 permanent water, 2 flood, 255 invalid
  <out>/<split>_<shard>_meta.csv: per-sample event id, AOI, flood fraction, DEM relief/slope stats
Each raw tar is deleted after conversion to save disk.

Allowed training data only (Kuro Siwo, CC BY; Bountos et al., NeurIPS 2024).
"""

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np

HF = "https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset/resolve/main"
SHARDS = {"train_GRD": 5, "test_GRD": 12}
BANDS = ["flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "sec2_vv", "sec2_vh"]
ALLOWED_DATASETS = {"kurosiwo"}


def to_db(a: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(a, 1e-5, None))


def dem_stats(dem: np.ndarray) -> tuple[float, float]:
    """Relief (p98 − p2, m) and mean slope (deg) of a 224² tile at 10 m."""
    d = dem[np.isfinite(dem)]
    if d.size < 100:
        return float("nan"), float("nan")
    relief = float(np.percentile(d, 98) - np.percentile(d, 2))
    gy, gx = np.gradient(np.nan_to_num(dem, nan=float(np.nanmean(dem))), 10.0)
    slope = float(np.degrees(np.arctan(np.hypot(gx, gy))).mean())
    return relief, slope


def stream_convert(url: str, out: Path, tag: str, max_samples: int = 1500) -> int:
    import urllib.request

    xs, ys, metas = [], [], []
    cur: dict[str, bytes] = {}
    cur_key = None

    def flush():
        if not cur or cur_key is None:
            return
        arrs = {k: np.load(io.BytesIO(v))[0] for k, v in cur.items() if k.endswith(".npy")}
        info = json.loads(cur["info.json"])
        x = np.stack([to_db(arrs[f"{b}.npy"]) for b in BANDS]).astype(np.float16)
        y = arrs["mask.npy"].astype(np.uint8)
        valid = (arrs["valid_mask.npy"] > 0) & (y <= 2)
        for b in BANDS:
            valid &= arrs[f"{b}.npy"] > 0
        y[~valid] = 255
        relief, slope = dem_stats(arrs["dem.npy"])
        xs.append(x)
        ys.append(y)
        metas.append(
            {
                "key": cur_key,
                "actid": info["actid"],
                "aoiid": info["aoiid"],
                "grid_id": info["grid_id"],
                "flood_date": info["flood_date"],
                "pflood": info["pflood"],
                "pwater": info["pwater"],
                "valid_frac": float(valid.mean()),
                "relief_m": relief,
                "slope_deg": slope,
            }
        )

    req = urllib.request.Request(url, headers={"User-Agent": "dhara/1.0"})
    with urllib.request.urlopen(req) as resp, tarfile.open(fileobj=resp, mode="r|*") as t:
        for m in t:
            if not m.isfile():
                continue
            key, field = m.name.split(".", 1)
            if key != cur_key:
                flush()
                if max_samples and len(xs) >= max_samples:
                    break
                cur, cur_key = {}, key
            f = t.extractfile(m)
            if f is not None:
                cur[field] = f.read()
        flush()

    np.save(out / f"{tag}_x.npy", np.stack(xs))
    np.save(out / f"{tag}_y.npy", np.stack(ys))
    with open(out / f"{tag}_meta.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metas[0]))
        w.writeheader()
        w.writerows(metas)
    return len(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ks_arr")
    ap.add_argument("--splits", nargs="+", default=["train_GRD"])
    ap.add_argument(
        "--max-shards", type=int, default=1, help="maximum shards per split to download"
    )
    ap.add_argument(
        "--max-samples", type=int, default=1500, help="max samples to extract per shard"
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [
        (split, i, f"{split}_{i:05d}")
        for split in args.splits
        for i in range(min(SHARDS[split], args.max_shards))
        if not (out / f"{split}_{i:05d}_meta.csv").exists()
    ]

    for split, i, tag in todo:
        url = f"{HF}/{split}/shard-{i:05d}.tar"
        print(
            f"Streaming {tag} directly from HuggingFace (up to {args.max_samples} samples)...",
            flush=True,
        )
        n = stream_convert(url, out, tag, max_samples=args.max_samples)
        print(f"{tag}: successfully extracted {n} samples directly into arrays.", flush=True)


if __name__ == "__main__":
    main()
