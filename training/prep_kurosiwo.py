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


# uint8 storage of dB: q = round((dB - DB_MIN) * DB_STEPS), 0.25 dB steps over [-45, +18.75] dB,
# far finer than speckle. Halves disk use vs float16 (Kaggle counts file cache as memory).
DB_MIN, DB_STEPS = -45.0, 4.0


def quantize_db(x: np.ndarray) -> np.ndarray:
    q = np.round((np.nan_to_num(x, nan=DB_MIN) - DB_MIN) * DB_STEPS)
    return np.clip(q, 0, 255).astype(np.uint8)


def dir_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("*.npy")) / 1e9


def dem_stats(dem: np.ndarray) -> tuple[float, float]:
    """Relief (p98 − p2, m) and mean slope (deg) of a 224² tile at 10 m."""
    d = dem[np.isfinite(dem)]
    if d.size < 100:
        return float("nan"), float("nan")
    relief = float(np.percentile(d, 98) - np.percentile(d, 2))
    gy, gx = np.gradient(np.nan_to_num(dem, nan=float(np.nanmean(dem))), 10.0)
    slope = float(np.degrees(np.arctan(np.hypot(gx, gy))).mean())
    return relief, slope


def stream_convert(
    url: str,
    out: Path,
    tag: str,
    max_samples: int = 0,
    per_event_cap: int = 200,
    chunk: int = 1500,
    water_only: bool = False,
    eval_events: frozenset[int] = frozenset(),
    eval_cap: int = 200,
    as_uint8: bool = False,
    max_gb: float = 0.0,
) -> int:
    """Stream one shard and keep at most `per_event_cap` valid tiles per event (0 = no cap), so
    every event in the shard is represented. Writes <tag>_pNN_{x,y,meta} every `chunk` tiles."""
    import urllib.request

    xs, ys, metas = [], [], []
    cur: dict[str, bytes] = {}
    cur_key = None
    per_event: dict[int, int] = {}
    state = {"part": 0, "total": 0, "stopped": False}

    def write_part():
        if not xs:
            return
        name = f"{tag}_p{state['part']:02d}"
        if max_gb and dir_gb(out) > max_gb:
            state["stopped"] = True
            xs.clear(), ys.clear(), metas.clear()
            return
        x = np.stack(xs)
        np.save(out / f"{name}_x.npy", quantize_db(x) if as_uint8 else x)
        np.save(out / f"{name}_y.npy", np.stack(ys))
        with open(out / f"{name}_meta.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(metas[0]))
            w.writeheader()
            w.writerows(metas)
        state["part"] += 1
        state["total"] += len(xs)
        xs.clear(), ys.clear(), metas.clear()

    def flush():
        if not cur or cur_key is None or "info.json" not in cur:
            return
        info = json.loads(cur["info.json"])
        act = int(info["actid"])
        is_eval = act in eval_events
        cap = eval_cap if is_eval else per_event_cap
        if cap and per_event.get(act, 0) >= cap:
            return
        arrs = {k: np.load(io.BytesIO(v))[0] for k, v in cur.items() if k.endswith(".npy")}
        x = np.stack([to_db(arrs[f"{b}.npy"]) for b in BANDS]).astype(np.float16)
        y = arrs["mask.npy"].astype(np.uint8)
        valid = (arrs["valid_mask.npy"] > 0) & (y <= 2)
        for b in BANDS:
            valid &= arrs[f"{b}.npy"] > 0
        y[~valid] = 255
        if valid.mean() < 0.5:  # training ignores these tiles anyway
            return
        if water_only and not is_eval and not np.isin(y, (1, 2)).any():
            return  # BlackBench protocol: training skips tiles without any water
        per_event[act] = per_event.get(act, 0) + 1
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
        if len(xs) >= chunk:
            write_part()

    req = urllib.request.Request(url, headers={"User-Agent": "dhara/1.0"})
    with urllib.request.urlopen(req) as resp, tarfile.open(fileobj=resp, mode="r|*") as t:
        for m in t:
            if not m.isfile():
                continue
            key, field = m.name.split(".", 1)
            if key != cur_key:
                flush()
                if max_samples and state["total"] + len(xs) >= max_samples:
                    break
                if state["stopped"]:
                    break
                cur, cur_key = {}, key
            f = t.extractfile(m)
            if f is not None:
                cur[field] = f.read()
        flush()
    write_part()
    done = {str(k): v for k, v in sorted(per_event.items())}
    if state["stopped"]:
        done["stopped_at_gb"] = max_gb
    (out / f"{tag}.done").write_text(json.dumps(done))
    return state["total"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ks_arr")
    ap.add_argument("--splits", nargs="+", default=["train_GRD"])
    ap.add_argument("--max-shards", type=int, default=99, help="maximum shards per split")
    ap.add_argument("--max-samples", type=int, default=0, help="max tiles per shard (0 = all)")
    ap.add_argument("--per-event-cap", type=int, default=200, help="max tiles per event per shard")
    ap.add_argument("--shard", type=str, default=None, help="only this shard, e.g. train_GRD:3")
    ap.add_argument(
        "--water-only", action="store_true", help="training events: tiles with water only"
    )
    ap.add_argument("--eval-events", default="", help="comma list: all tiles, capped at --eval-cap")
    ap.add_argument("--eval-cap", type=int, default=200)
    ap.add_argument("--uint8", action="store_true", help="store dB quantised to uint8")
    ap.add_argument("--max-gb", type=float, default=0.0, help="stop writing above this total size")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [
        (split, i, f"{split}_{i:05d}")
        for split in args.splits
        for i in range(min(SHARDS[split], args.max_shards))
        if not (out / f"{split}_{i:05d}.done").exists()
    ]
    if args.shard:
        sp, idx = args.shard.split(":")
        todo = [t for t in todo if t[0] == sp and t[1] == int(idx)]

    for split, i, tag in todo:
        url = f"{HF}/{split}/shard-{i:05d}.tar"
        print(
            f"Streaming {tag} directly from HuggingFace (up to {args.max_samples} samples)...",
            flush=True,
        )
        n = stream_convert(
            url,
            out,
            tag,
            max_samples=args.max_samples,
            per_event_cap=args.per_event_cap,
            water_only=args.water_only,
            eval_events=frozenset(int(e) for e in args.eval_events.split(",") if e),
            eval_cap=args.eval_cap,
            as_uint8=args.uint8,
            max_gb=args.max_gb,
        )
        print(f"{tag}: successfully extracted {n} samples directly into arrays.", flush=True)


if __name__ == "__main__":
    main()
