"""Convert Sen1Floods11 hand-labelled Sentinel-1 chips (train + valid splits) to training tiles in
the same format as prep_kurosiwo.py, so both listed training datasets are used together.

Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020; CC BY 4.0 per mirrors): one Sentinel-1 image
per 512 px chip at the flood date (VV, VH in dB), labels water / not water / no data. Differences
from Kuro Siwo, handled here and in train.py:
- no pre-event images: the post image fills all three dates ("post-only" input, as train.py's
  `no_pre` augmentation does for Kuro Siwo);
- no flood / permanent-water split: water pixels get label 3 ("water, kind unknown"); train.py
  scores them with -log(p_perm + p_flood), so they teach water detection without forcing a kind.
The test split and the Bolivia event are not converted: they stay held out for scoring.

Each chip gives four 224 px crops (corners). Output: <out>/s1f11_<split>_p00_{x,y,meta}.

Usage on the training machine: python prep_sen1floods11.py --out /tmp/ks_v4 [--uint8]
"""

import argparse
import csv
import io
import urllib.request
from pathlib import Path

import numpy as np
from prep_kurosiwo import quantize_db

BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLITS = {"train": "flood_train_data.csv", "valid": "flood_valid_data.csv"}
ACT_BASE = 9_000_000  # synthetic event ids: Sen1Floods11 country index + base; never Kuro Siwo ids
TILE, OFFS = 224, (0, 288)


def get(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def read_tif(data: bytes) -> np.ndarray:
    import rasterio

    with rasterio.MemoryFile(data) as mf, mf.open() as d:
        return d.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ks_v4")
    ap.add_argument("--uint8", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    countries: dict[str, int] = {}
    for split, name in SPLITS.items():
        xs, ys, metas = [], [], []
        lines = get(f"{BUCKET}/splits/flood_handlabeled/{name}").decode().splitlines()
        for line in lines:
            s1, lab = [v.strip() for v in line.split(",")[:2]]
            country = s1.split("_")[0]
            if country == "Bolivia":  # the held-out Sen1Floods11 event must never be trained on
                raise RuntimeError(f"Bolivia chip {s1} in the {split} split")
            act = ACT_BASE + countries.setdefault(country, len(countries))
            db = read_tif(get(f"{BUCKET}/data/flood_events/HandLabeled/S1Hand/{s1}")).astype(
                np.float32
            )
            y = read_tif(get(f"{BUCKET}/data/flood_events/HandLabeled/LabelHand/{lab}"))[0]
            for r0 in OFFS:
                for c0 in OFFS:
                    d = db[:, r0 : r0 + TILE, c0 : c0 + TILE]
                    yy = y[r0 : r0 + TILE, c0 : c0 + TILE]
                    lab3 = np.full(yy.shape, 255, np.uint8)
                    lab3[yy == 0] = 0
                    lab3[yy == 1] = 3  # water, flood or permanent unknown
                    lab3[~np.isfinite(d).all(0)] = 255
                    valid = float((lab3 != 255).mean())
                    if valid < 0.5:
                        continue
                    x = np.concatenate([d, d, d]).astype(np.float16)  # post repeated as pre1, pre2
                    xs.append(x)
                    ys.append(lab3)
                    metas.append(
                        {
                            "key": f"{s1}:{r0}:{c0}",
                            "actid": act,
                            "aoiid": country,
                            "grid_id": s1,
                            "flood_date": "",
                            "pflood": 0.0,
                            "pwater": float((lab3 == 3).mean()),
                            "valid_frac": valid,
                            "relief_m": float("nan"),
                            "slope_deg": float("nan"),
                            "source": "sen1floods11",
                        }
                    )
        tag = f"s1f11_{split}_p00"
        x = np.stack(xs)
        np.save(out / f"{tag}_x.npy", quantize_db(x) if args.uint8 else x)
        np.save(out / f"{tag}_y.npy", np.stack(ys))
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(metas[0]))
        w.writeheader()
        w.writerows(metas)
        (out / f"{tag}_meta.csv").write_text(buf.getvalue())
        print(
            f"{tag}: {len(xs)} tiles from {len(lines)} chips, countries {sorted(countries)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
