"""Score the flood segmentation model on its own against EMSR927 (Trishuli, August 2026): Himalayan
scenes it never saw in training. Validation only: nothing here is read by `src/dhara`.
Credit: "European Union, Copernicus Emergency Management Service data".

For each EMS area (01 Syapru Besi, 02 Timure, 03 Bidur) and each Sentinel-1 track with a full
same-track triplet in the local cache (pre2, pre, post: 12 days apart, σ⁰ VV+VH as the pipeline
feeds the model), the model predicts no water / water before / flood. Scored inside the EMS area
minus EMS "not analysed" areas, on the cache's 10 m UTM grid, against the EMS footprint
(observedEventA, a debris / mass-movement footprint photo-interpreted from very-high-resolution
images):

- flood: the model's flood class;
- any water: flood or water before (the footprint is mostly debris on the riverbed, which the
  model can call either).

A control triplet whose images all predate the flood (post 16 Aug on R085) shows how much of the
same footprint the model flags with no flood in the images.

Usage: uv run python validation/compare_ml_emsr927.py name=model_dir [...] > results.jsonl
"""

import json
import re
import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_emsr927 import ROOT, _layer, _scores, ems_aois  # noqa: E402

from dhara.members import ml_seg  # noqa: E402

CACHE = ROOT / "data" / "cache"
# (label, track, post, pre, pre2) — dates are the pass days in the cache file names.
TRIPLETS = [
    ("flood", "R085", "20260828", "20260816", "20260804"),
    ("flood", "R019", "20260905", "20260824", "20260812"),
    ("control (all images before the flood)", "R085", "20260816", "20260804", "20260723"),
]


def _passes(c: Path) -> dict[tuple[str, str], Path]:
    out = {}
    for f in c.glob("s1_*_SIGMA0_ELLIPSOID.tif"):
        m = re.match(r"s1_sentinel-1\w_(R\d+)_(\d{8})T", f.name)
        if m:
            out[(m.group(1), m.group(2))] = f
    return out


def _read(p: Path) -> tuple[np.ndarray, rasterio.Affine, object]:
    with rasterio.open(p) as ds:
        return ds.read([1, 2]).astype(np.float32), ds.transform, ds.crs


def main(models: dict[str, Path]) -> None:
    loaded = {k: ml_seg.load_model(d) for k, d in models.items()}
    aois = ems_aois()
    for n, (name, ext_) in aois.items():
        if n not in (1, 2, 3):
            continue
        ext = cast(BaseGeometry, ext_)
        ev = _layer(n, "observedEventA")
        na = _layer(n, "notAnalysedA")
        # The largest cache area that has each triplet over this EMS area.
        for label, track, post_d, pre_d, pre2_d in TRIPLETS:
            best = None
            for c in CACHE.iterdir():
                if not c.is_dir():
                    continue
                ps = _passes(c)
                keys = [(track, post_d), (track, pre_d), (track, pre2_d)]
                if not all(k in ps for k in keys):
                    continue
                with rasterio.open(ps[keys[0]]) as ds:
                    b = box(*transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))
                cover = b.intersection(ext).area / ext.area
                if cover > 0.5 and (best is None or cover > best[0]):
                    best = (cover, [ps[k] for k in keys])
            if best is None:
                continue
            (post, tf, crs), (pre, _, _), (pre2, _, _) = (_read(p) for p in best[1])
            shp = post.shape[1:]

            def mask(geoms, shp=shp, tf=tf) -> np.ndarray:
                geoms = [g for g in geoms if g is not None and not g.is_empty]
                if not geoms:
                    return np.zeros(shp, bool)
                return ~geometry_mask(geoms, shp, tf)

            region = mask(list(gpd.GeoSeries([ext], crs=4326).to_crs(crs)))
            region &= np.isfinite(post).all(0) & np.isfinite(pre).all(0)
            if na is not None and len(na):
                region &= ~mask(list(na.to_crs(crs).geometry))
            truth = mask(list(ev.to_crs(crs).geometry)) if ev is not None else np.zeros(shp, bool)
            px_km2 = abs(tf.a * tf.e) / 1e6
            for mname, model in loaded.items():
                if model is None:
                    continue
                r = ml_seg.predict(model, post, pre, pre2)
                p_none = 1.0 - r.p_flood - r.p_perm_water
                cls = np.argmax(np.stack([p_none, r.p_perm_water, r.p_flood]), 0)
                flood, water = cls == 2, cls >= 1
                print(
                    json.dumps(
                        {
                            "ems_aoi": f"{n:02d} {name}",
                            "triplet": label,
                            "track": track,
                            "dates": f"{pre2_d}/{pre_d} -> {post_d}",
                            "model": mname,
                            "region_km2": round(region.sum() * px_km2, 2),
                            "ems_footprint_km2": round((truth & region).sum() * px_km2, 3),
                            "flood_km2": round((flood & region).sum() * px_km2, 3),
                            "water_km2": round((water & region).sum() * px_km2, 3),
                            "flood": _scores(flood, truth, region),
                            "any_water": _scores(water, truth, region),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main({a.split("=", 1)[0]: Path(a.split("=", 1)[1]) for a in sys.argv[1:]})
