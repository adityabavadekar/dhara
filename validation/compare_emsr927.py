"""Compare Dhara runs with the official Copernicus EMS Rapid Mapping products for EMSR927
(Flood in Nepal, August 2026). Validation only: nothing here is read by `src/dhara`.
Credit: "European Union, Copernicus Emergency Management Service data".

EMSR927 maps the event as landslide / mass-movement footprints (observedEventA, photo-interpreted
from very-high-resolution imagery) plus graded building damage (builtUpP). For each EMS area of
interest that overlaps a job, inside (EMS AOI ∩ job AOI) minus the EMS "not analysed" areas:

- pixel scores on the job's 10 m grid: our flood ∪ debris vs the EMS footprint (precision,
  recall, F1, IoU), strict and with a one-pixel (10 m) tolerance on both sides;
- building recall: EMS buildings graded Destroyed or Damaged lying on or within 20 m of a mapped
  pixel; and how many of them sit on pixels we could not observe.

Usage: uv run python validation/compare_emsr927.py <job_id> [...]   (DHARA_JOBS = other job folder)
"""

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import binary_dilation
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
EMS = ROOT / "validation" / "data" / "emsr927"
FLOOD, DEBRIS, NOT_OBS, OUTSIDE = 3, 4, 0, 255  # dhara.fusion.fuse class codes


def _layer(aoi: int, kind: str) -> gpd.GeoDataFrame | None:
    """Latest delivered version of one EMS layer for an AOI (monitoring supersedes the first map)."""
    hits = sorted(EMS.glob(f"EMSR927_AOI{aoi:02d}_GRA_*_{kind}_v*.json"))
    mon = [h for h in hits if "MONIT" in h.name]
    path = (mon or hits)[-1] if hits else None
    return gpd.read_file(path) if path else None


def ems_aois() -> dict[int, tuple[str, object]]:
    r = json.loads((EMS / "activation.json").read_text())["results"][0]
    from shapely import wkt

    return {
        a["number"]: (a["name"], wkt.loads(a["extent"].split(";")[-1]))
        for a in r["aois"]
        if any(p.get("version", {}).get("statusCode") == "F" for p in a["products"])
    }


def _scores(ours: np.ndarray, truth: np.ndarray, region: np.ndarray) -> dict:
    o, t = ours & region, truth & region
    tp, fp, fn = int((o & t).sum()), int((o & ~t).sum()), int((~o & t).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(2 * p * r / max(p + r, 1e-9), 3),
        "iou": round(tp / max(tp + fp + fn, 1), 3),
    }


def compare(job_id: str, jobs: Path | None = None) -> list[dict]:
    d = (jobs or Path(os.environ.get("DHARA_JOBS", ROOT / "data" / "jobs"))) / job_id
    if not (d / "classes.tif").exists() or not (d / "provenance.json").exists():
        return []  # failed or unfinished job
    prov = json.loads((d / "provenance.json").read_text())
    with rasterio.open(d / "classes.tif") as ds:
        cls, tf, crs = ds.read(1), ds.transform, ds.crs
    job_aoi = gpd.GeoSeries([shape(prov["inputs"]["aoi"]["geojson"])], crs=4326)
    shp = cls.shape

    def mask(geoms) -> np.ndarray:
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros(shp, bool)
        return ~geometry_mask(geoms, shp, tf, all_touched=False)

    ours = (cls == FLOOD) | (cls == DEBRIS)
    unobs = cls == NOT_OBS
    out = []
    for n, (name, ext) in ems_aois().items():
        ems_aoi = gpd.GeoSeries([ext], crs=4326)
        inter = job_aoi.intersection(ems_aoi.iloc[0])
        if inter.is_empty.iloc[0] or float(inter.to_crs(crs).area.iloc[0]) < 1e6:
            continue
        region = mask(list(inter.to_crs(crs))) & (cls != OUTSIDE)
        na = _layer(n, "notAnalysedA")
        if na is not None and len(na):
            region &= ~mask(list(na.to_crs(crs).geometry))
        ev = _layer(n, "observedEventA")
        truth = mask(list(ev.to_crs(crs).geometry)) if ev is not None else np.zeros(shp, bool)
        px_km2 = abs(tf.a * tf.e) / 1e6
        row = {
            "job": job_id,
            "event": prov["inputs"]["event"],
            "ems_aoi": f"{n:02d} {name}",
            "region_km2": round(region.sum() * px_km2, 2),
            "ems_footprint_km2": round((truth & region).sum() * px_km2, 3),
            "ours_km2": round((ours & region).sum() * px_km2, 3),
            "ems_footprint_unobservable_pct": round(
                100 * (truth & region & unobs).sum() / max((truth & region).sum(), 1), 1
            ),
            "strict": _scores(ours, truth, region),
            "tol_10m": {
                # one-pixel tolerance: precision against dilated truth, recall of dilated ours
                "precision": _scores(ours, binary_dilation(truth), region)["precision"],
                "recall": _scores(binary_dilation(ours), truth, region)["recall"],
            },
        }
        b = _layer(n, "builtUpP")
        if b is not None and len(b):
            b = b[b["damage_gra"].isin(["Destroyed", "Damaged"])].to_crs(crs)
            c = ((b.geometry.x - tf.c) / tf.a).astype(int).to_numpy()
            r = ((b.geometry.y - tf.f) / tf.e).astype(int).to_numpy()
            ok = (r >= 0) & (r < shp[0]) & (c >= 0) & (c < shp[1])
            r, c = r[ok], c[ok]
            ok2 = region[r, c]
            r, c = r[ok2], c[ok2]
            near = binary_dilation(ours, iterations=2)
            row["ems_damaged_buildings"] = len(r)
            row["building_recall_20m"] = round(float(near[r, c].mean()), 3) if len(r) else None
            row["buildings_on_unobservable"] = int(unobs[r, c].sum())
        out.append(row)
    return out


if __name__ == "__main__":
    for j in sys.argv[1:]:
        for row in compare(j):
            print(json.dumps(row))
