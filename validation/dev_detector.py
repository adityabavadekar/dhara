"""Development check of the footprint detector (src/dhara/fusion/detector.py) against EMSR927.
Validation only: nothing here is read by `src/dhara`.
Credit: "European Union, Copernicus Emergency Management Service data".

EMS areas 01 Syapru Besi, 02 Timure and 03 Bidur are development areas: detector settings may be
chosen by looking at them. Area 05 (Phosretar) is held out and is scored once, after the settings
are frozen (`--holdout`). Every flood run is paired with a control run whose images all predate
the flood, so a rule that only paints the valley shows up as a high control score.

For each saved job: re-fuse the saved member maps exactly as the pipeline does, apply the detector
with each setting, score flood ∪ debris against the EMS footprint inside (EMS area ∩ job area)
minus EMS "not analysed" areas, on the job's 10 m grid.

Usage: uv run python validation/dev_detector.py label=job_id [...]
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import binary_dilation
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_emsr927 import ROOT, _layer, _scores, ems_aois  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

from dhara.config import get_settings  # noqa: E402
from dhara.fusion import fuse as F  # noqa: E402
from dhara.fusion.detector import DetectorParams, refine  # noqa: E402
from dhara.geo import AOI  # noqa: E402
from dhara.jobs import pipeline as P  # noqa: E402
from dhara.terrain import hydro  # noqa: E402

DEV_AREAS = (1, 2, 3)
HOLDOUT = (5,)


def _read(p: Path) -> np.ndarray:
    with rasterio.open(p) as ds:
        return ds.read(1)


def load(job_id: str) -> dict:
    d = ROOT / "data" / "jobs" / job_id
    prov = json.loads((d / "provenance.json").read_text())
    aoi = AOI(geojson=prov["inputs"]["aoi"]["geojson"])
    grid = P.make_grid(aoi)
    cache = get_settings().cache_dir / aoi.hash
    terr, _ = P.cached_terrain(cache, grid)
    hand = hydro.resample_to(terr.hand, terr.grid, grid)
    slope = hydro.resample_to(terr.slope_deg, terr.grid, grid)
    acc_km2 = terr.acc * (terr.grid.res**2) / 1e6
    major = hydro.resample_to(
        (acc_km2 >= P.MAJOR_CHANNEL_KM2).astype(np.float32), terr.grid, grid, Resampling.max
    )
    major = binary_dilation(np.nan_to_num(major) > 0, iterations=3)
    with rasterio.open(d / "classes.tif") as ds:
        classes, tf, crs = ds.read(1), ds.transform, ds.crs
    members = prov["stages"]["fusion"]["members"]
    ev = [
        F.Evidence(
            m["name"], m["target"], _read(d / f"member_{m['name']}.tif"), m["weight"], m["sensor"]
        )
        for m in members
        if (d / f"member_{m['name']}.tif").exists()
    ]
    in_aoi = classes != F.OUTSIDE
    # The pipeline saves the raw pre-event water mask; older jobs only have it inside classes.tif.
    perm = (
        _read(d / "perm_water.tif") > 0
        if (d / "perm_water.tif").exists()
        else classes == F.PERMANENT_WATER
    )
    return {
        "id": job_id,
        "event": prov["inputs"]["event"],
        "aoi": shape(prov["inputs"]["aoi"]["geojson"]),
        "tf": tf,
        "crs": crs,
        "shape": classes.shape,
        "fr": F.fuse(ev, hand, slope, in_aoi, perm, major_channels=major),
        "hand": hand,
        "slope": slope,
        "major": major,
        "members": [e.name for e in ev],
    }


def regions(job: dict, areas) -> list[tuple[str, np.ndarray, np.ndarray]]:
    shp, tf, crs = job["shape"], job["tf"], job["crs"]

    def mask(geoms) -> np.ndarray:
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        return ~geometry_mask(geoms, shp, tf) if geoms else np.zeros(shp, bool)

    out = []
    for n, (name, ext) in ems_aois().items():
        if n not in areas:
            continue
        inter = gpd.GeoSeries([job["aoi"]], crs=4326).intersection(ext)
        if inter.is_empty.iloc[0] or float(inter.to_crs(crs).area.iloc[0]) < 1e6:
            continue
        region = mask(list(inter.to_crs(crs)))
        na = _layer(n, "notAnalysedA")
        if na is not None and len(na):
            region &= ~mask(list(na.to_crs(crs).geometry))
        ev = _layer(n, "observedEventA")
        truth = mask(list(ev.to_crs(crs).geometry)) if ev is not None else np.zeros(shp, bool)
        out.append((f"{n:02d} {name}", region, truth))
    return out


def score(job: dict, prm: DetectorParams | None, areas) -> list[dict]:
    fr = job["fr"]
    if prm is None:
        cls = fr.classes
    else:
        cls, _, _ = refine(
            fr.classes,
            fr.confidence,
            fr.p_flood,
            fr.p_debris,
            job["hand"],
            job["slope"],
            job["major"],
            prm,
        )
    ours = (cls == F.FLOOD) | (cls == F.DEBRIS)
    px_km2 = abs(job["tf"].a * job["tf"].e) / 1e6
    rows = []
    for name, region, truth in regions(job, areas):
        region = region & (cls != F.OUTSIDE)
        rows.append(
            {
                "area": name,
                "footprint_km2": round(float((truth & region).sum() * px_km2), 3),
                "mapped_km2": round(float((ours & region).sum() * px_km2), 3),
                "perm_water_in_footprint_pct": round(
                    100
                    * float((truth & region & (cls == F.PERMANENT_WATER)).sum())
                    / max(int((truth & region).sum()), 1),
                    1,
                ),
                **_scores(ours, truth, region),
            }
        )
    return rows


VARIANTS: dict[str, DetectorParams | None] = {
    "current detector": DetectorParams(),
    "local mark 150 m": DetectorParams(local_mark_px=15),
    "local mark 250 m": DetectorParams(local_mark_px=25),
    "local mark 150 m, dist 400": DetectorParams(local_mark_px=15, fill_max_dist_m=400),
}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    areas = HOLDOUT if "--holdout" in sys.argv else DEV_AREAS
    jobs = {a.split("=", 1)[0]: load(a.split("=", 1)[1]) for a in args}
    for vname, prm in VARIANTS.items():
        print(f"\n## {vname}")
        for label, job in jobs.items():
            for r in score(job, prm, areas):
                print(
                    f"{label:9} {r['area']:15} footprint {r['footprint_km2']:6.2f}  mapped {r['mapped_km2']:6.2f}  "
                    f"P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f}  perm-water-in-fp {r['perm_water_in_footprint_pct']}%"
                )
