"""Check Dhara runs against the HDX / Microsoft AI for Good reference (UNOSAT damage area x Overture
buildings, `damaged = 1`). Validation only: nothing here is read by `src/dhara`.

For each job, inside the job's AOI:
- recall: share of reference-damaged buildings within `tol_m` of mapped flood or debris;
- precision: share of our "hit" buildings within `tol_m` of a reference-damaged building.
The two building sets differ (OSM vs Overture), so matching is by distance, not by id.

Usage: uv run python validation/compare_buildings.py <job_id> [<job_id> ...]
(set DHARA_JOBS to read jobs from another folder)
"""

import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "validation" / "data" / "ref_damaged_buildings.parquet"
FLOOD, DEBRIS = 3, 4  # dhara.fusion.fuse class codes


def compare(job_id: str, tol_m: float = 50.0, jobs: Path | None = None) -> dict:
    d = (jobs or Path(os.environ.get("DHARA_JOBS", ROOT / "data" / "jobs"))) / job_id
    prov = json.loads((d / "provenance.json").read_text())
    aoi = gpd.GeoSeries([shape(prov["inputs"]["aoi"]["geojson"])], crs=4326)
    with rasterio.open(d / "classes.tif") as ds:
        cls = ds.read(1)
        crs, tf = ds.crs, ds.transform
    aoi_u = aoi.to_crs(crs)
    ref = gpd.read_parquet(REF).to_crs(crs)
    ref = ref[ref.intersects(aoi_u.iloc[0])]
    rr, cc = np.nonzero(np.isin(cls, (FLOOD, DEBRIS)))
    hz = gpd.GeoSeries(
        gpd.points_from_xy(tf.c + (cc + 0.5) * tf.a, tf.f + (rr + 0.5) * tf.e), crs=crs
    )
    ours = gpd.read_parquet(d / "buildings.parquet").to_crs(crs)
    hit = ours[ours["status"] == "hit"]
    out = {
        "job": job_id,
        "event": prov["inputs"]["event"],
        "aoi_km2": round(float(aoi_u.area.iloc[0]) / 1e6, 1),
        "ref_damaged_in_aoi": len(ref),
        "our_hit": len(hit),
        "our_hazard_km2": round(len(hz) * abs(tf.a * tf.e) / 1e6, 2),
    }
    if len(ref) and len(hz):
        near = gpd.sjoin_nearest(
            ref[["geometry"]].reset_index(drop=True),
            gpd.GeoDataFrame(geometry=hz),
            max_distance=tol_m,
        )
        out["recall"] = round(near.index.nunique() / len(ref), 3)
    if len(hit) and len(ref):
        near = gpd.sjoin_nearest(
            hit[["geometry"]].reset_index(drop=True),
            ref[["geometry"]].reset_index(drop=True),
            max_distance=tol_m,
        )
        out["precision"] = round(near.index.nunique() / len(hit), 3)
    return out


if __name__ == "__main__":
    for j in sys.argv[1:]:
        print(json.dumps(compare(j)))
