"""Web-ready exports for a finished job.

- Raster overlays: reprojected from local UTM to EPSG:3857 and written as RGBA PNGs. Corner
  coordinates are given in EPSG:4326 (a MapLibre `image` source).
- Vector layers: GeoJSON in EPSG:4326.
- layers.json: a manifest the API/UI reads.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from dhara.adapters.cdse import Grid
from dhara.fusion import fuse as F

# RGBA per class. not_observable is a distinct grey so it never reads as "no damage".
CLASS_RGBA = {
    F.NOT_OBSERVABLE: (120, 113, 108, 150),
    F.NO_CHANGE: (0, 0, 0, 0),
    F.PERMANENT_WATER: (56, 135, 190, 170),
    F.FLOOD: (29, 78, 216, 220),
    F.DEBRIS: (194, 120, 3, 220),
    F.OUTSIDE: (0, 0, 0, 0),
}
TIER_RGBA = {
    255: (0, 0, 0, 0),
    0: (0, 0, 0, 0),
    1: (250, 204, 21, 200),
    2: (249, 115, 22, 210),
    3: (185, 28, 28, 220),
}

VECTOR_LAYERS = {
    "buildings": "buildings.parquet",
    "roads": "roads.parquet",
    "bridges": "bridges.parquet",
    "settlements": "settlements.parquet",
    "failed_edges": "failed_edges.parquet",
}


def _to_mercator(path: Path, nodata) -> tuple[np.ndarray, list[list[float]]]:
    with rasterio.open(path) as src:
        dst_crs = "EPSG:3857"
        tf, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        out = np.full((h, w), nodata, dtype=src.dtypes[0])
        reproject(
            rasterio.band(src, 1),
            out,
            src_nodata=nodata,
            dst_transform=tf,
            dst_crs=dst_crs,
            dst_nodata=nodata,
            resampling=Resampling.nearest,
        )
    from pyproj import Transformer

    t = Transformer.from_crs(3857, 4326, always_xy=True)
    x0, y0 = tf.c, tf.f
    x1, y1 = tf.c + tf.a * w, tf.f + tf.e * h
    corners = [t.transform(x0, y0), t.transform(x1, y0), t.transform(x1, y1), t.transform(x0, y1)]
    return out, [[round(a, 7), round(b, 7)] for a, b in corners]


def _png(path: Path, rgba: np.ndarray) -> None:
    with rasterio.open(
        path, "w", driver="PNG", width=rgba.shape[2], height=rgba.shape[1], count=4, dtype="uint8"
    ) as d:
        d.write(rgba)


def _palette(arr: np.ndarray, pal: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    rgba = np.zeros((4, *arr.shape), np.uint8)
    for k, c in pal.items():
        m = arr == k
        for i in range(4):
            rgba[i][m] = c[i]
    return rgba


def _ramp(p: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    rgba = np.zeros((4, *p.shape), np.uint8)
    ok = np.isfinite(p) & (p >= 0.2)
    for i in range(3):
        rgba[i][ok] = rgb[i]
    rgba[3][ok] = (np.clip((p[ok] - 0.2) / 0.8, 0, 1) * 230).astype(np.uint8)
    return rgba


def _jsonable(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    return json.dumps(v) if isinstance(v, list | dict) else v


def export_layers(job_dir: Path, grid: Grid, in_aoi: np.ndarray) -> dict:
    """Write web/*.png, web/*.geojson and layers.json. Returns the manifest."""
    web = job_dir / "web"
    web.mkdir(exist_ok=True)
    manifest: dict = {"rasters": {}, "vectors": {}}

    cls, corners = _to_mercator(job_dir / "classes.tif", F.OUTSIDE)
    _png(web / "classes.png", _palette(cls, CLASS_RGBA))
    manifest["rasters"]["classes"] = {"url": "web/classes.png", "coordinates": corners}
    # one image per class so the UI can toggle each independently
    for name, code in (
        ("debris", F.DEBRIS),
        ("flood", F.FLOOD),
        ("water", F.PERMANENT_WATER),
        ("unobs", F.NOT_OBSERVABLE),
    ):
        _png(web / f"{name}.png", _palette(cls, {code: CLASS_RGBA[code]}))
        manifest["rasters"][name] = {"url": f"web/{name}.png", "coordinates": corners}
    tier, corners = _to_mercator(job_dir / "confidence.tif", 255)
    _png(web / "confidence.png", _palette(tier, TIER_RGBA))
    manifest["rasters"]["confidence"] = {"url": "web/confidence.png", "coordinates": corners}
    for name, rgb in (("p_flood", (29, 78, 216)), ("p_debris", (194, 120, 3))):
        p, corners = _to_mercator(job_dir / f"{name}.tif", np.float32(np.nan))
        _png(web / f"{name}.png", _ramp(p, rgb))
        manifest["rasters"][name] = {"url": f"web/{name}.png", "coordinates": corners}

    for name, fname in VECTOR_LAYERS.items():
        src = job_dir / fname
        if not src.exists():
            continue
        g = gpd.read_parquet(src).to_crs("EPSG:4326")
        if name == "buildings":
            g = g[g["status"] != "not_hit"]  # keep the file small: hit + not observable
        for c in g.columns:
            if c != "geometry" and g[c].dtype == object:
                g[c] = g[c].map(_jsonable)
        out = web / f"{name}.geojson"
        out.write_text(g.to_json(drop_id=True))
        manifest["vectors"][name] = {"url": f"web/{name}.geojson", "count": len(g)}
    (job_dir / "layers.json").write_text(json.dumps(manifest, indent=1))
    return manifest
