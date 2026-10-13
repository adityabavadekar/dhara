"""Web-ready exports for a finished job.

- Raster overlays: reprojected from local UTM to EPSG:3857 and written as RGBA PNGs. Corner
  coordinates are given in EPSG:4326 (a MapLibre `image` source).
- Vector layers: GeoJSON in EPSG:4326.
- layers.json: a manifest the API/UI reads.
"""

import contextlib
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
    "hospitals": "hospitals.parquet",
    "hlz": "hlz.parquet",
}


def _to_mercator(
    path: Path, nodata, resampling: Resampling = Resampling.nearest, max_px: int | None = None
) -> tuple[np.ndarray, list[list[float]]]:
    with rasterio.open(path) as src:
        dst_crs = "EPSG:3857"
        tf, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        if max_px and max(w, h) > max_px:  # coarser web copy for large areas
            f = max(w, h) / max_px
            tf = tf * tf.scale(f, f)
            w, h = int(np.ceil(w / f)), int(np.ceil(h / f))
        out = np.full((h, w), nodata, dtype=src.dtypes[0])
        reproject(
            rasterio.band(src, 1),
            out,
            src_nodata=nodata,
            dst_transform=tf,
            dst_crs=dst_crs,
            dst_nodata=nodata,
            resampling=resampling,
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


def _cached_s1(cache: Path, orbit: int, start: str) -> Path | None:
    """The cached γ⁰ pass for a relative orbit and start time (local UTM GeoTIFF), if present."""
    stamp = start[:19].replace("-", "").replace(":", "").replace(" ", "T")
    hits = sorted(cache.glob(f"s1_*_R{orbit:03d}_{stamp}Z_GAMMA0_TERRAIN*.tif"))
    return hits[0] if hits else None


def _optical_rgb(path: Path, max_px: int) -> tuple[np.ndarray, list[list[float]]]:
    """True colour (B04, B03, B02) of a cached Sentinel-2 L2A scene as RGBA in EPSG:3857.
    Reflectance 0–0.3 is stretched with a mild gamma, the usual look of natural-colour imagery."""
    bands = []
    corners: list[list[float]] = []
    for b in (3, 2, 1):  # cached band order: B02, B03, B04, B08, B11, B12, SCL
        arr, corners = _to_mercator_band(path, b, max_px)
        bands.append(arr)
    rgb = np.stack(bands)
    ok = np.all(np.isfinite(rgb), axis=0)
    v = np.clip(np.nan_to_num(rgb) / 0.3, 0, 1) ** (1 / 1.4)
    out = np.zeros((4, *rgb.shape[1:]), np.uint8)
    out[:3] = (v * 255).astype(np.uint8)
    out[3] = np.where(ok, 255, 0).astype(np.uint8)
    return out, corners


def _to_mercator_band(path: Path, band: int, max_px: int) -> tuple[np.ndarray, list[list[float]]]:
    from pyproj import Transformer

    with rasterio.open(path) as src:
        tf, w, h = calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds
        )
        if max(w, h) > max_px:
            f = max(w, h) / max_px
            tf = tf * tf.scale(f, f)
            w, h = int(np.ceil(w / f)), int(np.ceil(h / f))
        out = np.full((h, w), np.nan, dtype=np.float32)
        reproject(
            rasterio.band(src, band),
            out,
            src_nodata=np.nan,
            dst_transform=tf,
            dst_crs="EPSG:3857",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    t = Transformer.from_crs(3857, 4326, always_xy=True)
    x0, y0, x1, y1 = tf.c, tf.f, tf.c + tf.a * w, tf.f + tf.e * h
    cs = [t.transform(x0, y0), t.transform(x1, y0), t.transform(x1, y1), t.transform(x0, y1)]
    return out, [[round(a, 7), round(b, 7)] for a, b in cs]


def export_optical(job_dir: Path, cache: Path, s2: dict | None) -> dict | None:
    """Before/after Sentinel-2 true colour for the sweep, when both scenes of the job's S2 pair
    are cached. Returns the manifest entry or None."""
    if not s2 or not s2.get("pre") or not s2.get("post"):
        return None
    pre = cache / f"s2_{s2['pre']['day']}.tif"
    post = cache / f"s2_{s2['post']['day']}.tif"
    if not pre.exists() or not post.exists():
        return None
    web = job_dir / "web"
    out: dict = {"kind": "optical", "sensor": "Sentinel-2 true colour", "track": ""}
    for name, path, day in (("before", pre, s2["pre"]["day"]), ("after", post, s2["post"]["day"])):
        rgba, corners = _optical_rgb(path, 2600)
        _png(web / f"optical_{name}.png", rgba)
        out[name] = {
            "url": f"web/optical_{name}.png",
            "coordinates": corners,
            "date": str(day)[:10],
        }
    return out


def export_imagery(
    job_dir: Path, cache: Path, tracks: list[dict], manifest: dict, s2: dict | None = None
) -> None:
    """Before/after Sentinel-1 VV backscatter of one track as greyscale PNGs (EPSG:3857 image,
    EPSG:4326 corners) for the map's before/after sweep. Both images share one dB stretch so
    brightness differences are real changes. Skipped silently when the passes are not cached."""
    for t in tracks:
        pre = _cached_s1(cache, t["relative_orbit"], str(t["pre_start"]))
        post = _cached_s1(cache, t["relative_orbit"], str(t["post_start"]))
        if pre is None or post is None:
            continue
        nan = np.float32(np.nan)
        a, corners = _to_mercator(pre, nan, Resampling.bilinear, max_px=2600)
        b, _ = _to_mercator(post, nan, Resampling.bilinear, max_px=2600)
        da = 10 * np.log10(np.clip(a, 1e-4, None))
        db = 10 * np.log10(np.clip(b, 1e-4, None))
        both = np.concatenate([da[np.isfinite(a)], db[np.isfinite(b)]])
        if both.size == 0:
            continue
        lo, hi = np.percentile(both, [2, 98])
        web = job_dir / "web"
        out = {
            "kind": "radar",
            "sensor": "Sentinel-1 VV",
            "track": f"{t['orbit_state']} {t['relative_orbit']}",
        }
        for name, arr, raw, when in (
            ("before", da, a, t["pre_start"]),
            ("after", db, b, t["post_start"]),
        ):
            g = (np.nan_to_num(np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)) * 255).astype(
                np.uint8
            )
            rgba = np.stack([g, g, g, np.where(np.isfinite(raw), 255, 0).astype(np.uint8)])
            _png(web / f"radar_{name}.png", rgba)
            out[name] = {
                "url": f"web/radar_{name}.png",
                "coordinates": corners,
                "date": str(when)[:10],
            }
        manifest["imagery_radar"] = out
        break
    # Optical is what people recognise; radar stays available as the all-weather alternative.
    optical = export_optical(job_dir, cache, s2)
    if optical:
        manifest["imagery"] = optical
    elif "imagery_radar" in manifest:
        manifest["imagery"] = manifest["imagery_radar"]


def export_change(
    job_dir: Path, cache: Path, s2: dict | None, in_aoi: np.ndarray, manifest: dict
) -> None:
    """Measured before/after change (open water, river-channel width and shift) from the job's
    cached Sentinel-2 pair, for the sweep. Writes web/change.geojson (EPSG:4326) and
    manifest["change"]. Skipped when the pair or the terrain is not cached."""
    from dhara.damage.change import measure

    if not s2 or not s2.get("pre") or not s2.get("post"):
        return
    b, a = str(s2["pre"]["day"])[:10], str(s2["post"]["day"])[:10]
    pre, post = cache / f"s2_{b}.tif", cache / f"s2_{a}.tif"
    if not (pre.exists() and post.exists() and (cache / "terrain" / "terrain_acc.tif").exists()):
        return
    summary, g = measure(pre, post, cache / "terrain", in_aoi, b, a)
    if len(g):
        g = g.copy()
        g["geometry"] = g.geometry.simplify(5)
        (job_dir / "web" / "change.geojson").write_text(g.to_crs(4326).to_json(drop_id=True))
        manifest["vectors"]["change"] = {"url": "web/change.geojson", "count": len(g)}
    manifest["change"] = summary.model_dump()


def export_hazard_outline(job_dir: Path, manifest: dict) -> None:
    """Outlines of mapped flood and debris (EPSG:4326 GeoJSON), so the same places can be compared
    on both sides of the before/after sweep. Patches are slightly merged (one-pixel closing) and
    simplified to ~10 m; this is a display aid, the rasters remain the record."""
    from rasterio.features import shapes
    from scipy.ndimage import binary_closing
    from shapely.geometry import shape

    with rasterio.open(job_dir / "classes.tif") as ds:
        cls = ds.read(1)
        tf, crs = ds.transform, ds.crs
    feats = []
    for name, code in (("debris", F.DEBRIS), ("flood", F.FLOOD)):
        m = binary_closing(cls == code, iterations=1)
        for geom, v in shapes(m.astype(np.uint8), mask=m, transform=tf):
            if v:
                feats.append({"hazard": name, "geometry": shape(geom)})
    web = job_dir / "web"
    if not feats:
        return
    g = gpd.GeoDataFrame(feats, crs=crs)
    g["geometry"] = g.geometry.simplify(10)
    g.to_crs(4326).to_file(web / "hazard_outline.geojson", driver="GeoJSON")
    manifest["vectors"]["hazard_outline"] = {"url": "web/hazard_outline.geojson", "count": len(g)}


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


def export_layers(
    job_dir: Path,
    grid: Grid,
    in_aoi: np.ndarray,
    cache: Path | None = None,
    tracks: list[dict] | None = None,
    s2: dict | None = None,
) -> dict:
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

    try:
        from dhara.evac.hlz import identify_landing_zones

        hlz_gdf = identify_landing_zones(job_dir)
        if not hlz_gdf.empty:
            hlz_gdf.to_parquet(job_dir / "hlz.parquet")
    except Exception:
        pass

    for name, fname in VECTOR_LAYERS.items():
        src = job_dir / fname
        if not src.exists():
            continue
        g = gpd.read_parquet(src).to_crs("EPSG:4326")
        if name == "buildings":
            # keep hit + not_observable + any identified schools/medical/community facilities
            is_fac = (
                g["facility_type"].isin(["school", "health", "community"])
                if "facility_type" in g.columns
                else False
            )
            g = g[(g["status"] != "not_hit") | is_fac]
        for c in g.columns:
            if c != "geometry" and g[c].dtype == object:
                g[c] = g[c].map(_jsonable)
        out = web / f"{name}.geojson"
        out.write_text(g.to_json(drop_id=True))
        manifest["vectors"][name] = {"url": f"web/{name}.geojson", "count": len(g)}
    with contextlib.suppress(Exception):
        export_hazard_outline(job_dir, manifest)
    if cache is not None and tracks:
        with contextlib.suppress(Exception):  # the sweep is optional; never fail a job over it
            export_imagery(job_dir, cache, tracks, manifest, s2)
    if cache is not None:
        with contextlib.suppress(Exception):
            export_change(job_dir, cache, s2, in_aoi, manifest)
    (job_dir / "layers.json").write_text(json.dumps(manifest, indent=1))
    return manifest
