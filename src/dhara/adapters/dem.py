"""Copernicus DEM GLO-30 (allowed input) as 1°x1° COG tiles, read with windows.

Route 1 (preferred, needs CDSE S3 keys in .env): CDSE eodata S3,
  /vsis3/eodata/auxdata/CopDEM_COG/copernicus-dem-30m/<tile>/<tile>.tif
Route 2 (no credentials): the official open GLO-30 COG distribution (AWS Registry of Open Data),
  identical tiles. CDSE Sentinel Hub `COPERNICUS_30` returns 403 for this account, so it isn't used.
"""

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.errors import RasterioIOError
from rasterio.warp import Resampling, reproject

from dhara.adapters.allowlist import guarded_raster_open
from dhara.adapters.cdse import Grid
from dhara.config import get_settings
from dhara.geo import reproject as reproject_geom

MIRROR = "/vsicurl/https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
CDSE_S3 = "/vsis3/eodata/auxdata/CopDEM_COG/copernicus-dem-30m"


def tile_name(lat: int, lon: int) -> str:
    ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
    ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def tiles_for_bounds(bounds: tuple[float, float, float, float]) -> list[str]:
    """Tile names covering a lon/lat bbox (EPSG:4326)."""
    minx, miny, maxx, maxy = bounds
    return [
        tile_name(lat, lon)
        for lat in range(math.floor(miny), math.floor(maxy) + 1)
        for lon in range(math.floor(minx), math.floor(maxx) + 1)
    ]


def _route() -> tuple[str, dict]:
    # The official open GLO-30 COG distribution. CDSE S3 keys are not used here: they are refused
    # (AccessDenied) for the CopDEM_COG prefix, while the open mirror needs no credentials.
    return MIRROR, {"AWS_NO_SIGN_REQUEST": "YES"}


_tile_locks: dict[str, threading.Lock] = {}
_tile_locks_guard = threading.Lock()


def _local_tile(base: str, env: dict, name: str) -> Path | None:
    """Path of a cached copy of one GLO-30 tile, downloading it on first use (None = no tile).

    Tiles are 1°x1° and immutable, so every analysis and flow-path trace after the first one in a
    region reads them from disk instead of over HTTP.
    """
    cache = get_settings().cache_dir / "dem_tiles"
    cache.mkdir(parents=True, exist_ok=True)
    path, missing = cache / f"{name}.tif", cache / f"{name}.missing"
    with _tile_locks_guard:
        lock = _tile_locks.setdefault(name, threading.Lock())
    with lock:
        if path.exists():
            return path
        if missing.exists():
            return None
        gdal_env = {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        }
        try:
            with Env(**gdal_env, **env), guarded_raster_open(f"{base}/{name}/{name}.tif") as ds:
                data = ds.read(1)
                profile = ds.profile | {"driver": "GTiff", "compress": "deflate", "tiled": True}
        except RasterioIOError as e:
            msg = str(e).lower()
            if "does not exist" in msg or "no such file" in msg or "404" in msg:
                missing.touch()  # ocean tiles do not exist
                return None
            raise RuntimeError(
                f"Could not read Copernicus DEM tile {name} ({e}); elevation is required, so the "
                "job stops instead of running without it. Check the network and retry."
            ) from e
        tmp = path.with_suffix(".part")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data, 1)
        tmp.rename(path)
        return path


def fetch_dem(grid: Grid) -> tuple[np.ndarray, dict]:
    """GLO-30 heights (m, EGM2008) resampled bilinearly onto `grid` (projected CRS).

    Returns (array with NaN where no tile/sea, provenance dict). Tiles are fetched in parallel
    and cached on disk (see `_local_tile`).
    """
    from shapely import box

    ll = reproject_geom(box(*grid.bounds), grid.crs, "EPSG:4326").bounds
    base, env = _route()
    names = tiles_for_bounds(ll)
    with ThreadPoolExecutor(max_workers=min(4, len(names))) as pool:
        paths = list(pool.map(lambda n: _local_tile(base, env, n), names))
    out = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    used = [n for n, p in zip(names, paths, strict=True) if p is not None]
    missing = [n for n, p in zip(names, paths, strict=True) if p is None]
    for path in (p for p in paths if p is not None):
        with rasterio.open(path) as ds:
            tmp = np.full_like(out, np.nan)
            reproject(
                rasterio.band(ds, 1),
                tmp,
                src_transform=ds.transform,
                src_crs=ds.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                src_nodata=ds.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            fill = np.isnan(out) & np.isfinite(tmp)
            out[fill] = tmp[fill]
    if not used:
        raise RuntimeError(f"No Copernicus DEM GLO-30 tiles readable for {ll} (tried {missing})")
    return out, {
        "dataset": "Copernicus DEM GLO-30",
        "route": base,
        "tiles": used,
        "missing": missing,
    }
