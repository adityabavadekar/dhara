"""Copernicus DEM GLO-30 (allowed input) as 1°x1° COG tiles, read with windows.

Route 1 (preferred, needs CDSE S3 keys in .env): CDSE eodata S3,
  /vsis3/eodata/auxdata/CopDEM_COG/copernicus-dem-30m/<tile>/<tile>.tif
Route 2 (no credentials): the official open GLO-30 COG distribution (AWS Registry of Open Data),
  identical tiles. CDSE Sentinel Hub `COPERNICUS_30` returns 403 for this account, so it isn't used.
"""

import math

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
    s = get_settings()
    if s.cdse_s3_access_key and s.cdse_s3_secret_key:
        return CDSE_S3, {
            "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
            "AWS_ACCESS_KEY_ID": s.cdse_s3_access_key.get_secret_value(),
            "AWS_SECRET_ACCESS_KEY": s.cdse_s3_secret_key.get_secret_value(),
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_HTTPS": "YES",
        }
    return MIRROR, {"AWS_NO_SIGN_REQUEST": "YES"}


def fetch_dem(grid: Grid) -> tuple[np.ndarray, dict]:
    """GLO-30 heights (m, EGM2008) resampled bilinearly onto `grid` (projected CRS).

    Returns (array with NaN where no tile/sea, provenance dict).
    """
    from shapely import box

    ll = reproject_geom(box(*grid.bounds), grid.crs, "EPSG:4326").bounds
    base, env = _route()
    out = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    used, missing = [], []
    gdal_env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    }
    with Env(**gdal_env, **env):
        for t in tiles_for_bounds(ll):
            path = f"{base}/{t}/{t}.tif"
            try:
                with guarded_raster_open(path) as ds:
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
                    used.append(t)
            except RasterioIOError:
                missing.append(t)  # ocean tiles do not exist
    if not used:
        raise RuntimeError(f"No Copernicus DEM GLO-30 tiles readable for {ll} (tried {missing})")
    return out, {
        "dataset": "Copernicus DEM GLO-30",
        "route": base,
        "tiles": used,
        "missing": missing,
    }
