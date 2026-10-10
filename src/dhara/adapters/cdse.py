"""CDSE access: OAuth token, STAC scene search, and Sentinel Hub Process API (AOI-only rasters)."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from dhara.adapters.allowlist import guarded_request
from dhara.config import get_settings
from dhara.timeutil import iso_utc

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
SH_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Sentinel Hub limits a single process request to 2500 x 2500 output pixels.
SH_MAX_PX = 2500
# Smaller tiles fetched concurrently are much faster than few large ones.
SH_TILE_PX = 1024
SH_WORKERS = 8
# Global cap on in-flight Sentinel Hub requests across all concurrent fetches.
_SH_SLOTS = threading.Semaphore(12)


class CDSEError(RuntimeError):
    pass


# --------------------------------------------------------------------------- auth

_token: tuple[str, float] | None = None
_token_lock = threading.Lock()


def get_token() -> str:
    with _token_lock:
        return _get_token()


def _get_token() -> str:
    global _token
    if _token and _token[1] - 60 > time.time():
        return _token[0]
    s = get_settings()
    if not (s.cdse_client_id and s.cdse_client_secret):
        raise CDSEError(
            "CDSE credentials missing: set CDSE_SENTINEL_HUB_CLIENT_ID and CDSE_SENTINEL_HUB_CLIENT_SECRET in .env "
            "(Sentinel Hub OAuth client from https://shapps.dataspace.copernicus.eu/dashboard/)."
        )
    resp = guarded_request(
        "POST",
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": s.cdse_client_id.get_secret_value(),
            "client_secret": s.cdse_client_secret.get_secret_value(),
        },
    )
    if resp.status_code != 200:
        raise CDSEError(f"CDSE token request failed ({resp.status_code}): {resp.text[:300]}")
    body = resp.json()
    _token = (body["access_token"], time.time() + float(body.get("expires_in", 600)))
    return _token[0]


# --------------------------------------------------------------------------- STAC


def stac_search(
    collection: str,
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    *,
    max_items: int = 500,
) -> list[dict[str, Any]]:
    """Search the public CDSE STAC. bbox in EPSG:4326; start/end aware datetimes.

    Follows `next` links (POST body paging).
    """
    body: dict[str, Any] = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": f"{iso_utc(start)}/{iso_utc(end)}",
        "limit": 100,
    }
    items: list[dict[str, Any]] = []
    url = STAC_SEARCH_URL
    while url and len(items) < max_items:
        resp = guarded_request("POST", url, json=body)
        if resp.status_code != 200:
            raise CDSEError(f"STAC search failed ({resp.status_code}): {resp.text[:300]}")
        page = resp.json()
        items += page.get("features", [])
        nxt = next((ln for ln in page.get("links", []) if ln.get("rel") == "next"), None)
        if not nxt:
            break
        url = nxt["href"]
        body = nxt.get("body", body)
    return items


# --------------------------------------------------------------------------- Sentinel Hub


@dataclass(frozen=True)
class Grid:
    """Output pixel grid. CRS: projected (local UTM). bounds = (minx, miny, maxx, maxy) in metres."""

    crs: CRS
    bounds: tuple[float, float, float, float]
    res: float

    @property
    def width(self) -> int:
        return math.ceil((self.bounds[2] - self.bounds[0]) / self.res)

    @property
    def height(self) -> int:
        return math.ceil((self.bounds[3] - self.bounds[1]) / self.res)

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.bounds[0], 0, -self.res, self.bounds[3])

    def to_pixel(self, x, y):
        """World (grid CRS) → fractional (col, row). Vectorised; works on scalars and arrays."""
        return (np.asarray(x) - self.bounds[0]) / self.res, (
            self.bounds[3] - np.asarray(y)
        ) / self.res

    def to_world(self, col, row):
        """Fractional (col, row) → world coordinates (grid CRS); pixel corners at integers."""
        return self.bounds[0] + np.asarray(col) * self.res, self.bounds[3] - np.asarray(
            row
        ) * self.res

    @classmethod
    def snapped(cls, crs: CRS, bounds: tuple[float, float, float, float], res: float) -> "Grid":
        minx = math.floor(bounds[0] / res) * res
        miny = math.floor(bounds[1] / res) * res
        maxx = math.ceil(bounds[2] / res) * res
        maxy = math.ceil(bounds[3] / res) * res
        return cls(crs, (minx, miny, maxx, maxy), res)


def _sh_tile(
    grid: Grid,
    col0: int,
    row0: int,
    w: int,
    h: int,
    data: list[dict[str, Any]],
    evalscript: str,
    n_bands: int,
) -> np.ndarray:
    minx = grid.bounds[0] + col0 * grid.res
    maxy = grid.bounds[3] - row0 * grid.res
    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, maxy - h * grid.res, minx + w * grid.res, maxy],
                "properties": {
                    "crs": f"http://www.opengis.net/def/crs/EPSG/0/{grid.crs.to_epsg()}"
                },
            },
            "data": data,
        },
        "output": {
            "width": w,
            "height": h,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }
    with _SH_SLOTS:
        resp = guarded_request(
            "POST",
            SH_PROCESS_URL,
            json=payload,
            headers={"Authorization": f"Bearer {get_token()}", "Accept": "image/tiff"},
        )
    if resp.status_code != 200:
        raise CDSEError(f"Sentinel Hub process failed ({resp.status_code}): {resp.text[:500]}")
    with MemoryFile(resp.content) as mf, mf.open() as ds:
        arr = ds.read().astype(np.float32)
    if arr.shape[0] != n_bands:
        raise CDSEError(f"Sentinel Hub returned {arr.shape[0]} bands, expected {n_bands}")
    return arr


def sh_process(
    grid: Grid,
    data: list[dict[str, Any]],
    evalscript: str,
    n_bands: int,
    *,
    tile_px: int = SH_TILE_PX,
    workers: int = SH_WORKERS,
) -> np.ndarray:
    """Fetch a (bands, H, W) float32 array on `grid`.

    The grid is cut into tiles of `tile_px` (≤ the SH 2500 px limit), and the tiles are requested
    concurrently; server-side orthorectification is the slow part, so parallel tiles cut latency
    roughly linearly until the account's rate limit (429s are retried with backoff).
    """
    tile_px = min(tile_px, SH_MAX_PX)
    out = np.full((n_bands, grid.height, grid.width), np.nan, dtype=np.float32)
    jobs = [
        (row0, col0, min(tile_px, grid.height - row0), min(tile_px, grid.width - col0))
        for row0 in range(0, grid.height, tile_px)
        for col0 in range(0, grid.width, tile_px)
    ]

    def one(job):
        row0, col0, h, w = job
        return job, _sh_tile(grid, col0, row0, w, h, data, evalscript, n_bands)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        for (row0, col0, h, w), arr in pool.map(one, jobs):
            out[:, row0 : row0 + h, col0 : col0 + w] = arr
    return out


def write_cog(path: str, arr: np.ndarray, grid: Grid, nodata: float | None = None) -> None:
    """Write (bands, H, W) or (H, W) as a Cloud-Optimized GeoTIFF on `grid` (grid CRS)."""
    if arr.ndim == 2:
        arr = arr[None]
    profile = {
        "driver": "COG",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": arr.dtype.name,
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "deflate",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
