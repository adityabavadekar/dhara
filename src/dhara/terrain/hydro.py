"""Stage 2: terrain layers from Copernicus DEM GLO-30.

Conditioned DEM, D8 flow direction and accumulation, a drainage network, HAND (height above
nearest drainage) and slope. Computed on a 30 m grid in local UTM, buffered around the AOI so that
flow routing sees terrain upstream of the AOI edge. Results are resampled to the 10 m analysis grid.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from dhara.adapters.cdse import Grid, write_cog
from dhara.terrain import d8

TERRAIN_RES_M = 30.0
# Upstream area that starts a channel in the drainage network.
CHANNEL_AREA_KM2 = 0.5


@dataclass
class Terrain:
    """All arrays on `grid` (30 m, local UTM). fdir: index 0..7 into d8.DR/DC, -1 = outlet/no data."""

    grid: Grid
    dem: np.ndarray  # m, raw
    dem_cond: np.ndarray  # m, depression-filled, flats resolved
    fdir: np.ndarray  # int32 D8 index; -1 = no outflow
    acc: np.ndarray  # upstream cell count
    channels: np.ndarray  # bool
    hand: np.ndarray  # m
    slope_deg: np.ndarray

    @property
    def channel_area_km2(self) -> float:
        return CHANNEL_AREA_KM2


def terrain_grid(analysis: Grid, buffer_m: float) -> Grid:
    """30 m grid (same CRS) covering the analysis grid plus a buffer."""
    minx, miny, maxx, maxy = analysis.bounds
    b = (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)
    return Grid.snapped(analysis.crs, b, TERRAIN_RES_M)


def slope_degrees(dem: np.ndarray, res: float) -> np.ndarray:
    """Slope (degrees) of a DEM on a square grid of `res` metres."""
    return d8.slope_deg(np.ascontiguousarray(dem, dtype=np.float64), float(res))


def compute(dem: np.ndarray, grid: Grid, channel_area_km2: float = CHANNEL_AREA_KM2) -> Terrain:
    """Hydrological conditioning and derived layers. dem: (H, W) metres on `grid` (NaN = no data)."""
    dem = dem.astype(np.float64)
    cond, order = d8.priority_flood_eps(dem, 1e-3)
    fd = d8.flow_direction(cond)
    acc = d8.accumulation(cond, fd, order)
    channels = acc * (grid.res**2) / 1e6 >= channel_area_km2
    h = d8.hand(cond, fd, channels, order)
    h[h < 0] = 0.0
    filled = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    return Terrain(
        grid=grid,
        dem=dem.astype(np.float32),
        dem_cond=cond.astype(np.float32),
        fdir=fd.astype(np.int32),
        acc=acc.astype(np.float32),
        channels=channels,
        hand=h,
        slope_deg=slope_degrees(filled, grid.res),
    )


def resample_to(arr: np.ndarray, src: Grid, dst: Grid, method=Resampling.bilinear) -> np.ndarray:
    """Resample a 2-D array from grid `src` to grid `dst` (both projected; may differ in CRS/res)."""
    out = np.full((dst.height, dst.width), np.nan, dtype=np.float32)
    reproject(
        arr.astype(np.float32),
        out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst.transform,
        dst_crs=dst.crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=method,
    )
    return out


def trace_flowpath(t: Terrain, x: float, y: float, max_steps: int = 200_000) -> np.ndarray:
    """D8 steepest-descent path from (x, y) in t.grid.crs to the grid edge or a sink.

    Returns an (N, 2) array of cell-centre coordinates (t.grid.crs).
    """
    c, r = t.grid.to_pixel(x, y)
    r, c = int(r), int(c)
    H, W = t.fdir.shape
    if not (0 <= r < H and 0 <= c < W):
        raise ValueError("start point is outside the terrain grid")
    pts = []
    seen = set()
    for _ in range(max_steps):
        cx, cy = t.grid.to_world(c + 0.5, r + 0.5)
        pts.append((float(cx), float(cy)))
        seen.add((r, c))
        k = int(t.fdir[r, c])
        if k < 0:
            break
        r, c = r + int(d8.DR[k]), c + int(d8.DC[k])
        if not (0 <= r < H and 0 <= c < W) or (r, c) in seen:
            break
    return np.array(pts)


def save(t: Terrain, out_dir: Path) -> None:
    for name in ("dem", "dem_cond", "acc", "hand", "slope_deg"):
        write_cog(
            str(out_dir / f"terrain_{name}.tif"), getattr(t, name), t.grid, nodata=float("nan")
        )
    write_cog(str(out_dir / "terrain_fdir.tif"), t.fdir, t.grid, nodata=None)
    write_cog(
        str(out_dir / "terrain_channels.tif"), t.channels.astype(np.uint8), t.grid, nodata=None
    )


def load(out_dir: Path, grid: Grid) -> Terrain | None:
    files = {
        n: out_dir / f"terrain_{n}.tif"
        for n in ("dem", "dem_cond", "acc", "hand", "slope_deg", "fdir", "channels")
    }
    if not all(p.exists() for p in files.values()):
        return None
    a = {}
    for n, p in files.items():
        with rasterio.open(p) as ds:
            a[n] = ds.read(1)
    return Terrain(
        grid=grid,
        dem=a["dem"],
        dem_cond=a["dem_cond"],
        fdir=a["fdir"].astype(np.int32),
        acc=a["acc"],
        channels=a["channels"].astype(bool),
        hand=a["hand"],
        slope_deg=a["slope_deg"],
    )
