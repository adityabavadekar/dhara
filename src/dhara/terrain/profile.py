"""Longitudinal elevation profile along routes and debris surge paths.

Samples Copernicus DEM GLO-30 along an arbitrary EPSG:4326 LineString:
- Reprojects to local UTM (meters) for accurate geodesic distance measurement.
- Samples elevation at fixed intervals (default 30 m).
- Computes cumulative distance, elevation (m MSL), and grade/slope percentage.
- Outputs distance vs. elevation points and summary statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import LineString


def sample_profile_from_dem(
    coords_4326: list[list[float]],
    dem_path: Path,
    step_m: float = 30.0,
) -> dict[str, Any]:
    """Sample elevation along a line of [lng, lat] coordinates against a DEM GeoTIFF.

    Args:
        coords_4326: List of [lng, lat] coordinates in EPSG:4326.
        dem_path: Path to Copernicus DEM GeoTIFF in local UTM.
        step_m: Sampling step along the linestring in meters.

    Returns:
        Dictionary with 'points' list and 'summary' dictionary.
    """
    if len(coords_4326) < 2 or not dem_path.exists():
        return {
            "points": [],
            "summary": {
                "total_distance_km": 0.0,
                "min_elevation_m": 0,
                "max_elevation_m": 0,
                "elevation_gain_m": 0,
                "elevation_loss_m": 0,
                "max_slope_pct": 0.0,
                "avg_slope_pct": 0.0,
            },
        }

    with rasterio.open(dem_path) as ds:
        dem_crs = ds.crs
        transform = ds.transform
        dem_data = ds.read(1)

    to_utm = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    to_4326 = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)

    utm_coords = [to_utm.transform(c[0], c[1]) for c in coords_4326]
    line = LineString(utm_coords)
    total_len_m = line.length

    if total_len_m <= 0:
        return {
            "points": [],
            "summary": {
                "total_distance_km": 0.0,
                "min_elevation_m": 0,
                "max_elevation_m": 0,
                "elevation_gain_m": 0,
                "elevation_loss_m": 0,
                "max_slope_pct": 0.0,
                "avg_slope_pct": 0.0,
            },
        }

    # Number of sample intervals
    num_samples = max(2, int(np.ceil(total_len_m / step_m)) + 1)
    sample_dists = np.linspace(0, total_len_m, num_samples)

    points = []
    inv_transform = ~transform

    raw_elevs: list[float | None] = []
    for dist_m in sample_dists:
        pt = line.interpolate(dist_m)
        px, py = pt.x, pt.y

        # Pixel coords
        col, row = inv_transform * (px, py)
        r, c = int(round(row)), int(round(col))

        elev: float | None = None
        if 0 <= r < dem_data.shape[0] and 0 <= c < dem_data.shape[1]:
            val = dem_data[r, c]
            if np.isfinite(val) and val > -500:
                elev = float(val)

        raw_elevs.append(elev)

    # Fill nodata/out-of-bounds with nearest valid elevations
    valid_vals = [e for e in raw_elevs if e is not None]
    default_val = float(np.median(valid_vals)) if valid_vals else 0.0

    filled_elevs: list[float] = []
    last_seen = default_val
    for e in raw_elevs:
        if e is not None:
            last_seen = e
        filled_elevs.append(last_seen)

    # Back-fill start if needed
    for i in range(len(filled_elevs) - 2, -1, -1):
        if raw_elevs[i] is None and raw_elevs[i + 1] is not None:
            filled_elevs[i] = filled_elevs[i + 1]

    for dist_m, elev in zip(sample_dists, filled_elevs, strict=False):
        pt = line.interpolate(dist_m)
        lng, lat = to_4326.transform(pt.x, pt.y)
        points.append(
            {
                "dist_km": round(float(dist_m) / 1000.0, 3),
                "elevation_m": round(float(elev), 1),
                "lng": round(float(lng), 5),
                "lat": round(float(lat), 5),
                "slope_pct": 0.0,
            }
        )

    # Compute slopes between consecutive points
    total_gain = 0.0
    total_loss = 0.0
    max_slope = 0.0
    slopes = []

    for i in range(1, len(points)):
        p_prev = points[i - 1]
        p_curr = points[i]
        d_m = (p_curr["dist_km"] - p_prev["dist_km"]) * 1000.0
        dh = p_curr["elevation_m"] - p_prev["elevation_m"]

        if d_m > 0:
            slope = (dh / d_m) * 100.0
            p_curr["slope_pct"] = round(float(slope), 1)
            slopes.append(abs(slope))
            max_slope = max(max_slope, abs(slope))

        if dh > 0:
            total_gain += dh
        else:
            total_loss += abs(dh)

    if points:
        elevations = [p["elevation_m"] for p in points]
        min_elev = round(min(elevations))
        max_elev = round(max(elevations))
    else:
        min_elev = 0
        max_elev = 0

    return {
        "points": points,
        "summary": {
            "total_distance_km": round(total_len_m / 1000.0, 2),
            "min_elevation_m": min_elev,
            "max_elevation_m": max_elev,
            "elevation_gain_m": round(total_gain),
            "elevation_loss_m": round(total_loss),
            "max_slope_pct": round(max_slope, 1),
            "avg_slope_pct": round(float(np.mean(slopes)), 1) if slopes else 0.0,
        },
    }


def sample_profile(coords_4326: list[list[float]], step_m: float = 30.0) -> dict[str, Any]:
    """Profile along any EPSG:4326 line, wherever it goes: a 30 m DEM grid (local UTM) is built
    around the line from the cached Copernicus DEM tiles, so routes that leave the analysis area
    are still sampled. Falls back to an empty profile when the line is degenerate."""
    import tempfile

    from dhara.adapters.cdse import Grid, write_cog
    from dhara.adapters.dem import fetch_dem

    if len(coords_4326) < 2:
        return sample_profile_from_dem(coords_4326, Path("/nonexistent"), step_m)
    lon0 = float(np.mean([c[0] for c in coords_4326]))
    lat0 = float(np.mean([c[1] for c in coords_4326]))
    zone = int((lon0 + 180) // 6) + 1
    crs = CRS.from_epsg((32600 if lat0 >= 0 else 32700) + zone)
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xy = [to_utm.transform(c[0], c[1]) for c in coords_4326]
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    pad = 300.0
    grid = Grid.snapped(crs, (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad), 30.0)
    dem, _ = fetch_dem(grid)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "dem.tif"
        write_cog(str(path), dem, grid, nodata=float("nan"))
        return sample_profile_from_dem(coords_4326, path, step_m)
