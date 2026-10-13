"""Emergency Helicopter Landing Zone (HLZ) identification for cut-off communities.

Identifies viable rotorcraft landing clearings for settlements that lost ground road access:
- Analyzed in the job's local UTM CRS (meters).
- Slope constraint: <= 8.0 degrees (gentle gradient for safe skid/wheel contact).
- Hazard clearance: strictly outside flood, debris surge, and permanent water bodies.
- Distance threshold: <= 1200 m radius from the cut-off settlement.
- Minimum contiguous clearing area: >= 400 m^2 (approx 20 m x 20 m pad clearing).

Returns a GeoDataFrame in EPSG:4326.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from shapely.geometry import Point

from dhara.fusion import fuse as F

logger = logging.getLogger(__name__)

MAX_SLOPE_DEG = 8.0
MIN_CLEARING_M2 = 400.0
SEARCH_RADIUS_M = 1200.0


def _read_aligned(
    path: Path, dst_shape: tuple[int, int], dst_transform, dst_crs, nodata=np.nan
) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as src:
            if src.shape == dst_shape and src.transform == dst_transform and src.crs == dst_crs:
                return src.read(1)
            dst = np.full(dst_shape, nodata, dtype=np.float32)
            reproject(
                rasterio.band(src, 1),
                dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            return dst
    except Exception:
        return None


def identify_landing_zones(
    job_dir: Path,
    max_slope_deg: float = MAX_SLOPE_DEG,
    search_radius_m: float = SEARCH_RADIUS_M,
    min_area_m2: float = MIN_CLEARING_M2,
) -> gpd.GeoDataFrame:
    """Identify emergency helicopter landing zones for cut-off settlements.

    All geometric proximity and raster extraction operations run in local UTM.
    Final geometries and coordinates are output in EPSG:4326.
    """
    settlements_file = job_dir / "settlements.parquet"
    classes_file = job_dir / "classes.tif"

    if not settlements_file.exists() or not classes_file.exists():
        return gpd.GeoDataFrame(
            columns=[
                "name",
                "settlement",
                "distance_m",
                "elevation_m",
                "slope_deg",
                "area_m2",
                "suitability",
                "status",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    with rasterio.open(classes_file) as c_ds:
        utm_crs = c_ds.crs
        transform = c_ds.transform
        classes = c_ds.read(1)
        res_x = abs(transform.a)
        res_y = abs(transform.e)
        pixel_area_m2 = res_x * res_y

    settlements = gpd.read_parquet(settlements_file)
    if "status" not in settlements.columns:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    cut_off = settlements[settlements["status"].isin(["cut_off", "hospital_cut_off"])]
    if cut_off.empty:
        return gpd.GeoDataFrame(
            columns=[
                "name",
                "settlement",
                "distance_m",
                "elevation_m",
                "slope_deg",
                "area_m2",
                "suitability",
                "status",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    cut_off_utm = cut_off.to_crs(utm_crs)

    # Elevation and slope sources aligned to the classes raster
    dem = _read_aligned(job_dir / "dem.tif", classes.shape, transform, utm_crs)
    slope = _read_aligned(job_dir / "slope.tif", classes.shape, transform, utm_crs)

    # Fallback to computing slope from DEM if slope not cached separately
    if slope is None and dem is not None:
        gy, gx = np.gradient(dem.astype(np.float32), res_x)
        slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    # Safe dry ground mask: not flood, not debris, not permanent water, not unobservable
    safe_ground = (
        (classes != F.FLOOD)
        & (classes != F.DEBRIS)
        & (classes != F.PERMANENT_WATER)
        & (classes != F.NOT_OBSERVABLE)
        & (classes != F.OUTSIDE)
    )

    if slope is not None:
        safe_ground &= np.nan_to_num(slope, nan=99.0) <= max_slope_deg

    min_pixels = max(1, int(np.ceil(min_area_m2 / pixel_area_m2)))
    records = []

    # Pixel coordinate grid
    height, width = classes.shape
    rows, cols = np.indices((height, width))
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e

    for _, s_row in cut_off_utm.iterrows():
        s_point = s_row.geometry
        s_name = str(s_row.get("name") or "Settlement")
        sx, sy = s_point.x, s_point.y

        # Distance grid from settlement (m)
        dist_grid = np.hypot(xs - sx, ys - sy)
        roi_mask = safe_ground & (dist_grid <= search_radius_m)

        if not np.any(roi_mask):
            continue

        # Connected component labeling of flat, hazard-free ground
        labeled, num_features = ndimage.label(roi_mask)
        if num_features == 0:
            continue

        sizes = ndimage.sum(
            np.ones_like(roi_mask, dtype=np.int32), labeled, range(1, num_features + 1)
        )
        valid_comp_indices = [idx + 1 for idx, sz in enumerate(sizes) if sz >= min_pixels]

        if not valid_comp_indices:
            continue

        # Find the component with the minimum distance to the settlement
        best_comp = None
        min_comp_dist = float("inf")
        best_centroid_utm = None
        best_size_px = 0

        for comp_idx in valid_comp_indices:
            comp_mask = labeled == comp_idx
            mean_y = float(np.mean(rows[comp_mask]))
            mean_x = float(np.mean(cols[comp_mask]))
            cx = transform.c + (mean_x + 0.5) * transform.a
            cy = transform.f + (mean_y + 0.5) * transform.e
            d = float(np.hypot(cx - sx, cy - sy))

            if d < min_comp_dist:
                min_comp_dist = d
                best_comp = comp_mask
                best_centroid_utm = (cx, cy)
                best_size_px = int(np.sum(comp_mask))

        if best_comp is not None and best_centroid_utm is not None:
            cx, cy = best_centroid_utm
            comp_area_m2 = round(best_size_px * pixel_area_m2)
            comp_elevation = (
                round(float(np.nanmedian(dem[best_comp])))
                if dem is not None and np.any(np.isfinite(dem[best_comp]))
                else 0
            )
            comp_slope = (
                round(float(np.nanmean(slope[best_comp])), 1)
                if slope is not None and np.any(np.isfinite(slope[best_comp]))
                else 0.0
            )

            suitability = "optimal" if (comp_slope <= 4.5 and comp_area_m2 >= 800) else "viable"

            records.append(
                {
                    "name": f"HLZ · {s_name}",
                    "settlement": s_name,
                    "distance_m": round(min_comp_dist),
                    "elevation_m": comp_elevation,
                    "slope_deg": comp_slope,
                    "area_m2": comp_area_m2,
                    "suitability": suitability,
                    "status": "cleared",
                    "geometry": Point(cx, cy),
                }
            )

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "name",
                "settlement",
                "distance_m",
                "elevation_m",
                "slope_deg",
                "area_m2",
                "suitability",
                "status",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf_utm = gpd.GeoDataFrame(records, geometry="geometry", crs=utm_crs)
    gdf_4326 = gdf_utm.to_crs("EPSG:4326")
    return gdf_4326
