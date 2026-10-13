"""Before/after change measured from a Sentinel-2 pair: open water, river-channel width and how far
the channel moved, at stations along the main rivers.

Only pixels that are clear in both scenes (SCL, dilated as in `members.optical.clear`) and inside
the AOI are measured; everything else is left out, never counted as "no change".

Definitions (thresholds are uncalibrated and stated here so they can be quoted):
- open water: MNDWI > 0.05 (as `members.optical`);
- active channel: open water or unvegetated ground (NDVI < 0.15). After a debris flow this is the
  band of water, mud and fresh sediment a river occupies.
- main river: D8 channel cells (30 m Copernicus DEM) draining at least `MAIN_RIVER_KM2`.
- station: every `STATION_SPACING_M` along a main river; a transect perpendicular to the flow line,
  `TRANSECT_HALF_M` each side. A width is the run of channel pixels that crosses the river line.
  Stations with cloud on either date, outside the AOI, or whose run reaches the transect end are
  dropped.

CRS: everything is computed in the job's local UTM grid; vectors are returned in that CRS.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pydantic import BaseModel
from rasterio.features import shapes
from shapely.geometry import LineString, Point, shape

from dhara.adapters.sentinel2 import BANDS
from dhara.members.optical import clear, indices
from dhara.terrain import d8

WATER_MNDWI = 0.05
BARE_NDVI = 0.15
MAIN_RIVER_KM2 = 150.0
STATION_SPACING_M = 1000.0
TRANSECT_HALF_M = 700.0
SEARCH_M = 150.0  # how far from the DEM river line the channel may sit
SAMPLE_M = 10.0
WIDENED_M = 50.0  # a station counts as widened when the channel gained at least this much


class ChangeSummary(BaseModel):
    """Before/after measures over pixels clear on both dates. Areas in km², widths in m."""

    before_day: str
    after_day: str
    clear_both_km2: float
    clear_both_pct: float
    water_before_km2: float
    water_after_km2: float
    channel_before_km2: float
    channel_after_km2: float
    stations: int
    width_before_median_m: float | None = None
    width_after_median_m: float | None = None
    widened_stations: int = 0
    widened_threshold_m: float = WIDENED_M
    width_gain_max_m: float | None = None
    shift_median_m: float | None = None
    shift_max_m: float | None = None


def _read_s2(path: Path) -> tuple[dict[str, np.ndarray], rasterio.Affine]:
    with rasterio.open(path) as ds:
        a = ds.read().astype(np.float32)
        return {n: a[i] for i, n in enumerate([*BANDS, "SCL"])}, ds.transform


def _river_paths(acc: np.ndarray, fdir: np.ndarray, cell_m: float) -> list[list[tuple[int, int]]]:
    """Main-river cells (terrain grid) as downstream-ordered paths that stop at confluences."""
    stream = acc * cell_m * cell_m / 1e6 >= MAIN_RIVER_KM2
    h, w = stream.shape
    rr, cc = np.nonzero(stream)
    incoming = np.zeros(stream.shape, np.int32)
    for r, c in zip(rr, cc, strict=True):
        k = fdir[r, c]
        if k < 0:
            continue
        r2, c2 = r + int(d8.DR[k]), c + int(d8.DC[k])
        if 0 <= r2 < h and 0 <= c2 < w and stream[r2, c2]:
            incoming[r2, c2] += 1
    seen = np.zeros(stream.shape, bool)
    paths = []
    for r, c in zip(*np.nonzero(stream & (incoming == 0)), strict=True):
        path = []
        r, c = int(r), int(c)
        while 0 <= r < h and 0 <= c < w and stream[r, c] and not seen[r, c]:
            seen[r, c] = True
            path.append((r, c))
            k = fdir[r, c]
            if k < 0:
                break
            r, c = r + int(d8.DR[k]), c + int(d8.DC[k])
        if len(path) > 6:
            paths.append(path)
    return paths


def _run(mask: np.ndarray, centre: int, reach: int) -> tuple[int, int] | None:
    """[start, end] indices of the True run nearest to `centre` (within `reach` samples)."""
    for d in range(reach + 1):
        for i in (centre - d, centre + d):
            if 0 <= i < len(mask) and mask[i]:
                a = b = i
                while a > 0 and mask[a - 1]:
                    a -= 1
                while b < len(mask) - 1 and mask[b + 1]:
                    b += 1
                return a, b
    return None


def _polys(mask: np.ndarray, tf, kind: str, min_px: int = 4) -> list[dict]:
    out = []
    for geom, v in shapes(mask.astype(np.uint8), mask=mask, transform=tf):
        g = shape(geom)
        if v and g.area >= min_px * abs(tf.a * tf.e):
            out.append({"kind": kind, "geometry": g})
    return out


def measure(
    pre_s2: Path,
    post_s2: Path,
    terrain_dir: Path,
    in_aoi: np.ndarray,
    before_day: str,
    after_day: str,
) -> tuple[ChangeSummary, gpd.GeoDataFrame]:
    """Measure change between two cached S2 scenes on the job grid (local UTM).

    `in_aoi` is a bool mask on the S2 grid. Returns the summary and a GeoDataFrame (job CRS) with
    `kind` in {water_before, water_after, width_before, width_after, station}."""
    pre, tf = _read_s2(pre_s2)
    post, _ = _read_s2(post_s2)
    with rasterio.open(pre_s2) as ds:
        crs = ds.crs
    ib, ia = indices(pre), indices(post)
    ok = clear(pre) & clear(post) & in_aoi
    water_b = ok & (ib["mndwi"] > WATER_MNDWI)
    water_a = ok & (ia["mndwi"] > WATER_MNDWI)
    chan_b = water_b | (ok & (ib["ndvi"] < BARE_NDVI))
    chan_a = water_a | (ok & (ia["ndvi"] < BARE_NDVI))
    px_km2 = abs(tf.a * tf.e) / 1e6

    with rasterio.open(terrain_dir / "terrain_acc.tif") as ds:
        acc, ttf = ds.read(1), ds.transform
    with rasterio.open(terrain_dir / "terrain_fdir.tif") as ds:
        fdir = ds.read(1).astype(np.int32)
    cell = abs(ttf.a)
    h, w = ok.shape
    offs = np.arange(-TRANSECT_HALF_M, TRANSECT_HALF_M + SAMPLE_M / 2, SAMPLE_M)
    mid = len(offs) // 2
    reach = int(SEARCH_M / SAMPLE_M)
    rows: list[dict] = []
    every = max(1, int(round(STATION_SPACING_M / cell)))
    for path in _river_paths(acc, fdir, cell):
        rc = np.array(path, float) + 0.5  # north-up grids: no rotation terms
        xy = np.column_stack([ttf.c + rc[:, 1] * ttf.a, ttf.f + rc[:, 0] * ttf.e])
        for i in range(3, len(path) - 3, every):
            t = xy[i + 3] - xy[i - 3]
            n = np.array([-t[1], t[0]]) / max(float(np.hypot(*t)), 1e-6)
            pts = xy[i] + offs[:, None] * n
            col, row = (pts[:, 0] - tf.c) / tf.a, (pts[:, 1] - tf.f) / tf.e
            ci, ri = np.floor(col).astype(int), np.floor(row).astype(int)
            inside = (ri >= 0) & (ri < h) & (ci >= 0) & (ci < w)
            if not inside.all():
                continue
            okl = ok[ri, ci]
            rb, ra = _run(chan_b[ri, ci], mid, reach), _run(chan_a[ri, ci], mid, reach)
            if rb is None or ra is None:
                continue
            lo, hi = min(rb[0], ra[0]), max(rb[1], ra[1])
            if lo == 0 or hi == len(offs) - 1 or not okl[max(lo - 2, 0) : hi + 3].all():
                continue
            wb = (rb[1] - rb[0] + 1) * SAMPLE_M
            wa = (ra[1] - ra[0] + 1) * SAMPLE_M
            shift = abs((ra[0] + ra[1]) / 2 - (rb[0] + rb[1]) / 2) * SAMPLE_M
            props = {
                "width_before_m": wb,
                "width_after_m": wa,
                "width_change_m": wa - wb,
                "shift_m": round(shift),
            }
            rows.append(
                {"kind": "width_before", **props, "geometry": LineString(pts[[rb[0], rb[1]]])}
            )
            rows.append(
                {"kind": "width_after", **props, "geometry": LineString(pts[[ra[0], ra[1]]])}
            )
            rows.append({"kind": "station", **props, "geometry": Point(pts[hi + 3])})

    feats = rows + _polys(water_b, tf, "water_before") + _polys(water_a, tf, "water_after")
    gdf = (
        gpd.GeoDataFrame(feats, geometry="geometry", crs=crs)
        if feats
        else gpd.GeoDataFrame({"kind": []}, geometry=[], crs=crs)
    )
    st = gdf[gdf["kind"] == "station"] if len(gdf) else gdf
    wb = st["width_before_m"].to_numpy(float) if len(st) else np.array([])
    wa = st["width_after_m"].to_numpy(float) if len(st) else np.array([])
    sh = st["shift_m"].to_numpy(float) if len(st) else np.array([])
    gain = wa - wb
    area_aoi = float(in_aoi.sum()) * px_km2
    summary = ChangeSummary(
        before_day=before_day,
        after_day=after_day,
        clear_both_km2=round(float(ok.sum()) * px_km2, 2),
        clear_both_pct=round(100 * float(ok.sum()) * px_km2 / max(area_aoi, 1e-9), 1),
        water_before_km2=round(float(water_b.sum()) * px_km2, 2),
        water_after_km2=round(float(water_a.sum()) * px_km2, 2),
        channel_before_km2=round(float(chan_b.sum()) * px_km2, 2),
        channel_after_km2=round(float(chan_a.sum()) * px_km2, 2),
        stations=len(st),
        width_before_median_m=round(float(np.median(wb))) if len(wb) else None,
        width_after_median_m=round(float(np.median(wa))) if len(wa) else None,
        widened_stations=int((gain >= WIDENED_M).sum()),
        width_gain_max_m=round(float(gain.max())) if len(gain) else None,
        shift_median_m=round(float(np.median(sh))) if len(sh) else None,
        shift_max_m=round(float(sh.max())) if len(sh) else None,
    )
    return summary, gdf
