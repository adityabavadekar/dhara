"""Stage 5: what was hit. Per pre-event OSM feature: hazard sampling, confidence tier, bridge check.

At 10 m we cannot grade individual houses, so the product is "feature intersected by flood/debris
evidence" with a confidence tier, not "destroyed vs damaged".
"""

import json

import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from rasterio.transform import Affine, rowcol

from dhara.adapters.cdse import Grid
from dhara.osm_names import english_name

HIT_P = 0.5
TIER_NAMES = {0: "none", 1: "low", 2: "moderate", 3: "high"}


def _windows(geoms, grid: Grid):
    """Yield (i, r0, r1, c0, c1, inside-mask) for each geometry on `grid` (all_touched)."""
    H, W = grid.height, grid.width
    tf: Affine = grid.transform
    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        r0, c0 = rowcol(tf, minx, maxy)
        r1, c1 = rowcol(tf, maxx, miny)
        r0, c0 = max(int(r0), 0), max(int(c0), 0)
        r1, c1 = min(int(r1) + 1, H), min(int(c1) + 1, W)
        if r1 <= r0 or c1 <= c0:
            continue
        win_tf = tf @ Affine.translation(c0, r0)
        inside = ~geometry_mask([geom], (r1 - r0, c1 - c0), win_tf, all_touched=True)
        yield i, r0, r1, c0, c1, inside


def _features_px(
    geoms: gpd.GeoSeries, grid: Grid, line_step_m: float, max_window_px: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Pixels touched by each feature, vectorised.

    Returns (feature_index, row, col) for all touched pixels, plus the indices of polygons
    larger than `max_window_px`, which need an exact per-feature mask.
    - Lines: densified every `line_step_m`, so each sample hits one pixel.
    - Small polygons (most buildings at 10 m): every pixel overlapping the bounding box.
    """
    import shapely

    arr = geoms.to_numpy()
    kind = shapely.get_type_id(arr)  # 1 LineString, 3 Polygon, 5 MultiLineString, 6 MultiPolygon
    is_line = np.isin(kind, (1, 5))
    fi, rr, cc = [], [], []

    if is_line.any():
        li = np.flatnonzero(is_line)
        dens = shapely.segmentize(arr[li], line_step_m)
        xy, idx = shapely.get_coordinates(dens, return_index=True)
        col, row = grid.to_pixel(xy[:, 0], xy[:, 1])
        fi.append(li[idx])
        rr.append(np.floor(row).astype(int))
        cc.append(np.floor(col).astype(int))

    big: list[int] = []
    pi = np.flatnonzero(~is_line & ~shapely.is_empty(arr))
    if len(pi):
        b = shapely.bounds(arr[pi])
        c0, r0 = grid.to_pixel(b[:, 0], b[:, 3])
        c1, r1 = grid.to_pixel(b[:, 2], b[:, 1])
        c0, r0 = np.floor(c0).astype(int), np.floor(r0).astype(int)
        c1, r1 = np.floor(c1).astype(int), np.floor(r1).astype(int)
        small = ((r1 - r0) < max_window_px) & ((c1 - c0) < max_window_px)
        big = pi[~small].tolist()
        si = np.flatnonzero(small)
        for dr in range(max_window_px):
            for dc in range(max_window_px):
                ok = (r0[si] + dr <= r1[si]) & (c0[si] + dc <= c1[si])
                fi.append(pi[si[ok]])
                rr.append(r0[si[ok]] + dr)
                cc.append(c0[si[ok]] + dc)
    if not fi:
        z = np.zeros(0, int)
        return z, z, z, big
    f, r, c = np.concatenate(fi), np.concatenate(rr), np.concatenate(cc)
    inside = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
    return f[inside], r[inside], c[inside], big


def sample_features(
    gdf: gpd.GeoDataFrame,
    prob: np.ndarray,
    grid: Grid,
    *,
    p_flood: np.ndarray | None = None,
    p_debris: np.ndarray | None = None,
    tier: np.ndarray | None = None,
    line_buffer_m: float = 5.0,
) -> gpd.GeoDataFrame:
    """Hazard statistics per feature. Output CRS: grid.crs.

    prob: combined P(flood or debris) on `grid`, NaN = not observable. Columns: n_px,
    n_observable, frac_hit, p_max, hazard (flood|debris), confidence (none|low|moderate|high),
    status (outside|not_observable|hit|not_hit). Lines are sampled every `line_buffer_m`.
    """
    g = gdf.to_crs(grid.crs).copy()
    n = len(g)
    f, r, c, big = _features_px(g.geometry, grid, line_buffer_m)
    # exact masks for large polygons
    if big:
        extra = list(_windows(g.geometry.iloc[big], grid))
        for j, r0, _r1, c0, _c1, inside in extra:
            rr, cc = np.nonzero(inside)
            f = np.r_[f, np.full(len(rr), big[j])]
            r = np.r_[r, rr + r0]
            c = np.r_[c, cc + c0]
    v = prob[r, c]
    ok = np.isfinite(v)
    n_px = np.bincount(f, minlength=n).astype(np.int32)
    n_obs = np.bincount(f[ok], minlength=n).astype(np.int32)
    hits = np.bincount(f[ok & (v >= HIT_P)], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_obs > 0, hits / np.maximum(n_obs, 1), np.nan).astype(np.float32)
    pmax = np.full(n, -1.0)
    np.maximum.at(pmax, f[ok], v[ok])
    pmax = np.where(n_obs > 0, pmax, np.nan).astype(np.float32)
    hazard = np.full(n, "", object)
    conf = np.zeros(n, np.uint8)
    if p_flood is not None and p_debris is not None:
        fmax = np.zeros(n)
        dmax = np.zeros(n)
        np.maximum.at(fmax, f, np.nan_to_num(p_flood[r, c]))
        np.maximum.at(dmax, f, np.nan_to_num(p_debris[r, c]))
        is_hit = np.nan_to_num(pmax) >= HIT_P
        hazard[is_hit & (fmax >= HIT_P)] = "flood"
        hazard[is_hit & (fmax < HIT_P) & (dmax >= HIT_P)] = "debris"
    if tier is not None:
        np.maximum.at(conf, f, tier[r, c].astype(np.uint8))
        conf[~(np.nan_to_num(pmax) >= HIT_P)] = 0
    g["n_px"], g["n_observable"], g["frac_hit"], g["p_max"] = n_px, n_obs, frac, pmax
    g["hazard"] = hazard
    g["confidence"] = [TIER_NAMES[int(x)] for x in conf]
    g["status"] = np.where(
        n_px == 0,
        "outside",
        np.where(n_obs == 0, "not_observable", np.where(pmax >= HIT_P, "hit", "not_hit")),
    )

    if "tags" in g.columns:
        facility_types: list[str] = []
        names: list[str | None] = []
        for t_val in g["tags"]:
            amenity = tag(str(t_val), "amenity")
            healthcare = tag(str(t_val), "healthcare")
            bldg = tag(str(t_val), "building")
            try:
                nm = english_name(json.loads(str(t_val))) or None
            except (TypeError, ValueError):
                nm = None
            if amenity in ("school", "kindergarten", "college", "university"):
                ft = "school"
            elif amenity in ("hospital", "clinic", "doctors", "pharmacy") or healthcare is not None:
                ft = "health"
            elif amenity in ("place_of_worship", "community_centre", "townhall"):
                ft = "community"
            elif bldg and bldg in ("house", "residential", "apartments"):
                ft = "residential"
            elif bldg and bldg == "commercial":
                ft = "commercial"
            else:
                ft = "building"
            facility_types.append(ft)
            names.append(nm)
        g["facility_type"] = facility_types
        g["name"] = names

    return g


def tag(tags_json: str, key: str) -> str | None:
    try:
        return json.loads(tags_json).get(key)
    except (TypeError, ValueError):
        return None


def is_bridge(tags_json: str) -> bool:
    v = tag(tags_json, "bridge")
    return v is not None and v != "no"


def bridge_check(
    bridges: gpd.GeoDataFrame,
    grid: Grid,
    prob: np.ndarray,
    vv_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    buffer_m: float = 20.0,
) -> gpd.GeoDataFrame:
    """Dedicated bridge test. bridges: road ways tagged bridge=*, any CRS; output in grid.crs.

    Two independent clues:
    1. Hazard at both abutments: P(flood or debris) within `buffer_m` of each bridge end, taking
       the weaker end (both banks must be hit). The span itself is not used, because the river
       channel under any bridge changes in a flood whether or not the bridge survives.
    2. Loss of the bridge's bright radar return: a standing bridge over a channel is a strong,
       stable scatterer (double bounce). We compare the brightest pixels (p90 of γ⁰ VV, dB) inside
       the buffer before and after, on each track. A drop of 3 dB or more counts as evidence
       of loss.
    P(lost) = 1 − (1 − 0.85·p_bright_loss)(1 − 0.45·p_hazard). Hazard alone cannot make a
    bridge "likely lost" (max 0.45): at 10 m the abutment buffers overlap the changed channel
    even for bridges still standing. Statuses: likely_lost (P ≥ 0.5), at_risk (hazard at both
    ends but no radar evidence of loss), likely_standing, not_observable. Bridges shorter than
    2 pixels are flagged as below resolution, and their confidence is capped at low.
    """
    import shapely

    b = bridges.to_crs(grid.crs).copy()
    geoms = b.geometry.buffer(buffer_m)
    n = len(b)
    p_haz = np.full(n, np.nan, np.float32)
    drop = np.full(n, np.nan, np.float32)
    starts = shapely.get_point(shapely.line_merge(b.geometry.to_numpy()), 0).tolist()
    ends = shapely.get_point(shapely.line_merge(b.geometry.to_numpy()), -1).tolist()
    for which in (starts, ends):
        ab = gpd.GeoSeries(which, crs=b.crs).buffer(buffer_m)
        end_p = np.full(n, np.nan, np.float32)
        for i, r0, _r1, c0, _c1, inside in _windows(ab, grid):
            v = prob[r0:_r1, c0:_c1][inside]
            if np.isfinite(v).any():
                end_p[i] = float(np.nanmax(v))
        p_haz = np.where(np.isnan(p_haz), end_p, np.fmin(p_haz, end_p))
    for i, r0, r1, c0, c1, inside in _windows(geoms, grid):
        drops = []
        for pre_vv, post_vv in vv_pairs:
            a = pre_vv[r0:r1, c0:c1][inside]
            c = post_vv[r0:r1, c0:c1][inside]
            ok = np.isfinite(a) & np.isfinite(c) & (a > 0) & (c > 0)
            if ok.sum() >= 3:
                drops.append(
                    10 * np.log10(np.percentile(a[ok], 90))
                    - 10 * np.log10(np.percentile(c[ok], 90))
                )
        if drops:
            drop[i] = float(np.max(drops))
    p_bright = 1.0 / (1.0 + np.exp(-(np.nan_to_num(drop, nan=-10.0) - 3.0) / 1.0))
    p_lost = 1.0 - (1.0 - 0.85 * p_bright) * (1.0 - 0.45 * np.nan_to_num(p_haz))
    length = b.geometry.length.to_numpy()
    observed = np.isfinite(p_haz) | np.isfinite(drop)
    strong = (p_lost >= 0.8) & (np.nan_to_num(p_haz) >= 0.5) & (p_bright >= 0.5)
    conf = np.where(
        ~observed,
        "none",
        np.where(
            length < 2 * grid.res,
            "low",
            np.where(strong, "high", np.where(p_lost >= 0.65, "moderate", "low")),
        ),
    )
    b["length_m"] = length.round(1)
    b["p_hazard"] = p_haz
    b["bright_drop_db"] = drop
    b["p_lost"] = np.where(observed, p_lost, np.nan).astype(np.float32)
    b["status"] = np.where(
        ~observed,
        "not_observable",
        np.where(
            p_lost >= 0.5,
            "likely_lost",
            np.where(np.nan_to_num(p_haz) >= 0.5, "at_risk", "likely_standing"),
        ),
    )
    b["confidence"] = conf
    b["below_resolution"] = length < 2 * grid.res
    return b
