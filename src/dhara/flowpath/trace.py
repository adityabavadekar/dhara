"""Bonus: trace a flood path downstream from any upstream point and list the settlements along it.

The clicked point is snapped to the strongest drainage cell within `snap_m`, then followed
by D8 steepest descent on the conditioned Copernicus DEM (30 m, local UTM). It stops at a sink,
the edge of the domain, or `length_km` along the path. Settlements (pre-event OSM `place=*`)
within `corridor_m` of the path are listed in downstream order with their distance along the path.

No arrival times are given: travel speed depends on the flow volume and type (water vs debris),
which the system does not observe.
"""

import json
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from shapely import LineString

from dhara.adapters.cdse import Grid
from dhara.adapters.dem import fetch_dem
from dhara.adapters.ohsome import fetch_layer
from dhara.config import OSM_SNAPSHOT_MAX, get_settings
from dhara.network.graph import SETTLEMENTS, _tags
from dhara.terrain import hydro


@dataclass
class FlowPath:
    path: gpd.GeoDataFrame  # one LineString, EPSG:4326
    corridor: gpd.GeoDataFrame  # buffered path, EPSG:4326
    settlements: gpd.GeoDataFrame  # ordered downstream, EPSG:4326
    meta: dict


def _utm(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


_ARRAYS = ("dem", "dem_cond", "fdir", "acc", "channels", "hand", "slope_deg")


def _save_npz(t: hydro.Terrain, tdir: Path) -> None:
    """Raw arrays, uncompressed: a few hundred ms to load, versus seconds for COGs."""
    tmp = tdir / "terrain.part.npz"
    np.savez(tmp, **{n: getattr(t, n) for n in _ARRAYS})
    tmp.rename(tdir / "terrain.npz")
    (tdir / "bounds.json").write_text(json.dumps(list(t.grid.bounds)))


def _load_npz(tdir: Path, grid: Grid) -> hydro.Terrain | None:
    f = tdir / "terrain.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        return hydro.Terrain(grid=grid, **{n: z[n] for n in _ARRAYS})


def _terrain(lon: float, lat: float, half_km: float) -> hydro.Terrain:
    """Terrain (local UTM) covering a square of `half_km` around the point.

    Reuses any cached flow-path terrain in the same UTM zone that already covers the square, so
    a second trace in the same valley skips the DEM read and the drainage computation. A new
    terrain is written to the cache in the background so the trace does not wait for it.
    """
    crs = _utm(lon, lat)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    h = half_km * 1000
    want = (x - h, y - h, x + h, y + h)
    cache = get_settings().cache_dir
    epsg = crs.to_epsg()
    for meta in sorted(cache.glob(f"flowpath_{epsg}_*/bounds.json")):
        b = json.loads(meta.read_text())
        if b[0] <= want[0] and b[1] <= want[1] and b[2] >= want[2] and b[3] >= want[3]:
            t = _load_npz(meta.parent, Grid.snapped(crs, tuple(b), hydro.TERRAIN_RES_M))
            if t is not None:
                return t
    # A 5 km margin lets the next click a few km away reuse this terrain.
    m = 5000.0
    grid = Grid.snapped(
        crs, (want[0] - m, want[1] - m, want[2] + m, want[3] + m), hydro.TERRAIN_RES_M
    )
    tdir = cache / f"flowpath_{epsg}_{int(grid.bounds[0])}_{int(grid.bounds[1])}_{int(h)}"
    tdir.mkdir(parents=True, exist_ok=True)
    dem, _ = fetch_dem(grid)
    t = hydro.compute(dem, grid)
    threading.Thread(target=_save_npz, args=(t, tdir), daemon=True).start()
    return t


def _smooth(line: LineString, rounds: int = 2) -> LineString:
    """Chaikin corner cutting for display: removes the 45°/90° staircase of a D8 path on a 30 m
    grid. Distances, times and settlement order are computed on the raw path, not this one."""
    pts = np.asarray(line.coords)
    for _ in range(rounds):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        pts = np.vstack([pts[:1], np.column_stack([q, r]).reshape(-1, 2), pts[-1:]])
    return LineString(pts)


def trace(
    lon: float,
    lat: float,
    *,
    length_km: float = 60.0,
    corridor_m: float = 600.0,
    snap_m: float = 300.0,
    snapshot: date = OSM_SNAPSHOT_MAX,
) -> FlowPath:
    """Trace downstream flowpath and debris surge wave kinematics.

    Analysis runs in local UTM; inputs and returned geodataframes are in EPSG:4326.
    """
    t = _terrain(lon, lat, half_km=0.75 * length_km)  # paths are rarely straight
    crs = t.grid.crs
    to_utm = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = to_utm.transform(lon, lat)
    # Start where the user clicked: water runs downhill from there. Only if that cell has no
    # outflow (a flat or a sink in the conditioned DEM) snap to the strongest drainage nearby.
    c, r = t.grid.to_pixel(x, y)
    r, c = int(r), int(c)
    sx, sy = t.grid.to_world(c + 0.5, r + 0.5)
    pts = hydro.trace_flowpath(t, float(sx), float(sy))
    if len(pts) < 3:
        k = int(snap_m // t.grid.res)
        r0, r1 = max(r - k, 0), min(r + k + 1, t.acc.shape[0])
        c0, c1 = max(c - k, 0), min(c + k + 1, t.acc.shape[1])
        win = t.acc[r0:r1, c0:c1]
        rr, cc = np.unravel_index(int(np.nanargmax(win)), win.shape)
        sx, sy = t.grid.to_world(cc + c0 + 0.5, rr + r0 + 0.5)
        pts = hydro.trace_flowpath(t, float(sx), float(sy))
    seg = np.hypot(*np.diff(pts, axis=0).T) if len(pts) > 1 else np.zeros(0)
    cum = np.r_[0.0, np.cumsum(seg)]
    keep = cum <= length_km * 1000
    pts, cum = pts[keep], cum[keep]
    if len(pts) < 2:
        raise ValueError("Flow path is shorter than one cell: the point is in a sink or flat area")
    line = LineString(pts)
    corridor = line.buffer(corridor_m)

    # Sample river bed elevation along the D8 path nodes (local UTM)
    pix_coords = [t.grid.to_pixel(p[0], p[1]) for p in pts]
    rows_c = np.clip([int(r) for _, r in pix_coords], 0, t.dem.shape[0] - 1)
    cols_c = np.clip([int(c) for c, _ in pix_coords], 0, t.dem.shape[1] - 1)
    node_elevs = t.dem_cond[rows_c, cols_c].astype(float)

    # Rough arrival-time estimate: v(s) = clip(k * slope^0.35, v_min, v_max) in m/s. k, the exponent
    # and the bounds are assumed, not calibrated against any observed event, so arrival times are
    # an order-of-magnitude guide only. The UI and sitrep label them as such.
    seg_len = np.maximum(np.diff(cum), 1.0)
    drop_per_seg = np.maximum(node_elevs[:-1] - node_elevs[1:], 0.0)
    slope = np.maximum(drop_per_seg / seg_len, 0.0001)
    seg_speeds = np.clip(42.0 * (slope**0.35), 6.0, 50.0)  # m/s
    seg_times_sec = seg_len / seg_speeds
    node_times_sec = np.r_[0.0, np.cumsum(seg_times_sec)]
    node_speeds_kmh = np.r_[seg_speeds[0], seg_speeds] * 3.6
    node_times_min = node_times_sec / 60.0

    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    bx = corridor.bounds
    lo0, la0 = to_ll.transform(bx[0], bx[1])
    lo1, la1 = to_ll.transform(bx[2], bx[3])
    places = fetch_layer("places", (lo0, la0, lo1, la1), min(snapshot, OSM_SNAPSHOT_MAX))
    rows = []
    if len(places):
        pu = places.to_crs(crs)
        for geom, tags, oid in zip(pu.geometry, pu["tags"], pu["osm_id"], strict=True):
            tg = _tags(tags)
            p = geom.representative_point()
            if tg.get("place") not in SETTLEMENTS or not corridor.contains(p):
                continue
            d = line.project(p)
            ci, ri = t.grid.to_pixel(p.x, p.y)
            ri, ci = int(ri), int(ci)
            elev_m = float(t.dem[ri, ci])
            hand_m = float(t.hand[ri, ci])
            arr_min = float(np.interp(d, cum, node_times_min))
            local_spd = float(np.interp(d, cum, node_speeds_kmh))
            riv_z = float(np.interp(d, cum, node_elevs))
            delta_z = elev_m - riv_z

            # Classify exposure: distinguish direct gorge flooding from perched ridge settlements
            if delta_z <= 15.0 or hand_m <= 12.0:
                exposure = "direct_path"
            elif delta_z <= 30.0:
                exposure = "fringe"
            else:
                exposure = "ridge_safe"

            rows.append(
                {
                    "name": tg.get("name") or tg.get("name:en") or "",
                    "place": tg.get("place"),
                    "osm_id": oid,
                    "distance_km": round(d / 1000, 2),
                    "offset_m": round(line.distance(p), 0),
                    "elevation_m": round(elev_m, 0),
                    "river_elev_m": round(riv_z, 0),
                    "delta_z_river_m": round(delta_z, 1),
                    "hand_m": round(hand_m, 1),
                    "arrival_time_min": round(arr_min, 1),
                    "evacuation_window": f"+{int(round(arr_min))}m",
                    "wave_speed_kmh": round(local_spd, 1),
                    "exposure_status": exposure,
                    "geometry": p,
                }
            )
    sdf = (
        gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        if rows
        else gpd.GeoDataFrame({"name": []}, geometry=[], crs=crs)
    )
    if len(sdf):
        sdf = sdf.sort_values("distance_km").reset_index(drop=True)
        sdf.insert(0, "order", np.arange(1, len(sdf) + 1))
    cs, rs = t.grid.to_pixel(*pts[0])
    z0 = float(t.dem_cond[min(int(rs), t.dem.shape[0] - 1), min(int(cs), t.dem.shape[1] - 1)])
    ce, re_ = t.grid.to_pixel(*pts[-1])
    z1 = float(t.dem_cond[min(int(re_), t.dem.shape[0] - 1), min(int(ce), t.dem.shape[1] - 1)])
    node_slopes_pct = np.r_[slope[0], slope] * 100.0
    to_ll_nodes = Transformer.from_crs(crs, 4326, always_xy=True)
    step_skip = max(1, len(pts) // 250)
    profile_indices = list(range(0, len(pts), step_skip))
    if profile_indices and profile_indices[-1] != len(pts) - 1:
        profile_indices.append(len(pts) - 1)

    prof_points = []
    for idx in profile_indices:
        px, py = pts[idx]
        plng, plat = to_ll_nodes.transform(px, py)
        prof_points.append(
            {
                "dist_km": round(float(cum[idx]) / 1000.0, 3),
                "elevation_m": round(float(node_elevs[idx]), 1),
                "lng": round(float(plng), 5),
                "lat": round(float(plat), 5),
                "slope_pct": round(float(node_slopes_pct[idx]), 1),
            }
        )

    meta = {
        "start_clicked": [lon, lat],
        "start_snapped": list(to_ll.transform(float(sx), float(sy))),
        "length_km": round(float(cum[-1]) / 1000, 2),
        "drop_m": round(z0 - z1, 1),
        "corridor_m": corridor_m,
        "stopped_because": "length limit" if not keep.all() else "domain edge or sink",
        "settlements": int(len(sdf)),
        "osm_snapshot": f"{min(snapshot, OSM_SNAPSHOT_MAX).isoformat()}T00:00:00Z",
        "dem": "Copernicus DEM GLO-30",
        "simulation": {
            "model": "Kinematic slope-dependent Himalayan debris surge wave propagation",
            "peak_speed_kmh": round(float(np.max(node_speeds_kmh)), 1),
            "avg_speed_kmh": round(float(np.mean(node_speeds_kmh)), 1),
            "total_transit_time_min": round(float(node_times_min[-1]), 1),
        },
        "profile": {
            "points": prof_points,
            "summary": {
                "total_distance_km": round(float(cum[-1]) / 1000.0, 2),
                "min_elevation_m": round(float(np.min(node_elevs)), 1),
                "max_elevation_m": round(float(np.max(node_elevs)), 1),
                "elevation_gain_m": round(float(np.sum(np.maximum(0, np.diff(node_elevs)))), 1),
                "elevation_loss_m": round(float(np.sum(np.maximum(0, -np.diff(node_elevs)))), 1),
                "max_slope_pct": round(float(np.max(node_slopes_pct)), 1)
                if len(node_slopes_pct)
                else 0.0,
                "avg_slope_pct": round(float(np.mean(node_slopes_pct)), 1)
                if len(node_slopes_pct)
                else 0.0,
            },
        },
    }
    return FlowPath(
        path=gpd.GeoDataFrame(
            {
                "length_km": [meta["length_km"]],
                "transit_time_min": [meta["simulation"]["total_transit_time_min"]],
                "avg_speed_kmh": [meta["simulation"]["avg_speed_kmh"]],
            },
            geometry=[_smooth(line)],
            crs=crs,
        ).to_crs(4326),
        corridor=gpd.GeoDataFrame(
            {"corridor_m": [corridor_m]}, geometry=[corridor], crs=crs
        ).to_crs(4326),
        settlements=sdf.to_crs(4326),
        meta=meta,
    )


def save(fp: FlowPath, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fp.path.to_file(out / "flowpath.geojson", driver="GeoJSON")
    fp.corridor.to_file(out / "corridor.geojson", driver="GeoJSON")
    (out / "settlements.geojson").write_text(fp.settlements.to_json(drop_id=True))
    (out / "meta.json").write_text(json.dumps(fp.meta, indent=1))
    if "profile" in fp.meta:
        (out / "profile.json").write_text(json.dumps(fp.meta["profile"], indent=1))
