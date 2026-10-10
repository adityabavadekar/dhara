"""Bonus: trace a flood path downstream from any upstream point and list the settlements along it.

The clicked point is snapped to the strongest drainage cell within `snap_m`, then followed
by D8 steepest descent on the conditioned Copernicus DEM (30 m, local UTM). It stops at a sink,
the edge of the domain, or `length_km` along the path. Settlements (pre-event OSM `place=*`)
within `corridor_m` of the path are listed in downstream order with their distance along the path.

No arrival times are given: travel speed depends on the flow volume and type (water vs debris),
which the system does not observe.
"""

import json
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


def _terrain(lon: float, lat: float, half_km: float) -> hydro.Terrain:
    crs = _utm(lon, lat)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    h = half_km * 1000
    grid = Grid.snapped(crs, (x - h, y - h, x + h, y + h), hydro.TERRAIN_RES_M)
    key = f"flowpath_{crs.to_epsg()}_{int(grid.bounds[0])}_{int(grid.bounds[1])}_{int(h)}"
    tdir = get_settings().cache_dir / key
    tdir.mkdir(parents=True, exist_ok=True)
    t = hydro.load(tdir, grid)
    if t is None:
        dem, _ = fetch_dem(grid)
        t = hydro.compute(dem, grid)
        hydro.save(t, tdir)
    return t


def trace(
    lon: float,
    lat: float,
    *,
    length_km: float = 60.0,
    corridor_m: float = 600.0,
    snap_m: float = 300.0,
    snapshot: date = OSM_SNAPSHOT_MAX,
) -> FlowPath:
    """lon/lat in EPSG:4326. Returns path, corridor and ordered settlements (EPSG:4326)."""
    t = _terrain(lon, lat, half_km=0.75 * length_km)  # paths are rarely straight
    crs = t.grid.crs
    to_utm = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = to_utm.transform(lon, lat)
    # snap to the strongest drainage cell nearby
    c, r = t.grid.to_pixel(x, y)
    r, c = int(r), int(c)
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
            rows.append(
                {
                    "name": tg.get("name") or tg.get("name:en") or "",
                    "place": tg.get("place"),
                    "osm_id": oid,
                    "distance_km": round(d / 1000, 2),
                    "offset_m": round(line.distance(p), 0),
                    "elevation_m": round(float(t.dem[ri, ci]), 0),
                    "hand_m": round(float(t.hand[ri, ci]), 1),
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
    z0 = float(t.dem_cond[int(rr + r0), int(cc + c0)])
    ce, re_ = t.grid.to_pixel(*pts[-1])
    z1 = float(t.dem_cond[min(int(re_), t.dem.shape[0] - 1), min(int(ce), t.dem.shape[1] - 1)])
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
    }
    return FlowPath(
        path=gpd.GeoDataFrame({"length_km": [meta["length_km"]]}, geometry=[line], crs=crs).to_crs(
            4326
        ),
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
