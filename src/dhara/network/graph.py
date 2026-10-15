"""Stage 6: who is cut off. Reachability on the pre-event OSM road + trail network.

All geometry in the analysis CRS (local UTM, metres). Steps:
1. Graph: every OSM `highway=*` way (motor roads, tracks, paths, footways, steps; isolation in
   the Himalaya is often trail-level). Each consecutive vertex pair is an edge; ways meet at shared
   OSM nodes. Bridges keep their identity as their own edges.
2. Edge failure: max P(flood or debris) sampled every 5 m along the edge; bridge edges also take
   the bridge check's P(lost). Unobserved edges (radar shadow, outside the AOI) are kept open in
   the deterministic run and flagged.
3. Targets: hospitals, health facilities, towns, and "exits" (where the analysed network leaves
   the region; we cannot see beyond it, so reaching an exit counts as reaching the outside world).
4. Per settlement (`place=*`): pre/post shortest path to the nearest hospital and to any target,
   status, extra distance, the failed edges on the pre-event route, building count (exposure
   proxy; population rasters are not allowed inputs).
5. Repair ranking: restore one failed way at a time and count settlements and buildings that
   regain access; ranked by buildings reconnected (bridges are also ranked separately).
6. Monte Carlo: ways fail independently with their probability (the segments of one way fail
   together); P(cut off) = share of draws in which a settlement loses access to all targets.
"""

import json
from dataclasses import dataclass, field
from typing import TypedDict

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import KDTree
from shapely import LineString, Point

from dhara.adapters.cdse import Grid
from dhara.osm_names import english_name, nepali_name

HOSPITAL = {"hospital"}
HEALTH = {"hospital", "clinic", "doctors", "health_post", "healthcare"}
TOWNS = {"town", "city"}
SETTLEMENTS = {
    "city",
    "town",
    "village",
    "hamlet",
    "isolated_dwelling",
    "locality",
    "neighbourhood",
}


@dataclass
class Network:
    nodes: np.ndarray  # (N, 2) xy
    u: np.ndarray
    v: np.ndarray
    length: np.ndarray
    way: np.ndarray  # index into ways table
    is_bridge: np.ndarray
    highway: np.ndarray
    ways: gpd.GeoDataFrame
    p_fail: np.ndarray = field(default_factory=lambda: np.zeros(0))
    observed: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    in_region: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    tier: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))

    @property
    def n(self) -> int:
        return len(self.nodes)


def _tags(s: str) -> dict:
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return {}


def build(roads: gpd.GeoDataFrame, crs) -> Network:
    """roads: OSM highway ways (any CRS). Nodes are vertices rounded to 0.5 m in `crs`.

    Vectorised: ways meet where they share a vertex (OSM node), so equal rounded coordinates
    become one graph node.
    """
    import shapely

    r = roads.to_crs(crs).explode(index_parts=False).reset_index(drop=True)
    r = r[r.geom_type == "LineString"].reset_index(drop=True)
    tags = [_tags(t) for t in r["tags"]]
    way_bridge = np.array([t.get("bridge") not in (None, "no") for t in tags], bool)
    way_hw = np.array([t.get("highway", "") for t in tags], object)
    xy, wi = shapely.get_coordinates(r.geometry.to_numpy(), return_index=True)
    if len(xy) == 0:
        z = np.zeros(0, int)
        return Network(
            np.zeros((0, 2)), z, z, np.zeros(0), z, np.zeros(0, bool), np.zeros(0, object), r
        )
    keys = np.round(xy * 2).astype(np.int64)
    uniq, node_of = np.unique(keys, axis=0, return_inverse=True)
    node_of = node_of.ravel()
    nodes = uniq / 2.0
    same_way = wi[1:] == wi[:-1]
    u, v = node_of[:-1][same_way], node_of[1:][same_way]
    seg_len = np.hypot(*(xy[1:] - xy[:-1]).T)[same_way]
    w = wi[:-1][same_way]
    ok = u != v
    return Network(
        nodes=nodes,
        u=u[ok],
        v=v[ok],
        length=seg_len[ok],
        way=w[ok],
        is_bridge=way_bridge[w[ok]],
        highway=way_hw[w[ok]],
        ways=r,
    )


def edge_hazard(
    net: Network,
    prob: np.ndarray,
    grid: Grid,
    in_region: np.ndarray | None = None,
    step_m: float = 5.0,
    *,
    tier: np.ndarray | None = None,
    low_tier_weight: float = 0.3,
    min_fail_tier: int = 2,
) -> None:
    """Max probability along each edge (sampled every `step_m`).

    Sets net.p_fail, net.observed (some sample has a finite probability) and net.in_region (some
    sample lies inside the analysed AOI, `in_region` mask on `grid`). Edges outside the region
    are open with no failure risk; edges in the region but unobserved carry the unobserved prior
    in the Monte Carlo.

    With `tier` (fusion confidence, 0–3): an edge whose strongest evidence is below
    `min_fail_tier` (moderate) cannot fail deterministically, and its probability is scaled by
    `low_tier_weight` for the Monte Carlo. Low-confidence pixels are evidence, not proof.
    """
    a, b = net.nodes[net.u], net.nodes[net.v]
    k = np.maximum(2, np.ceil(net.length / step_m).astype(int) + 1)
    seg = np.repeat(np.arange(len(k)), k)
    t = np.concatenate([np.linspace(0, 1, kk) for kk in k]) if len(k) else np.zeros(0)
    pts = a[seg] + (b[seg] - a[seg]) * t[:, None]
    cols, rows = grid.to_pixel(pts[:, 0], pts[:, 1])
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    vals = np.full(len(pts), np.nan, np.float32)
    vals[inside] = prob[rows[inside], cols[inside]]
    p = np.full(len(k), -1.0)
    np.maximum.at(p, seg, np.nan_to_num(vals, nan=-1.0))
    net.observed = p >= 0
    net.p_fail = np.clip(p, 0, 1)
    net.tier = np.full(len(k), 3, np.uint8)
    if tier is not None:
        tv = np.zeros(len(pts), np.uint8)
        tv[inside] = tier[rows[inside], cols[inside]]
        net.tier = np.zeros(len(k), np.uint8)
        np.maximum.at(net.tier, seg, tv)
        weak = net.tier < min_fail_tier
        net.p_fail = np.where(weak, net.p_fail * low_tier_weight, net.p_fail)
    reg = np.zeros(len(pts), bool)
    if in_region is not None:
        reg[inside] = in_region[rows[inside], cols[inside]]
    else:
        reg = inside
    net.in_region = np.zeros(len(k), bool)
    np.logical_or.at(net.in_region, seg, reg)


def apply_bridges(net: Network, bridges: gpd.GeoDataFrame) -> None:
    """Raise bridge edges' failure probability to the bridge check's P(lost) (matched by OSM id)."""
    if len(bridges) == 0:
        return
    lost = dict(zip(bridges["osm_id"], np.nan_to_num(bridges["p_lost"].to_numpy()), strict=True))
    way_ids = net.ways["osm_id"].to_numpy()
    for e in np.flatnonzero(net.is_bridge):
        pl = lost.get(way_ids[net.way[e]])
        if pl is not None:
            net.p_fail[e] = max(net.p_fail[e], pl)
            net.observed[e] = True
            if len(net.tier):
                net.tier[e] = max(int(net.tier[e]), 2 if pl >= 0.5 else 0)


def _matrix(net: Network, keep: np.ndarray, weight: np.ndarray | None = None):
    w = net.length if weight is None else weight
    u, v, ww = net.u[keep], net.v[keep], w[keep]
    m = coo_matrix((np.r_[ww, ww], (np.r_[u, v], np.r_[v, u])), shape=(net.n, net.n)).tocsr()
    return m


def _snap(net: Network, pts: np.ndarray, max_m: float) -> np.ndarray:
    """Nearest graph node per point, or -1 if farther than max_m."""
    if net.n == 0 or len(pts) == 0:
        return np.full(len(pts), -1)
    d, i = KDTree(net.nodes).query(pts)
    return np.where(d <= max_m, i, -1)


def exit_nodes(
    net: Network, bounds: tuple[float, float, float, float], margin_m: float
) -> np.ndarray:
    """Nodes within `margin_m` of the network bbox edge, where the analysed region ends."""
    x, y = net.nodes[:, 0], net.nodes[:, 1]
    minx, miny, maxx, maxy = bounds
    near = (
        (x - minx < margin_m)
        | (maxx - x < margin_m)
        | (y - miny < margin_m)
        | (maxy - y < margin_m)
    )
    return np.flatnonzero(near)


def _points(gdf: gpd.GeoDataFrame, crs) -> np.ndarray:
    g = gdf.to_crs(crs)
    c = g.geometry.representative_point()
    return np.c_[c.x.to_numpy(), c.y.to_numpy()] if len(g) else np.zeros((0, 2))


class Repair(TypedDict):
    osm_id: str
    name: str
    highway: str
    is_bridge: bool
    settlements_reconnected: int
    buildings_reconnected: int
    settlement_names: list[str]


@dataclass
class IsolationResult:
    settlements: gpd.GeoDataFrame
    failed_edges: gpd.GeoDataFrame
    repairs: list[Repair]
    meta: dict
    hospitals: gpd.GeoDataFrame = field(default_factory=lambda: gpd.GeoDataFrame())


def analyse(
    net: Network,
    places: gpd.GeoDataFrame,
    health: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    network_bounds: tuple[float, float, float, float],
    crs,
    *,
    fail_p: float = 0.5,
    snap_m: float = 500.0,
    exposure_radius_m: float = 500.0,
    mc_draws: int = 500,
    p_unobserved: float = 0.1,
    p_min_risk: float = 0.05,
    seed: int = 0,
) -> IsolationResult:
    rng = np.random.default_rng(seed)
    E = len(net.u)
    failed = net.observed & (net.p_fail >= fail_p)
    pre_keep = np.ones(E, bool)
    post_keep = ~failed

    # --- settlements and targets ---------------------------------------------------------
    pl = places.copy()
    pl["place"] = [_tags(t).get("place", "") for t in pl["tags"]]
    pl["name"] = [english_name(_tags(t)) for t in pl["tags"]]
    pl["name_ne"] = [nepali_name(_tags(t)) for t in pl["tags"]]
    pl = pl[pl["place"].isin(SETTLEMENTS)].reset_index(drop=True)
    sp = _points(pl, crs)
    s_node = _snap(net, sp, snap_m)

    ht = health.copy()
    kinds = []
    names = []
    amenities = []
    for t in ht["tags"]:
        tg = _tags(t)
        a, h = tg.get("amenity"), tg.get("healthcare")
        kinds.append("hospital" if "hospital" in (a, h) else "health")
        amenities.append(a or h or "medical")
        names.append(
            english_name(tg) or ("Hospital" if "hospital" in (a, h) else "Health Facility")
        )
    ht["kind"] = kinds
    ht["name"] = names
    ht["amenity"] = amenities
    h_points = _points(ht, crs)
    h_all_nodes = _snap(net, h_points, snap_m)
    hosp_nodes = _snap(net, _points(ht[ht["kind"] == "hospital"], crs), snap_m)
    health_nodes = h_all_nodes
    town_nodes = s_node[(pl["place"].isin(TOWNS)).to_numpy() & (s_node >= 0)]
    exits = exit_nodes(net, network_bounds, 300.0)
    hosp_nodes = np.unique(hosp_nodes[hosp_nodes >= 0])
    health_nodes = np.unique(health_nodes[health_nodes >= 0])
    any_targets = np.unique(np.r_[hosp_nodes, health_nodes, town_nodes, exits]).astype(int)

    hosp_names: dict[int, str] = {}
    hosp_amenities: dict[int, str] = {}
    for idx, nd in enumerate(h_all_nodes):
        if nd >= 0 and nd not in hosp_names:
            hosp_names[nd] = names[idx]
            hosp_amenities[nd] = amenities[idx]

    # --- exposure: buildings near each settlement ----------------------------------------
    bp = _points(buildings, crs) if len(buildings) else np.zeros((0, 2))
    if len(bp) and len(sp):
        counts = np.array([len(x) for x in KDTree(bp).query_ball_point(sp, exposure_radius_m)])
    else:
        counts = np.zeros(len(sp), int)

    # --- shortest paths & travel durations -----------------------------------------------
    # Empirical traversal speeds by OSM highway tag for mountain road networks
    hw_speeds: dict[str, float] = {
        "motorway": 65.0,
        "trunk": 50.0,
        "primary": 40.0,
        "secondary": 30.0,
        "tertiary": 25.0,
        "unclassified": 20.0,
        "residential": 20.0,
        "service": 15.0,
        "track": 10.0,
        "path": 4.0,
        "footway": 4.0,
        "steps": 2.0,
        "pedestrian": 4.0,
    }
    speeds_mps = (
        np.array([hw_speeds.get(str(h), 15.0) / 3.6 for h in net.highway], dtype=float)
        if len(net.highway)
        else np.zeros(0, float)
    )
    edge_durations_hr = (
        (net.length / np.maximum(speeds_mps, 0.5)) / 3600.0
        if len(net.length)
        else np.zeros(0, float)
    )

    def dist(keep, targets):
        if len(targets) == 0 or net.n == 0:
            return np.full(net.n, np.inf), np.full(net.n, -9999), np.full(net.n, -9999)
        d, pred, src = dijkstra(
            _matrix(net, keep),
            directed=False,
            indices=targets,
            min_only=True,
            return_predecessors=True,
        )
        return d, pred, src

    def duration(keep, targets):
        if len(targets) == 0 or net.n == 0:
            return np.full(net.n, np.inf)
        d = dijkstra(
            _matrix(net, keep, weight=edge_durations_hr),
            directed=False,
            indices=targets,
            min_only=True,
        )
        return d

    d_pre_h, pred_pre_h, src_pre_h = dist(pre_keep, hosp_nodes)
    d_post_h, pred_post_h, src_post_h = dist(post_keep, hosp_nodes)
    d_pre_a, pred_pre_a, _ = dist(pre_keep, any_targets)
    d_post_a, _, _ = dist(post_keep, any_targets)

    t_pre_h = duration(pre_keep, hosp_nodes)
    t_post_h = duration(post_keep, hosp_nodes)
    t_pre_a = duration(pre_keep, any_targets)
    t_post_a = duration(post_keep, any_targets)

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def trace_coords(start_node: int, pred: np.ndarray, target_node: int) -> list[list[float]]:
        if start_node < 0 or target_node < 0 or net.n == 0:
            return []
        path = [start_node]
        cur = start_node
        guard = 0
        while cur != target_node and cur >= 0 and pred[cur] >= 0 and guard < 50000:
            cur = int(pred[cur])
            path.append(cur)
            guard += 1
        if cur != target_node or len(path) < 2:
            return []
        xy = net.nodes[path]
        lons, lats = transformer.transform(xy[:, 0], xy[:, 1])
        return [[round(float(x), 6), round(float(y), 6)] for x, y in zip(lons, lats, strict=False)]

    def break_coords(failed_edge_indices: list[int]) -> list[list[float]]:
        pts = []
        for e in failed_edge_indices:
            mx = float((net.nodes[net.u[e], 0] + net.nodes[net.v[e], 0]) / 2)
            my = float((net.nodes[net.u[e], 1] + net.nodes[net.v[e], 1]) / 2)
            lon, lat = transformer.transform(mx, my)
            pts.append([round(float(lon), 6), round(float(lat), 6)])
        return pts

    edge_of: dict[tuple[int, int], int] = {}  # only failed edges are ever reported
    for e in np.flatnonzero(failed):
        edge_of[(int(net.u[e]), int(net.v[e]))] = int(e)
        edge_of[(int(net.v[e]), int(net.u[e]))] = int(e)

    def path_failed_edges(node, pred) -> list[int]:
        out, cur, guard = [], node, 0
        while cur >= 0 and pred[cur] >= 0 and guard < 100000:
            e = edge_of.get((int(cur), int(pred[cur])))
            if e is not None:
                out.append(e)
            cur, guard = pred[cur], guard + 1
        return out

    def status(i: int) -> str:
        n = s_node[i]
        if n < 0:
            return "not_on_network"
        if not np.isfinite(d_pre_a[n]):
            return "no_baseline_route"
        if not np.isfinite(d_post_a[n]):
            return "cut_off"
        if np.isfinite(d_pre_h[n]) and not np.isfinite(d_post_h[n]):
            return "hospital_cut_off"
        if d_post_a[n] > d_pre_a[n] + 50 or (
            np.isfinite(d_pre_h[n]) and d_post_h[n] > d_pre_h[n] + 50
        ):
            return "detour"
        return "connected"

    rows = []
    for i in range(len(pl)):
        n = s_node[i]
        st = status(i)
        crit = path_failed_edges(n, pred_pre_a) if n >= 0 and st in ("cut_off", "detour") else []
        crit_h = (
            path_failed_edges(n, pred_pre_h)
            if n >= 0 and st in ("cut_off", "hospital_cut_off", "detour")
            else []
        )
        ok = n >= 0

        has_hosp = ok and np.isfinite(d_pre_h[n]) and len(hosp_nodes) > 0
        h_target = int(src_pre_h[n]) if has_hosp else -1
        h_post_target = (
            int(src_post_h[n]) if ok and np.isfinite(d_post_h[n]) and len(hosp_nodes) > 0 else -1
        )
        h_name = hosp_names.get(h_target) if has_hosp else None
        h_amenity = hosp_amenities.get(h_target) if has_hosp else None
        path_pre = trace_coords(n, pred_pre_h, h_target) if has_hosp else []
        path_post = trace_coords(n, pred_post_h, h_post_target) if h_post_target >= 0 else []
        breaks = break_coords(crit_h) if crit_h else []

        rows.append(
            {
                "name": pl["name"].iat[i],
                "name_ne": pl["name_ne"].iat[i],
                "place": pl["place"].iat[i],
                "osm_id": pl["osm_id"].iat[i],
                "status": st,
                "buildings_500m": int(counts[i]),
                "hospital_name": h_name,
                "hospital_amenity": h_amenity,
                "hospital_path_pre": json.dumps(path_pre) if path_pre else None,
                "hospital_path_post": json.dumps(path_post) if path_post else None,
                "hospital_breaks": json.dumps(breaks) if breaks else None,
                "km_to_hospital_pre": round(d_pre_h[n] / 1000, 2)
                if ok and np.isfinite(d_pre_h[n])
                else None,
                "km_to_hospital_post": round(d_post_h[n] / 1000, 2)
                if ok and np.isfinite(d_post_h[n])
                else None,
                "hours_to_hospital_pre": round(float(t_pre_h[n]), 2)
                if ok and np.isfinite(t_pre_h[n])
                else None,
                "hours_to_hospital_post": round(float(t_post_h[n]), 2)
                if ok and np.isfinite(t_post_h[n])
                else None,
                "km_to_target_pre": round(d_pre_a[n] / 1000, 2)
                if ok and np.isfinite(d_pre_a[n])
                else None,
                "km_to_target_post": round(d_post_a[n] / 1000, 2)
                if ok and np.isfinite(d_post_a[n])
                else None,
                "hours_to_target_pre": round(float(t_pre_a[n]), 2)
                if ok and np.isfinite(t_pre_a[n])
                else None,
                "hours_to_target_post": round(float(t_post_a[n]), 2)
                if ok and np.isfinite(t_post_a[n])
                else None,
                "critical_ways": sorted(
                    {str(net.ways["osm_id"].iat[net.way[e]]) for e in crit + crit_h}
                ),
            }
        )
    sdf = gpd.GeoDataFrame(rows, geometry=[Point(xy) for xy in sp], crs=crs)
    sdf["extra_km_target"] = [
        round(b - a, 2) if a is not None and b is not None else None
        for a, b in zip(sdf["km_to_target_pre"], sdf["km_to_target_post"], strict=True)
    ]
    sdf["delay_hours_target"] = [
        round(b - a, 2) if a is not None and b is not None else None
        for a, b in zip(sdf["hours_to_target_pre"], sdf["hours_to_target_post"], strict=True)
    ]

    # --- reachability on a contracted graph ---------------------------------------------
    # Edges that can never fail (zero risk) are merged into super-nodes once; repairs and
    # Monte Carlo then only operate on the small set of risky edges between super-nodes.
    pe = np.where(net.observed, net.p_fail, np.where(net.in_region, p_unobserved, 0.0))
    pe = np.where(pe >= p_min_risk, pe, 0.0)  # below this, treat as no risk in the simulation
    risky = (pe >= p_min_risk) | failed
    if net.n:
        n_super, lab = connected_components(_matrix(net, ~risky), directed=False)
    else:
        n_super, lab = 0, np.zeros(0, int)
    ru, rv, rway = lab[net.u[risky]], lab[net.v[risky]], net.way[risky]
    r_failed = failed[risky]
    t_super = np.unique(lab[any_targets]) if len(any_targets) else np.zeros(0, int)

    def reach_mask(keep_risky: np.ndarray) -> np.ndarray:
        """Per original node: can it reach any target, keeping these risky edges?"""
        if net.n == 0:
            return np.zeros(0, bool)
        u, v = ru[keep_risky], rv[keep_risky]
        m = coo_matrix(
            (np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(n_super, n_super)
        ).tocsr()
        _, lab2 = connected_components(m, directed=False)
        good = np.zeros(lab2.max() + 1, bool)
        good[lab2[t_super]] = True
        return good[lab2][lab]

    base_reach = reach_mask(~r_failed)
    pre_reach = reach_mask(np.ones(len(ru), bool))
    cut_idx = [
        i
        for i in range(len(pl))
        if s_node[i] >= 0 and pre_reach[s_node[i]] and not base_reach[s_node[i]]
    ]
    failed_ways = np.unique(net.way[failed])
    repairs: list[Repair] = []
    if cut_idx:
        for w in failed_ways:
            reach = reach_mask(~r_failed | (rway == w))
            rec = [i for i in cut_idx if reach[s_node[i]]]
            if not rec:
                continue
            wrow = net.ways.iloc[w]
            t = _tags(wrow["tags"])
            repairs.append(
                {
                    "osm_id": str(wrow["osm_id"]),
                    "name": english_name(t),
                    "highway": t.get("highway", ""),
                    "is_bridge": t.get("bridge") not in (None, "no"),
                    "settlements_reconnected": len(rec),
                    "buildings_reconnected": int(sum(counts[i] for i in rec)),
                    "settlement_names": [pl["name"].iat[i] for i in rec][:20],
                }
            )
        repairs.sort(  # most buildings reconnected, then most settlements
            key=lambda r: (-r["buildings_reconnected"], -r["settlements_reconnected"])
        )

    # --- Monte Carlo --------------------------------------------------------------------
    nw = len(net.ways)
    way_p = np.zeros(nw)
    if E:
        np.maximum.at(way_p, net.way, pe)
    p_cut = np.zeros(len(pl))
    on = (
        np.flatnonzero((s_node >= 0) & pre_reach[np.maximum(s_node, 0)])
        if net.n
        else np.zeros(0, int)
    )
    if mc_draws > 0 and len(on) and len(ru):
        for _ in range(mc_draws):
            way_fail = rng.random(nw) < way_p
            reach = reach_mask(~way_fail[rway])
            p_cut[on] += ~reach[s_node[on]]
        p_cut /= mc_draws
    sdf["p_cut_off"] = np.where(np.isin(np.arange(len(pl)), on), p_cut.round(3), np.nan)

    fe = np.flatnonzero(failed)
    fgeoms = [LineString([net.nodes[net.u[e]], net.nodes[net.v[e]]]) for e in fe]
    fdf = gpd.GeoDataFrame(
        {
            "osm_id": [str(net.ways["osm_id"].iat[net.way[e]]) for e in fe],
            "highway": net.highway[fe],
            "is_bridge": net.is_bridge[fe],
            "p_fail": net.p_fail[fe].round(3),
        },
        geometry=fgeoms,
        crs=crs,
    )
    meta = {
        "edges": int(E),
        "nodes": int(net.n),
        "edges_failed": int(failed.sum()),
        "edges_unobserved": int((~net.observed & net.in_region).sum()),
        "edges_in_region": int(net.in_region.sum()),
        "edges_risky": int(risky.sum()),
        "km_network": round(float(net.length.sum()) / 1000, 2),
        "km_failed": round(float(net.length[failed].sum()) / 1000, 2),
        "hospitals_on_network": int(len(hosp_nodes)),
        "health_facilities_on_network": int(len(health_nodes)),
        "towns_on_network": int(len(town_nodes)),
        "exit_nodes": int(len(exits)),
        "fail_p": fail_p,
        "mc_draws": mc_draws,
        "p_unobserved_edge": p_unobserved,
        "p_min_risk": p_min_risk,
        "snap_m": snap_m,
        "exposure_radius_m": exposure_radius_m,
    }
    ht_points = [Point(xy) for xy in h_points] if len(ht) else []
    h_status = []
    for hs in h_all_nodes:
        if hs < 0:
            h_status.append("not_on_network")
        elif not np.isfinite(d_post_a[hs]):
            h_status.append("isolated")
        else:
            h_status.append("accessible")
    hdf = gpd.GeoDataFrame(
        {
            "name": names,
            "kind": kinds,
            "amenity": amenities,
            "osm_id": ht["osm_id"].to_list()
            if "osm_id" in ht.columns
            else [f"health/{k}" for k in range(len(ht))],
            "status": h_status,
        },
        geometry=ht_points,
        crs=crs,
    )

    return IsolationResult(sdf, fdf, repairs, meta, hdf)
