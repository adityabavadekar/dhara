"""Check the cut-off analysis against EMSR927's damaged roads and bridges. Validation only: nothing
here is read by `src/dhara`. Credit: "European Union, Copernicus Emergency Management Service data".

Same road network, settlements, health facilities and routing as the pipeline
(`dhara.network.graph.analyse`), two ways of deciding which roads failed, both restricted to the
EMS areas (EMS graded roads only there, so a status difference is a disagreement inside them):

- ours: the job's flood/debris probability and confidence, as in the pipeline;
- EMS: OSM road segments within 15 m of an EMS road line graded Destroyed or Damaged, and OSM
  bridges within 30 m of an EMS transport point graded Destroyed or Damaged.

Reported: settlement status agreement (connected / longer route / no hospital access / cut off)
for settlements whose status is not "connected" in at least one of the two runs, and overall.
"Possibly damaged" EMS roads are counted as open (EMS itself is unsure); a second run counts them
as failed.

Usage: uv run python validation/compare_cutoff.py label=job_id [...]
"""

import json
import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_emsr927 import ROOT, _layer, ems_aois  # noqa: E402

from dhara.config import get_settings  # noqa: E402
from dhara.geo import AOI, reproject  # noqa: E402
from dhara.jobs import pipeline as P  # noqa: E402
from dhara.network import graph as G  # noqa: E402

AREAS = (1, 2, 3, 5)


def _read(p: Path) -> np.ndarray:
    with rasterio.open(p) as ds:
        return ds.read(1)


def run(job_id: str, possibly_failed: bool = False) -> dict:
    d = ROOT / "data" / "jobs" / job_id
    prov = json.loads((d / "provenance.json").read_text())
    aoi = AOI(geojson=prov["inputs"]["aoi"]["geojson"])
    grid = P.make_grid(aoi)
    cache = get_settings().cache_dir / aoi.hash
    snap = prov["data_sources"]["osm"]["snapshot"][:10]
    osm = {
        ly: gpd.read_parquet(cache / f"osm_net_{ly}_{snap}.parquet")
        for ly in ("roads", "buildings", "places", "health")
    }
    net_bounds = P._buffered(grid.bounds, P.NETWORK_BUFFER_M)
    in_aoi = _read(d / "classes.tif") != 255
    areas = [(n, nm, cast(BaseGeometry, g)) for n, (nm, g) in ems_aois().items() if n in AREAS]
    ems = [(n, nm, g) for n, nm, g in areas if g.intersects(aoi.geometry)]
    if not ems:
        return {}
    ems_area = reproject(unary_union([g for _, _, g in ems]), "EPSG:4326", grid.crs)

    def network() -> tuple[G.Network, np.ndarray]:
        net = G.build(osm["roads"], grid.crs)
        mid = (net.nodes[net.u] + net.nodes[net.v]) / 2
        pts = gpd.GeoSeries(gpd.points_from_xy(mid[:, 0], mid[:, 1]), crs=grid.crs)
        return net, pts.within(ems_area).to_numpy()

    # Ours, as in the pipeline, then failures kept inside the EMS areas only.
    ours, ours_in_ems = network()
    p_any = np.fmax(_read(d / "p_flood.tif"), _read(d / "p_debris.tif")).astype(np.float32)
    p_any[~in_aoi] = np.nan
    G.edge_hazard(ours, p_any, grid, in_aoi, tier=np.where(in_aoi, _read(d / "confidence.tif"), 0))
    bridges = (
        gpd.read_parquet(d / "bridges.parquet")
        if (d / "bridges.parquet").exists()
        else gpd.GeoDataFrame()
    )
    if len(bridges) and "p_lost" in bridges:
        G.apply_bridges(ours, bridges)
    ours.p_fail = np.where(ours_in_ems, ours.p_fail, 0.0)

    # EMS-graded damage.
    grades = {"Destroyed", "Damaged"} | ({"Possibly damaged"} if possibly_failed else set())
    lines, points = [], []
    for n, _, _ in ems:
        tl, tp = _layer(n, "transportationL"), _layer(n, "transportationP")
        if tl is not None and len(tl):
            lines.append(tl[tl["damage_gra"].isin(grades)].to_crs(grid.crs))
        if tp is not None and len(tp):
            points.append(tp[tp["damage_gra"].isin(grades)].to_crs(grid.crs))
    dmg_lines = unary_union(list(pd.concat(lines).geometry)) if lines else None
    dmg_pts = unary_union(list(pd.concat(points).geometry)) if points else None
    ref, ref_in_ems = network()
    seg = gpd.GeoSeries(
        [
            shape({"type": "LineString", "coordinates": [tuple(a), tuple(b)]})
            for a, b in zip(ref.nodes[ref.u], ref.nodes[ref.v], strict=True)
        ],
        crs=grid.crs,
    )
    fail = np.zeros(len(seg), bool)
    if dmg_lines is not None and not dmg_lines.is_empty:
        fail |= seg.distance(dmg_lines).to_numpy() <= 15.0
    if dmg_pts is not None and not dmg_pts.is_empty:
        fail |= ref.is_bridge & (seg.distance(dmg_pts).to_numpy() <= 30.0)
    ref.p_fail = np.where(fail & ref_in_ems, 1.0, 0.0)
    ref.observed = np.ones(len(seg), bool)
    ref.in_region = ours.in_region
    ref.tier = np.full(len(seg), 3, np.uint8)

    kw = dict(mc_draws=50)
    a = G.analyse(
        ours, osm["places"], osm["health"], osm["buildings"], net_bounds, grid.crs, **kw
    ).settlements
    b = G.analyse(
        ref, osm["places"], osm["health"], osm["buildings"], net_bounds, grid.crs, **kw
    ).settlements
    st = pd.DataFrame({"ours": a["status"].to_numpy(), "ems": b["status"].to_numpy()})
    affected = st[(st.ours != "connected") | (st.ems != "connected")]
    affected = affected[~affected.ours.isin(["not_on_network"])]
    return {
        "job": job_id,
        "ems_areas": [f"{n:02d} {nm}" for n, nm, _ in ems],
        "ems_failed_edges": int(fail.sum()),
        "our_failed_edges_in_ems": int(((ours.p_fail >= 0.5) & ours.observed).sum()),
        "settlements": int(len(st)),
        "agree_all": round(float((st.ours == st.ems).mean()), 3),
        "affected": int(len(affected)),
        "agree_affected": int((affected.ours == affected.ems).sum()),
        "confusion": pd.crosstab(affected.ours, affected.ems).to_dict(),
    }


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        label, jid = arg.split("=", 1)
        for possibly in (False, True):
            r = run(jid, possibly)
            if r:
                print(
                    json.dumps({"label": label, "possibly_damaged_counts_as_failed": possibly, **r})
                )
