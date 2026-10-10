import json

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import LineString, Point, box

from dhara.adapters.cdse import Grid
from dhara.network import graph as G

CRS_ = CRS.from_epsg(32645)
GRID = Grid(CRS_, (0.0, 0.0, 4000.0, 1000.0), 10.0)


def _gdf(geoms, tags):
    return gpd.GeoDataFrame(
        {"osm_id": [f"way/{i}" for i in range(len(geoms))], "tags": [json.dumps(t) for t in tags]},
        geometry=geoms,
        crs=CRS_,
    )


def _scene():
    # Village A — track — bridge — track — town T (hospital at T). Village B sits on the T side.
    roads = _gdf(
        [
            LineString([(500, 500), (1900, 500)]),
            LineString([(1900, 500), (2100, 500)]),  # bridge over the river at x≈2000
            LineString([(2100, 500), (3500, 500)]),
        ],
        [{"highway": "track"}, {"highway": "track", "bridge": "yes"}, {"highway": "unclassified"}],
    )
    places = _gdf(
        [Point(500, 500), Point(3000, 500), Point(3500, 500)],
        [
            {"place": "village", "name": "A"},
            {"place": "village", "name": "B"},
            {"place": "town", "name": "T"},
        ],
    )
    health = _gdf([Point(3500, 520)], [{"amenity": "hospital"}])
    buildings = _gdf(
        [box(480 + 8 * i, 480, 485 + 8 * i, 485) for i in range(12)], [{"building": "yes"}] * 12
    )
    return roads, places, health, buildings


def test_bridge_loss_cuts_off_village_and_ranks_repair():
    roads, places, health, buildings = _scene()
    prob = np.zeros((GRID.height, GRID.width), np.float32)
    prob[:, 195:205] = 0.95  # debris across the river at the bridge only
    net = G.build(roads, CRS_)
    G.edge_hazard(net, prob, GRID)
    # Analysed region = exactly the road extent, so no edge sits near an "exit" except the ends.
    res = G.analyse(net, places, health, buildings, (-5000, -5000, 9000, 6000), CRS_, mc_draws=200)
    s = res.settlements.set_index("name")
    assert s.loc["A", "status"] == "cut_off"
    assert s.loc["B", "status"] == "connected"
    assert s.loc["A", "buildings_500m"] == 12
    assert s.loc["A", "km_to_hospital_pre"] == 3.0
    assert pd.isna(s.loc["A", "km_to_hospital_post"])
    assert s.loc["A", "p_cut_off"] > 0.8
    assert res.repairs[0]["is_bridge"] and res.repairs[0]["settlements_reconnected"] == 1
    assert res.repairs[0]["buildings_reconnected"] == 12


def test_unobserved_edges_stay_open_deterministically():
    roads, places, health, buildings = _scene()
    prob = np.full((GRID.height, GRID.width), np.nan, np.float32)  # nothing observed
    net = G.build(roads, CRS_)
    G.edge_hazard(net, prob, GRID)
    res = G.analyse(net, places, health, buildings, (-5000, -5000, 9000, 6000), CRS_, mc_draws=0)
    assert set(res.settlements["status"]) == {"connected"}
    assert res.meta["edges_unobserved"] == res.meta["edges"]
