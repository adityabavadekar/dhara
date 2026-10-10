import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely import LineString, box

from dhara.adapters.cdse import Grid
from dhara.damage.overlay import sample_features

GRID = Grid(CRS.from_epsg(32645), (0.0, 0.0, 1000.0, 1000.0), 10.0)


def test_status_per_feature():
    prob = np.zeros((100, 100), np.float32)
    prob[:, :50] = 0.9  # west half flooded (x < 500 m)
    prob[:10, 90:] = np.nan  # NE corner not observable
    gdf = gpd.GeoDataFrame(
        {"osm_id": ["a", "b", "c", "d", "e"]},
        geometry=[
            box(100, 100, 108, 108),  # sub-pixel building, flooded
            box(700, 500, 720, 520),  # dry
            box(950, 950, 960, 960),  # not observable
            LineString([(400, 300), (600, 300)]),  # road crossing the boundary
            box(5000, 5000, 5010, 5010),  # off-grid
        ],
        crs=GRID.crs,
    )
    out = sample_features(gdf, prob, GRID).set_index("osm_id")
    assert out.loc["a", "status"] == "hit" and out.loc["a", "n_px"] >= 1
    assert out.loc["b", "status"] == "not_hit"
    assert out.loc["c", "status"] == "not_observable"
    assert out.loc["d", "status"] == "hit" and 0.3 < out.loc["d", "frac_hit"] < 0.7
    assert out.loc["e", "status"] == "outside"
