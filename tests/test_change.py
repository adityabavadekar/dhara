"""Before/after change on a synthetic straight river: width and sideways shift are recovered."""

import numpy as np
import rasterio
from rasterio import CRS
from rasterio.transform import from_origin

from dhara.damage.change import measure

UTM = CRS.from_epsg(32645)
X0, Y0 = 300_000.0, 3_100_000.0  # top-left, metres (EPSG:32645)


def _s2(path, river_rows: slice) -> None:
    n = 300  # 10 m pixels, 3 km square
    b = np.zeros((7, n, n), np.float32)
    b[0], b[1], b[2], b[3], b[4], b[5], b[6] = 0.03, 0.06, 0.04, 0.4, 0.2, 0.1, 4  # vegetation
    b[1, river_rows], b[4, river_rows], b[3, river_rows] = 0.1, 0.02, 0.05  # open water
    with rasterio.open(
        path, "w", driver="GTiff", width=n, height=n, count=7, dtype="float32", crs=UTM,
        transform=from_origin(X0, Y0, 10, 10),
    ) as d:  # fmt: skip
        d.write(b)


def _terrain(tdir) -> None:
    tdir.mkdir()
    n = 100  # 30 m cells over the same 3 km
    acc = np.ones((n, n), np.float32)
    acc[50, :] = 200_000  # one main river flowing east along row 50 (y = Y0 - 1515 m)
    fdir = np.full((n, n), 4, np.int32)
    fdir[50, :] = 2  # east
    for name, a in (("acc", acc), ("fdir", fdir)):
        with rasterio.open(
            tdir / f"terrain_{name}.tif", "w", driver="GTiff", width=n, height=n, count=1,
            dtype=a.dtype, crs=UTM, transform=from_origin(X0, Y0, 30, 30),
        ) as d:  # fmt: skip
            d.write(a, 1)


def test_width_and_shift(tmp_path):
    # river line at S2 row ~151.5; before: 6 px wide, after: 18 px wide and 6 px further south
    _s2(tmp_path / "pre.tif", slice(149, 155))
    _s2(tmp_path / "post.tif", slice(149, 167))
    _terrain(tmp_path / "terrain")
    s, g = measure(
        tmp_path / "pre.tif",
        tmp_path / "post.tif",
        tmp_path / "terrain",
        np.ones((300, 300), bool),
        "2026-08-12",
        "2026-08-27",
    )
    assert s.stations >= 2
    assert s.width_before_median_m == 60
    assert s.width_after_median_m == 180
    assert s.shift_median_m == 60
    assert s.water_before_km2 < s.water_after_km2
    assert s.clear_both_pct == 100.0
    assert set(g["kind"]) >= {
        "width_before",
        "width_after",
        "station",
        "water_before",
        "water_after",
    }


def test_cloud_is_not_measured(tmp_path):
    _s2(tmp_path / "pre.tif", slice(149, 155))
    _s2(tmp_path / "post.tif", slice(149, 167))
    with rasterio.open(tmp_path / "post.tif", "r+") as d:
        scl = d.read(7)
        scl[:] = 9  # cloud everywhere after the event
        d.write(scl, 7)
    _terrain(tmp_path / "terrain")
    s, _ = measure(
        tmp_path / "pre.tif",
        tmp_path / "post.tif",
        tmp_path / "terrain",
        np.ones((300, 300), bool),
        "2026-08-12",
        "2026-08-27",
    )
    assert s.stations == 0 and s.width_after_median_m is None
    assert s.clear_both_km2 == 0


def test_change_facts_for_report(tmp_path):
    import json

    from dhara.sitrep.facts import Facts, _change

    c = {
        "before_day": "2026-08-12",
        "after_day": "2026-08-27",
        "clear_both_pct": 39.7,
        "water_before_km2": 3.77,
        "water_after_km2": 5.61,
        "channel_before_km2": 6.59,
        "channel_after_km2": 12.31,
        "stations": 49,
        "width_before_median_m": 80.0,
        "width_after_median_m": 170.0,
        "widened_stations": 31,
        "shift_median_m": 20.0,
        "shift_max_m": 160.0,
    }
    (tmp_path / "layers.json").write_text(json.dumps({"change": c}))
    f = Facts()
    _change(f, tmp_path)
    assert f.t("change_width_delta_m") == "+90"
    assert f.t("change_water_pct") == "+49%"
    assert f.t("change_widened_threshold_m") == "50"
    empty = Facts()
    _change(empty, tmp_path / "missing")
    assert not any(k.startswith("change_") for k in empty.items)
