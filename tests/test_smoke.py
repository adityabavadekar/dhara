"""End-to-end smoke tests on a tiny bbox.

- Offline: network adapters replaced by synthetic fixtures; runs in CI.
- Live: real CDSE + ohsome + DEM. Run with `DHARA_LIVE=1 uv run pytest -m live` (needs .env).
"""

import json
import os
from datetime import UTC, date, datetime

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely import LineString, Point, box

from dhara.adapters.sentinel1 import S1Pass
from dhara.config import get_settings
from dhara.fusion import fuse as F
from dhara.geo import AOI
from dhara.jobs import pipeline
from dhara.jobs.scene_selection import S1Pair

TINY = AOI.from_bbox(85.330, 28.145, 85.350, 28.170)  # ~5 km², Bhote Koshi at Syabrubesi


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DHARA_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _pass(day: int, orbit: int, state: str) -> S1Pass:
    t = datetime(2026, 8, day, 12, 21, 41, tzinfo=UTC)
    return S1Pass(
        platform="sentinel-1d",
        relative_orbit=orbit,
        orbit_state=state,
        start=t,
        end=t,
        scene_ids=[f"FAKE_{orbit}_{day}"],
        aoi_coverage=1.0,
    )


def _fake_fetch(p, grid, radiometry, *, with_geometry_bands=False):
    rng = np.random.default_rng(p.start.day + p.relative_orbit)
    h, w = grid.height, grid.width
    vv = 10 ** (rng.normal(-8, 1, (h, w)) / 10)
    vh = 10 ** (rng.normal(-15, 1, (h, w)) / 10)
    if p.start.day >= 27:  # post: flood stripes (dark) and debris stripes (bright)
        for c in range(0, w, 96):
            vv[:, c + 16 : c + 40] = 10 ** (-22 / 10)
            vh[:, c + 16 : c + 40] = 10 ** (-28 / 10)
            vv[:, c + 56 : c + 72] = 10 ** (-1 / 10)
            vh[:, c + 56 : c + 72] = 10 ** (-7 / 10)
    bands = [vv, vh]
    if with_geometry_bands:
        shadow = np.zeros((h, w))
        shadow[:15] = 1
        bands += [np.full((h, w), 40.0), shadow]
    return np.stack(bands).astype(np.float32), []


def _fake_dem(grid):
    """A gentle valley draining south; the AOI sits on its floor."""
    xs = np.arange(grid.width)
    ys = np.arange(grid.height)[:, None]
    dem = 1400 + 0.3 * np.abs(xs - grid.width / 2) + 0.05 * (grid.height - ys)
    return dem.astype(np.float32), {"dataset": "fake", "tiles": []}


def _fake_osm(layer, bbox, snap):
    """Features placed on the AOI (the request bbox is the larger network area)."""
    assert snap <= date(2026, 7, 27)
    minx, miny, maxx, maxy = TINY.bounds
    cy = (miny + maxy) / 2
    if layer == "buildings":
        geoms = [
            box(minx + i * 0.0008, cy, minx + i * 0.0008 + 0.0001, cy + 0.0001) for i in range(20)
        ]
    elif layer == "roads":
        geoms = [LineString([(minx + 0.001, cy), (maxx - 0.001, cy)])]
    elif layer == "places":
        geoms = [Point(minx + 0.002, cy), Point(maxx - 0.002, cy)]
    else:
        geoms = [Point(maxx - 0.002, cy - 0.001)]
    tags = {
        "roads": json.dumps({"highway": "track"}),
        "places": json.dumps({"place": "village", "name": "X"}),
        "health": json.dumps({"amenity": "hospital"}),
    }.get(layer, json.dumps({"building": "yes"}))
    return gpd.GeoDataFrame(
        {"osm_id": [f"way/{i}" for i in range(len(geoms))], "tags": [tags] * len(geoms)},
        geometry=geoms,
        crs="EPSG:4326",
    )


def test_offline_pipeline(data_dir, monkeypatch):
    asc = S1Pair(
        relative_orbit=85,
        orbit_state="ascending",
        pre=_pass(16, 85, "ascending"),
        post=_pass(28, 85, "ascending"),
    )
    desc = S1Pair(
        relative_orbit=19,
        orbit_state="descending",
        pre=_pass(24, 19, "descending"),
        post=_pass(29, 19, "descending"),
    )
    monkeypatch.setattr(pipeline, "find_s1_pairs", lambda *a, **k: ([asc, desc], []))
    monkeypatch.setattr(pipeline, "fetch_pass", _fake_fetch)
    monkeypatch.setattr(pipeline, "fetch_dem", _fake_dem)
    monkeypatch.setattr(pipeline.ohsome, "fetch_layer", _fake_osm)
    monkeypatch.setattr(pipeline.s2, "candidate_days", lambda *a, **k: [])

    res = pipeline.run_job(pipeline.JobSpec(aoi=TINY, event="2026-08-26", use_ml=False))
    d = res.job_dir
    expected = ("classes.tif", "p_flood.tif", "p_debris.tif", "confidence.tif", "hand.tif")
    for f in (*expected, "buildings.parquet", "provenance.json", "summary.json"):
        assert (d / f).exists(), f
    prov = json.loads((d / "provenance.json").read_text())
    assert [s["relative_orbit"] for s in prov["data_sources"]["sentinel1"]] == [85, 19]
    assert prov["data_sources"]["osm"]["snapshot"] == "2026-07-27T00:00:00Z"
    failed = {k: v for k, v in prov["stages"].items() if v["status"] != "ok"}
    assert not failed, failed
    with rasterio.open(d / "classes.tif") as ds:
        cls = ds.read(1)
    assert (cls == F.FLOOD).any(), np.unique(cls, return_counts=True)
    assert (cls == F.DEBRIS).any(), np.unique(cls, return_counts=True)
    assert res.summary["buildings"]["count"] == 20


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("DHARA_LIVE") != "1", reason="set DHARA_LIVE=1 for live run")
def test_live_pipeline_tiny_bbox(data_dir):
    res = pipeline.run_job(pipeline.JobSpec(aoi=TINY, event="2026-08-26T08:40:00+05:45"))
    prov = json.loads((res.job_dir / "provenance.json").read_text())
    failed = {k: v for k, v in prov["stages"].items() if v["status"] != "ok"}
    assert not failed, failed
