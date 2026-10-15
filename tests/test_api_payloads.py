import gzip
import json

import pytest
from fastapi.testclient import TestClient

from dhara.config import get_settings

ROUTE = [[85.30, 28.10], [85.31, 28.11], [85.32, 28.12]]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DHARA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASSISTANT_LLM", "off")
    get_settings.cache_clear()
    from dhara.api.app import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _settlements_job(job_id: str, n: int = 400) -> None:
    """A finished job folder whose settlements layer carries hospital routes (EPSG:4326)."""
    web = get_settings().jobs_dir / job_id / "web"
    web.mkdir(parents=True)
    feats = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [85.3 + i * 1e-3, 28.1]},
            "properties": {
                "name": f"S{i}",
                "osm_id": f"node/{i}",
                "hospital_path_pre": json.dumps(ROUTE * 20),
                "hospital_path_post": json.dumps(ROUTE * 20),
                "hospital_breaks": json.dumps([ROUTE[1]]),
            },
        }
        for i in range(n)
    ]
    (web / "settlements.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats})
    )


def test_job_list_drops_heavy_summary_fields(client):
    from dhara.api.app import store

    store().create("fp_test", "flowpath", {"lon": 85.3, "lat": 28.1, "length_km": 5})
    store().update(
        "fp_test",
        status="done",
        summary={"settlements": 2, "length_km": 5, "profile": [[0, 1]] * 5000},
    )
    listed = next(j for j in client.get("/api/jobs").json() if j["id"] == "fp_test")
    assert listed["summary"] == {"settlements": 2, "length_km": 5}
    # The job itself still has the full summary.
    assert len(client.get("/api/jobs/fp_test").json()["summary"]["profile"]) == 5000


def test_slim_settlements_layer_and_route_lookup(client):
    _settlements_job("j1")
    full = client.get("/api/jobs/j1/files/web/settlements.geojson").json()
    slim = client.get("/api/jobs/j1/files/web/settlements.geojson?slim=1").json()
    assert len(slim["features"]) == len(full["features"])
    props = slim["features"][0]["properties"]
    assert "hospital_path_pre" not in props and "hospital_path_post" not in props
    assert props["name"] == "S0"

    by_id = client.get("/api/jobs/j1/settlements/route?osm_id=node/7").json()
    assert json.loads(by_id["hospital_path_pre"]) == ROUTE * 20
    # Without an id, the nearest settlement point within ~50 m.
    near = client.get("/api/jobs/j1/settlements/route?lon=85.30701&lat=28.10001")
    assert near.status_code == 200
    far = client.get("/api/jobs/j1/settlements/route?lon=85.9&lat=28.9")
    assert far.status_code == 404


def test_large_layers_are_sent_gzipped(client):
    _settlements_job("j2")
    r = client.get(
        "/api/jobs/j2/files/web/settlements.geojson",
        headers={"Accept-Encoding": "gzip"},
    )
    assert r.headers["content-encoding"] == "gzip"
    assert len(r.json()["features"]) == 400  # decoded once, so not compressed twice
    raw = get_settings().jobs_dir / "j2" / "cache" / "gz"
    assert any(p.suffix == ".gz" for p in raw.iterdir())
    gz = next(raw.iterdir())
    assert json.loads(gzip.decompress(gz.read_bytes()))["type"] == "FeatureCollection"


def test_report_status_reports_missing_pdf(client):
    _settlements_job("j3")
    st = client.get("/api/jobs/j3/report/status").json()
    assert st == {"en": "missing", "ne": "missing"}
    assert client.get("/api/jobs/nope/report/status").status_code == 404
