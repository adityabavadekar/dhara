import json
import time

import pytest
from fastapi.testclient import TestClient

from dhara.config import get_settings

SMALL = {
    "type": "Polygon",
    "coordinates": [
        [[85.33, 28.14], [85.35, 28.14], [85.35, 28.16], [85.33, 28.16], [85.33, 28.14]]
    ],
}
HUGE = {
    "type": "Polygon",
    "coordinates": [[[84.0, 27.0], [86.0, 27.0], [86.0, 29.0], [84.0, 29.0], [84.0, 27.0]]],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DHARA_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    from dhara.api.app import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_health_and_config(client):
    assert client.get("/api/health").json() == {"ok": True}
    cfg = client.get("/api/config").json()
    assert cfg["osm_snapshot_max"] == "2026-07-27T00:00:00Z"
    assert "Educational prototype" in cfg["disclaimer"]
    assert any("Copernicus Sentinel" in a for a in cfg["attributions"])


def test_places_search(client):
    res = client.get("/api/places").json()
    assert len(res) > 0
    assert any("Rasuwa" in p["name"] for p in res)

    # Search by prefix
    syab = client.get("/api/places?q=syab").json()
    assert len(syab) > 0
    assert "Syabrubesi" in syab[0]["name"]
    assert "polygon" in syab[0]


def test_rejects_large_aoi_and_naive_time(client):
    r = client.post("/api/jobs", json={"aoi": HUGE, "event": "2026-08-26"})
    assert r.status_code == 422 and "cap" in r.text
    r = client.post("/api/jobs", json={"aoi": SMALL, "event": "2026-08-26T08:40:00"})
    assert r.status_code == 422


def test_job_lifecycle_with_stub_pipeline(client, monkeypatch, tmp_path):
    from dhara.jobs import pipeline

    def fake_run_job(spec, job_id, progress=None):
        d = get_settings().jobs_dir / job_id
        (d / "web").mkdir(parents=True)
        (d / "web" / "settlements.geojson").write_text('{"type":"FeatureCollection","features":[]}')
        (d / "facts.json").write_text(
            json.dumps({"facts": {"x": {"value": 1, "unit": "", "text": "1", "source": "t"}}})
        )
        assert progress is not None
        progress("fusion", 0.6)
        return pipeline.JobResult(job_id=job_id, job_dir=d, summary={"ok": True})

    monkeypatch.setattr(pipeline, "run_job", fake_run_job)
    r = client.post("/api/jobs", json={"aoi": SMALL, "event": "2026-08-26T08:40:00+05:45"})
    assert r.status_code == 202
    jid = r.json()["id"]
    for _ in range(100):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    assert j["request"]["event"] == "2026-08-26T02:55:00Z"
    assert client.get(f"/api/jobs/{jid}/facts").json()[0]["key"] == "x"
    assert client.get(f"/api/jobs/{jid}/files/web/settlements.geojson").status_code == 200
    assert client.get(f"/api/jobs/{jid}/files/..%2F..%2F.env").status_code in (400, 404)
    assert client.get(f"/api/jobs/{jid}/files/error.txt").status_code == 404
    z = client.get(f"/api/jobs/{jid}/download.zip")
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"

    # Test AI Multilingual Assistant
    chat_resp = client.post(
        f"/api/jobs/{jid}/chat",
        json={"message": "What is the damage status?", "lang": "ne"},
    )
    assert chat_resp.status_code == 200
    assert len(chat_resp.json()["answer"]) > 0
    assert chat_resp.json()["lang"] == "ne"

    explain_resp = client.post(
        f"/api/jobs/{jid}/explain-asset",
        json={"asset_type": "bridge", "asset_name": "Main Bridge", "lang": "en"},
    )
    assert explain_resp.status_code == 200
    assert explain_resp.json()["asset_type"] == "bridge"


def test_model_metrics_come_from_training_outputs(client):
    res = client.get("/api/model").json()
    assert set(res) == {"deployed", "runs"}
    for run in res["runs"]:
        assert set(run["test"]) >= {"f1_flood", "iou_flood"}
