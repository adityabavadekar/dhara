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
    monkeypatch.setenv("ASSISTANT_LLM", "off")
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


def test_job_elevation_profile(client):
    from dhara.api.app import store
    from dhara.config import get_settings

    jid = "proftest123"
    d = get_settings().jobs_dir / jid
    d.mkdir(parents=True, exist_ok=True)
    mock_prof = {
        "points": [
            {"dist_km": 0.0, "elevation_m": 1200.0, "lng": 85.3, "lat": 28.1, "slope_pct": 0.0},
            {"dist_km": 1.0, "elevation_m": 1400.0, "lng": 85.31, "lat": 28.11, "slope_pct": 20.0},
        ],
        "summary": {
            "total_distance_km": 1.0,
            "min_elevation_m": 1200,
            "max_elevation_m": 1400,
            "elevation_gain_m": 200,
            "elevation_loss_m": 0,
            "max_slope_pct": 20.0,
            "avg_slope_pct": 20.0,
        },
    }
    (d / "profile.json").write_text(json.dumps(mock_prof))
    store().create(jid, "flowpath", {})

    resp = client.post(f"/api/jobs/{jid}/profile", json={"path": [], "step_m": 30.0})
    assert resp.status_code == 200
    assert resp.json()["summary"]["total_distance_km"] == 1.0
    assert len(resp.json()["points"]) == 2


def test_job_location_name_and_migration(client, monkeypatch):
    from dhara.api import app as appmod
    from dhara.api.app import store

    # No real analysis: the job only needs to exist.
    monkeypatch.setattr(appmod, "_run_analysis", lambda *a, **k: None)

    # Job created with explicit name
    r = client.post(
        "/api/jobs", json={"aoi": SMALL, "event": "2026-08-26", "name": "Custom Valley"}
    )
    assert r.status_code == 202
    jid = r.json()["id"]

    job = client.get(f"/api/jobs/{jid}").json()
    assert job["location_name"] == "Custom Valley"

    # In job list, location_name is preserved
    jobs = client.get("/api/jobs").json()
    match = next(j for j in jobs if j["id"] == jid)
    assert match["location_name"] == "Custom Valley"

    # Migration test on unmigrated row
    store()._con.execute("UPDATE jobs SET location_name = NULL WHERE id = ?", [jid])
    store()._migrate_locations()
    updated = client.get(f"/api/jobs/{jid}").json()
    assert updated["location_name"] is not None


def test_runs_queue_one_at_a_time_and_identical_requests_reuse(client, monkeypatch):
    from dhara.api import app as appmod

    # Stub runs never start, so every submitted analysis stays queued.
    monkeypatch.setattr(appmod, "_run_analysis", lambda *a, **k: None)
    assert appmod.state["analysis_pool"]._max_workers == 1

    def box(dx: float) -> dict:
        x0, y0 = 85.33 + dx, 28.14
        ring = [[x0, y0], [x0 + 0.02, y0], [x0 + 0.02, y0 + 0.02], [x0, y0 + 0.02], [x0, y0]]
        return {"type": "Polygon", "coordinates": [ring]}

    body = {"aoi": box(0), "event": "2026-08-26", "note": "  first   check  "}
    first = client.post("/api/jobs", json=body).json()
    # Same area, time and options: the existing run is returned instead of a second one.
    again = client.post("/api/jobs", json={**body, "note": "other"}).json()
    assert again == {"id": first["id"], "reused": True}
    assert client.get(f"/api/jobs/{first['id']}").json()["request"]["note"] == "first check"

    ids = [first["id"]]
    for i in range(1, appmod.MAX_PENDING["analysis"]):
        ids.append(client.post("/api/jobs", json={**body, "aoi": box(0.05 * i)}).json()["id"])
    r = client.post("/api/jobs", json={**body, "aoi": box(0.5)})
    assert r.status_code == 429 and "one at a time" in r.json()["detail"]

    listed = {j["id"]: j for j in client.get("/api/jobs").json()}
    assert [listed[i]["queue_ahead"] for i in ids] == list(range(len(ids)))
    assert client.post("/api/jobs", json={**body, "note": "x" * 201}).status_code == 422


def test_job_list_carries_brief_facts(client, monkeypatch):
    from dhara.api import app as appmod
    from dhara.api.app import store

    monkeypatch.setattr(appmod, "_run_analysis", lambda *a, **k: None)
    jid = client.post("/api/jobs", json={"aoi": SMALL, "event": "2026-08-26"}).json()["id"]
    fact = {"value": 3, "unit": "", "source": "t"}
    store().put_facts(
        jid, {"buildings_hit": {**fact, "text": "3"}, "job_id": {**fact, "text": jid}}
    )
    store().update(jid, status="done")
    j = next(j for j in client.get("/api/jobs").json() if j["id"] == jid)
    assert j["brief"] == {"buildings_hit": "3"}
    assert j["queue_ahead"] is None
