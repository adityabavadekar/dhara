"""HTTP API: submit analysis/flow-path jobs, poll progress, fetch layers, sitrep and downloads.

Run: `uv run uvicorn dhara.api.app:app --host 0.0.0.0 --port 8000`.
Jobs run in an in-process thread pool (the pipeline is I/O- and NumPy/GDAL-bound). State lives in
DuckDB. All geometry crossing the API is EPSG:4326 GeoJSON; all times are ISO 8601 UTC (…Z).
"""

import io
import json
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from dhara.api.store import Store
from dhara.config import OSM_SNAPSHOT_MAX, get_settings
from dhara.geo import AOI, AOIError
from dhara.timeutil import EventTime

ATTRIBUTIONS = [
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space "
    "GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "© OpenStreetMap contributors (ODbL), via the ohsome API (HeiGIT).",
    "Training data: Kuro Siwo (Bountos et al., NeurIPS 2024).",
]
DISCLAIMER = "Educational prototype, not an operational tool."

# Files a client may fetch from a job directory.
ALLOWED_FILES = {
    "provenance.json",
    "summary.json",
    "facts.json",
    "layers.json",
    "repairs.json",
    "sitrep_en.html",
    "sitrep_ne.html",
    "sitrep_en.pdf",
    "sitrep_ne.pdf",
    "classes.tif",
    "p_flood.tif",
    "p_debris.tif",
    "confidence.tif",
}
WEB_EXT = {".png", ".geojson"}

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    state["store"] = Store(s.dhara_data_dir / "dhara.duckdb")
    state["pool"] = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dhara-job")
    yield
    state["pool"].shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Dhara API", version="0.1.0", lifespan=lifespan)


def store() -> Store:
    return state["store"]


# ----------------------------------------------------------------------------- models


class AnalysisRequest(BaseModel):
    aoi: dict = Field(description="GeoJSON Polygon/MultiPolygon in EPSG:4326")
    event: EventTime = Field(description="ISO 8601 date or datetime with offset")
    relative_orbit: int | None = None
    use_s2: bool = True
    use_ml: bool = True


class FlowPathRequest(BaseModel):
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    length_km: float = Field(default=40.0, gt=1, le=120)
    corridor_m: float = Field(default=600.0, gt=50, le=3000)


class JobOut(BaseModel):
    id: str
    kind: Literal["analysis", "flowpath"]
    status: str
    stage: str
    progress: float
    created_utc: str
    updated_utc: str
    request: dict
    summary: dict | None = None
    error: str | None = None


# ----------------------------------------------------------------------------- workers


def _run_analysis(job_id: str, req: AnalysisRequest) -> None:
    from dhara.jobs.pipeline import JobSpec, run_job

    st = store()
    st.update(job_id, status="running", stage="starting")
    try:
        spec = JobSpec(
            aoi=AOI(geojson=req.aoi),
            event=req.event,
            relative_orbit=req.relative_orbit,
            use_s2=req.use_s2,
            use_ml=req.use_ml,
        )
        res = run_job(
            spec,
            job_id,
            progress=lambda stage, p: st.update(job_id, stage=stage, progress=round(p, 3)),
        )
        facts = json.loads((res.job_dir / "facts.json").read_text())["facts"]
        st.put_facts(job_id, facts)
        st.update(job_id, status="done", stage="done", progress=1.0, summary=res.summary)
    except Exception as e:
        st.update(job_id, status="failed", error=f"{type(e).__name__}: {e}")
        (get_settings().jobs_dir / job_id).mkdir(parents=True, exist_ok=True)
        (get_settings().jobs_dir / job_id / "error.txt").write_text(traceback.format_exc())


def _run_flowpath(job_id: str, req: FlowPathRequest) -> None:
    from dhara.flowpath.trace import save, trace

    st = store()
    st.update(job_id, status="running", stage="terrain", progress=0.1)
    try:
        fp = trace(req.lon, req.lat, length_km=req.length_km, corridor_m=req.corridor_m)
        save(fp, get_settings().jobs_dir / job_id)
        st.update(job_id, status="done", stage="done", progress=1.0, summary=fp.meta)
    except Exception as e:
        st.update(job_id, status="failed", error=f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- routes


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/config")
def config() -> dict:
    s = get_settings()
    return {
        "max_aoi_km2": s.dhara_max_aoi_km2,
        "osm_snapshot_max": f"{OSM_SNAPSHOT_MAX.isoformat()}T00:00:00Z",
        "attributions": ATTRIBUTIONS,
        "disclaimer": DISCLAIMER,
        "ml_model_available": _ml_available(),
    }


def _ml_available() -> bool:
    from dhara.members.ml_seg import model_available

    return model_available()


@app.post("/api/scenes")
def scenes(req: AnalysisRequest) -> dict:
    """Preview the Sentinel-1 pairs a job would use (quick, catalogue only)."""
    from dhara.jobs.scene_selection import SceneSelectionError, find_s1_pairs

    try:
        aoi = AOI(geojson=req.aoi)
        aoi.check_size()
        pairs, notes = find_s1_pairs(aoi, req.event)
    except (AOIError, SceneSelectionError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    return {
        "pairs": [json.loads(p.model_dump_json()) for p in pairs],
        "notes": notes,
        "area_km2": round(aoi.area_km2, 2),
    }


@app.post("/api/jobs", status_code=202)
def create_job(req: AnalysisRequest) -> dict:
    try:
        aoi = AOI(geojson=req.aoi)
        aoi.check_size()
    except (AOIError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    job_id = uuid.uuid4().hex[:12]
    store().create(job_id, "analysis", json.loads(req.model_dump_json()))
    state["pool"].submit(_run_analysis, job_id, req)
    return {"id": job_id}


@app.post("/api/flowpath", status_code=202)
def create_flowpath(req: FlowPathRequest) -> dict:
    job_id = "fp_" + uuid.uuid4().hex[:10]
    store().create(job_id, "flowpath", req.model_dump())
    state["pool"].submit(_run_flowpath, job_id, req)
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs(limit: int = 30) -> list[JobOut]:
    return [JobOut(**j) for j in store().recent(limit)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JobOut:
    j = store().get(job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    return JobOut(**j)


@app.get("/api/jobs/{job_id}/facts")
def job_facts(job_id: str) -> list[dict]:
    return store().facts(job_id)


def _job_dir(job_id: str) -> Path:
    if not job_id.replace("_", "").isalnum():
        raise HTTPException(400, "bad job id")
    d = get_settings().jobs_dir / job_id
    if not d.is_dir():
        raise HTTPException(404, "job not found")
    return d


@app.get("/api/jobs/{job_id}/files/{name:path}")
def job_file(job_id: str, name: str):
    d = _job_dir(job_id)
    p = (d / name).resolve()
    if d.resolve() not in p.parents:
        raise HTTPException(400, "bad path")
    ok = name in ALLOWED_FILES or (
        p.parent in (d.resolve() / "web", d.resolve()) and p.suffix in WEB_EXT
    )
    if not ok or not p.is_file():
        raise HTTPException(404, "file not available")
    media = {
        ".geojson": "application/geo+json",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".tif": "image/tiff",
    }.get(p.suffix, "application/octet-stream")
    return FileResponse(p, media_type=media)


@app.get("/api/jobs/{job_id}/download.zip")
def download(job_id: str) -> Response:
    d = _job_dir(job_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((d / "web").glob("*.geojson")):
            z.write(f, f"geojson/{f.name}")
        for name in (
            "sitrep_en.html",
            "sitrep_ne.html",
            "sitrep_en.pdf",
            "provenance.json",
            "facts.json",
            "summary.json",
        ):
            if (d / name).exists():
                z.write(d / name, name)
        for name in ("flowpath.geojson", "corridor.geojson", "settlements.geojson", "meta.json"):
            if (d / name).exists():
                z.write(d / name, f"flowpath/{name}")
        z.writestr("ATTRIBUTION.txt", "\n".join([DISCLAIMER, *ATTRIBUTIONS]) + "\n")
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="dhara_{job_id}.zip"'},
    )
