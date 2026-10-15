"""HTTP API: submit analysis/flow-path jobs, poll progress, fetch layers, sitrep and downloads.

Run: `uv run uvicorn dhara.api.app:app --host 0.0.0.0 --port 8000`.
Jobs run in an in-process thread pool (the pipeline is I/O- and NumPy/GDAL-bound). State lives in
DuckDB. All geometry crossing the API is EPSG:4326 GeoJSON; all times are ISO 8601 UTC (…Z).
"""

import contextlib
import functools
import gzip
import io
import json
import shutil
import threading
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

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
    "assistant_log.jsonl",
    "classes.tif",
    "p_flood.tif",
    "p_debris.tif",
    "confidence.tif",
}
WEB_EXT = {".png", ".geojson"}

state: dict[str, Any] = {}


def _resync_facts(store: Store, jobs_dir: Path) -> None:
    """Load each job's facts.json into the store, so a re-rendered sitrep and the API agree.

    Finished analyses made outside the API (for example with the CLI) are registered as done,
    so they appear in the job list."""
    if not jobs_dir.exists():
        return

    def flat(req: dict) -> dict:
        # Provenance records the AOI as {"geojson": polygon}; the API and UI use the polygon.
        aoi = req.get("aoi")
        return {**req, "aoi": aoi["geojson"]} if isinstance(aoi, dict) and "geojson" in aoi else req

    for d in jobs_dir.iterdir():
        row = store.get(d.name)
        if row is not None:
            req = row.get("request") or {}
            if isinstance(req, dict) and flat(req) is not req:
                store.update(d.name, request=flat(req))
            continue
        if (d / "layers.json").exists() and (d / "summary.json").exists():
            with contextlib.suppress(Exception):
                prov = json.loads((d / "provenance.json").read_text())
                store.create(d.name, "analysis", flat(prov.get("inputs", {})))
                store.update(
                    d.name,
                    status="done",
                    stage="done",
                    progress=1.0,
                    summary=json.loads((d / "summary.json").read_text()),
                )
    for d in jobs_dir.iterdir():
        f = d / "facts.json"
        if f.exists() and store.get(d.name) is not None:
            store.put_facts(d.name, json.loads(f.read_text())["facts"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    s = get_settings()
    state["store"] = Store(s.dhara_data_dir / "dhara.duckdb")
    _resync_facts(state["store"], s.jobs_dir)
    state["pool"] = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dhara-job")
    yield
    state["pool"].shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Dhara API", version="0.1.0", lifespan=lifespan)


class _CompressText:
    """Gzip JSON/GeoJSON/HTML responses (map layers shrink several-fold); files that are already
    compressed (PDF, ZIP, GeoTIFF, PNG) pass through untouched."""

    SKIP = (".pdf", ".zip", ".tif", ".png")

    def __init__(self, app_):
        self.app = app_
        self.gzip = GZipMiddleware(app_, minimum_size=1024, compresslevel=5)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not scope["path"].endswith(self.SKIP):
            return await self.gzip(scope, receive, send)
        return await self.app(scope, receive, send)


app.add_middleware(_CompressText)
app.add_middleware(
    CORSMiddleware,
    # The API holds no cookies or sessions, so any origin may read it without credentials.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


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
        # Build the report PDFs now, in the background, so they are ready when someone opens them.
        threading.Thread(
            target=_prebuild_reports, args=(job_id,), daemon=True, name=f"report-{job_id}"
        ).start()
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


_REPO = Path(__file__).resolve().parents[3]
_METRIC_KEYS = ("f1_flood", "iou_flood", "precision_flood", "recall_flood", "event_f1_flood_ci95")


def _split_metrics(m: dict, split: str) -> dict:
    return {k: m.get(split, {}).get(k) for k in _METRIC_KEYS}


@app.get("/api/model")
def model() -> dict:
    """The deployed segmentation model and every trained variant, as written by `training/train.py`.

    `test` is the held-out Kuro Siwo test split; `mountain` is the held-out high-relief events.
    """
    from dhara.members.ml_seg import MODEL_DIR

    def read(d: Path) -> dict | None:
        if not (d / "metrics.json").exists() or not (d / "model_card.json").exists():
            return None
        m = json.loads((d / "metrics.json").read_text())
        card = json.loads((d / "model_card.json").read_text())
        # Runs trained before the split fix used a tile-level fallback that also put mountain
        # hold-out tiles into training. Their val and test sets are then the same object, so
        # identical val/test metrics identify them; their mountain score is not held out.
        held_out = m.get("val") != m.get("test")
        s1 = None
        if (d / "metrics_sen1floods11.json").exists():
            sp = json.loads((d / "metrics_sen1floods11.json").read_text()).get("splits", {})
            s1 = {
                k: (sp.get(k) or {}).get("model", {}).get("iou_water") for k in ("test", "bolivia")
            }
        return {
            "run_name": card.get("run_name", d.name),
            "mountain_held_out": held_out,
            "architecture": card.get("model"),
            "training_data": card.get("training_data"),
            "encoder_weights": card.get("encoder_weights"),
            "splits": card.get("splits", {}),
            "test": _split_metrics(m, "test"),
            "mountain": _split_metrics(m, "mountain"),
            # Water IoU on Sen1Floods11 (post-only input): test split and the held-out Bolivia event.
            "sen1floods11": s1,
        }

    runs = [r for d in sorted((_REPO / "experiments").glob("*/")) if (r := read(d))]
    return {"deployed": read(MODEL_DIR), "runs": runs}


def _ml_available() -> bool:
    from dhara.members.ml_seg import model_available

    return model_available()


@app.get("/api/places")
def places(q: str = "", limit: int = 8) -> list[dict]:
    """Search pre-event gazetteer for places, towns, valleys and corridors."""
    from dhara.places import search_places

    return search_places(q, limit=limit)


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


# The job list is polled every few seconds, so it carries only the summary fields the list shows;
# the full summary (e.g. a flow path's elevation profile) comes from /api/jobs/{id}.
LIST_SUMMARY_KEYS = ("area_km2", "settlements", "length_km")


@app.get("/api/jobs")
def list_jobs(limit: int = 30) -> list[JobOut]:
    out = []
    for j in store().recent(limit):
        row: dict[str, Any] = dict(j)
        if isinstance(s := row.get("summary"), dict):
            row["summary"] = {k: s[k] for k in LIST_SUMMARY_KEYS if k in s}
        out.append(JobOut(**row))
    return out


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JobOut:
    j = store().get(job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    return JobOut(**j)


@app.get("/api/jobs/{job_id}/facts")
def job_facts(job_id: str) -> list[dict]:
    """The job's facts ledger. For a finished analysis these are the report's facts (cached, rebuilt
    when an input or the facts code changes), so the dashboard and the report never disagree."""
    j = store().get(job_id)
    if j and j.get("kind") == "analysis" and j.get("status") == "done":
        d = _job_dir(job_id)
        if (d / "summary.json").exists() and (d / "provenance.json").exists():
            payload, _ = _sitrep_facts(d)
            return [{"key": k, **v} for k, v in payload["items"].items()]
    return store().facts(job_id)


# Bump when the report map drawing changes, so cached maps are redrawn.
SITREP_MAPS_VERSION = 6
# Built frontend (pnpm build in web/); served at / and printed for the PDF report.
_web_dist = Path(__file__).resolve().parents[3] / "web" / "dist"


# Bump when the report facts code changes, so cached facts are rebuilt.
SITREP_FACTS_VERSION = 5
# Files the report writes into a job folder; they are outputs, not inputs of the report facts.
_REPORT_OUTPUTS = ("sitrep_", "report_", "assistant_log")


def _facts_inputs_key(d: Path) -> str:
    """Modification times of everything the report facts read: the job folder and the flow-path
    folders next to it (a trace started inside the area adds rows)."""
    files = [
        p
        for p in sorted(d.rglob("*"))
        if p.is_file()
        and not p.name.startswith(_REPORT_OUTPUTS)
        and "cache" not in p.relative_to(d).parts
    ]
    for fp in sorted(d.parent.glob("fp_*")):
        files += [fp / n for n in ("meta.json", "settlements.geojson") if (fp / n).exists()]
    return json.dumps(
        [SITREP_FACTS_VERSION]
        + [[str(p.relative_to(d.parent)), p.stat().st_mtime_ns] for p in files]
    )


def _json_default(o: Any) -> Any:
    return o.item() if hasattr(o, "item") else str(o)


def _sitrep_facts(d: Path) -> tuple[dict, Any]:
    """Report facts, values, tables and ids. Building them reads every job output (seconds), so
    they are cached in sitrep_facts.json until an input changes. Returns (payload, Facts or None
    when served from the cache)."""
    from dhara.sitrep.facts import build
    from dhara.sitrep.render import ATTR_DEM, ATTR_OSM, ATTR_S

    key = _facts_inputs_key(d)
    cache = d / "sitrep_facts.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("key") == key:
            return cached["payload"], None
    f = build(d)
    f.add("attr_s", ATTR_S, source="required attribution")
    f.add("attr_dem", ATTR_DEM, source="required attribution")
    f.add("attr_osm", ATTR_OSM, source="required attribution")
    f.add("sensor_s1", "Sentinel-1", source="sensor name")
    f.add("sensor_s2", "Sentinel-2", source="sensor name")
    f.add("dem_name", "Copernicus DEM GLO-30", source="dataset name")
    f.add(
        "attr_train",
        "Training data: Kuro Siwo (Bountos et al., NeurIPS 2024).",
        source="required citation",
    )
    payload = {
        "facts": {k: v.text for k, v in f.items.items()},
        "values": {k: v.value for k, v in f.items.items()},
        "tables": f.tables,
        "ids": f.ids,
        "items": {
            k: {"value": v.value, "unit": v.unit, "text": v.text, "source": v.source}
            for k, v in f.items.items()
        },
    }
    payload = json.loads(json.dumps(payload, default=_json_default))
    cache.write_text(json.dumps({"key": key, "payload": payload}))
    store().put_facts(d.name, payload["items"])  # the assistant reads the ledger from the store
    return payload, f


def _maps_key(d: Path, ids: dict) -> str:
    # Rendering the report maps takes about a minute, so they are cached per job and redrawn only
    # when an input they are drawn from changes (or a flow path is traced in the area).
    inputs = [
        d / n for n in ("classes.tif", "layers.json", "settlements.parquet", "web/change.geojson")
    ]
    return json.dumps(
        [SITREP_MAPS_VERSION]
        + [[p.name, p.stat().st_mtime_ns] for p in inputs if p.exists()]
        + [ids.get("flowpaths", [])]
    )


def _sitrep_maps(d: Path, payload: dict, f: Any) -> dict:
    from dhara.sitrep.facts import build
    from dhara.sitrep.render import _maps

    key = _maps_key(d, payload["ids"])
    cache = d / "sitrep_maps.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("key") == key:
            return cached["maps"]
    m = _maps(d, f if f is not None else build(d))
    cache.write_text(json.dumps({"key": key, "maps": m}))
    return m


@app.get("/api/jobs/{job_id}/sitrep/data")
def job_sitrep_data(job_id: str, maps: str = "") -> dict:
    """Facts, tables and map images for the situation report. `maps` (comma list, e.g.
    "overview,satellite_pair") limits the images returned; empty returns all."""
    d = _job_dir(job_id)
    payload, f = _sitrep_facts(d)
    m = _sitrep_maps(d, payload, f)
    if maps:
        wanted = set(maps.split(","))
        m = {k: v for k, v in m.items() if k in wanted}
    layers = json.loads((d / "layers.json").read_text()) if (d / "layers.json").exists() else {}
    img = layers.get("imagery") or layers.get("imagery_radar")
    return {
        "job_id": job_id,
        "facts": payload["facts"],
        "values": payload["values"],
        "tables": payload["tables"],
        "maps": m,
        # Which images the satellite-pair figure shows, so the page can label it correctly.
        "imagery": {
            "kind": img.get("kind"),
            "sensor": img.get("sensor"),
            "before": img.get("before", {}).get("date"),
            "after": img.get("after", {}).get("date"),
        }
        if img
        else None,
    }


class ProfileRequest(BaseModel):
    path: list[list[float]] = Field(description="List of [lng, lat] coordinates in EPSG:4326")
    step_m: float = Field(default=30.0, description="Sampling step in meters")


@app.post("/api/jobs/{job_id}/profile")
def get_elevation_profile(job_id: str, req: ProfileRequest):
    """Elevation profile along a route (EPSG:4326), sampled from Copernicus DEM GLO-30 around
    the route itself, so routes that leave the analysis area are covered too."""
    d = _job_dir(job_id)
    if (not req.path or len(req.path) < 2) and (d / "profile.json").exists():
        return json.loads((d / "profile.json").read_text())
    from dhara.terrain.profile import sample_profile

    return sample_profile(req.path, step_m=req.step_m)


# Report PDFs are built in the background as soon as an analysis finishes, so the download is
# ready by the time someone asks for it. State per (job, language): building / failed.
_report_state: dict[tuple[str, str], str] = {}
_report_locks: dict[tuple[str, str], threading.Lock] = {}
_report_guard = threading.Lock()


def _report_key(d: Path, payload: dict, lang: str) -> str:
    return json.dumps(
        [
            lang,
            get_settings().public_url,
            _facts_inputs_key(d),
            _maps_key(d, payload["ids"]),
            (_web_dist / "index.html").stat().st_mtime_ns,
        ]
    )


def _build_report_pdf(job_id: str, lang: str) -> Path:
    """The job's one-page report PDF (job folder), rendered by headless Chromium from the report
    page this server serves; reused while its inputs and the built frontend are unchanged."""
    from dhara.sitrep.pdf import render_pdf

    if not (_web_dist / "index.html").exists():
        raise RuntimeError(
            "The frontend is not built, so the report page cannot be printed: run `pnpm build` in web/"
        )
    d = _job_dir(job_id)
    with _report_guard:
        lock = _report_locks.setdefault((job_id, lang), threading.Lock())
    with lock:
        payload, f = _sitrep_facts(d)
        _sitrep_maps(d, payload, f)  # the page asks for these; build them first
        key = _report_key(d, payload, lang)
        out, key_file = d / f"report_{lang}.pdf", d / f"report_{lang}.key"
        if out.exists() and key_file.exists() and key_file.read_text() == key:
            return out
        _report_state[(job_id, lang)] = "building"
        s = get_settings()
        query = urlencode(
            {"job": job_id, "origin": s.public_url.rstrip("/")}
            | ({"lang": "ne"} if lang == "ne" else {})
        )
        try:
            render_pdf(f"http://127.0.0.1:{s.api_port}/report?{query}", out, s.api_port)
        except Exception as e:
            _report_state[(job_id, lang)] = f"failed: {e}"[:300]
            raise
        key_file.write_text(key)
        _report_state.pop((job_id, lang), None)
        return out


def _prebuild_reports(job_id: str) -> None:
    for lang in ("en", "ne"):
        try:
            _build_report_pdf(job_id, lang)
            _report_state.pop((job_id, lang), None)
        except Exception as e:  # recorded for the status endpoint, so the UI can say why
            _report_state[(job_id, lang)] = f"failed: {type(e).__name__}: {e}"[:300]


@app.get("/api/jobs/{job_id}/report/status")
def job_report_status(job_id: str) -> dict[str, str]:
    """Per language: ready, building, failed: <why>, or missing (built on first download)."""
    d = _job_dir(job_id)
    out: dict[str, str] = {}
    for lang in ("en", "ne"):
        state = _report_state.get((job_id, lang))
        if state:
            out[lang] = state
            continue
        pdf, key_file = d / f"report_{lang}.pdf", d / f"report_{lang}.key"
        ready = False
        if pdf.exists() and key_file.exists() and (_web_dist / "index.html").exists():
            cache = d / "sitrep_facts.json"
            with contextlib.suppress(Exception):
                payload = json.loads(cache.read_text())["payload"]
                ready = key_file.read_text() == _report_key(d, payload, lang)
        out[lang] = "ready" if ready else "missing"
    return out


@app.post("/api/jobs/{job_id}/report/build")
def job_report_build(job_id: str) -> dict[str, str]:
    """Start building the report PDFs in the background (no-op when they are ready or building)."""
    _job_dir(job_id)
    status = job_report_status(job_id)
    if any(v == "missing" or v.startswith("failed") for v in status.values()):
        for lang in ("en", "ne"):
            if status[lang] != "ready":
                _report_state[(job_id, lang)] = "building"
        threading.Thread(target=_prebuild_reports, args=(job_id,), daemon=True).start()
    return job_report_status(job_id)


@app.get("/api/jobs/{job_id}/report.pdf")
def job_report_pdf(job_id: str, lang: Literal["en", "ne"] = "en") -> Response:
    """The one-page situation report as PDF (pre-built when the analysis finished; otherwise built
    now). File name: Dhara_SitRep_<event day>_<place>_<job id>[_NE].pdf."""
    from dhara.sitrep.pdf import ascii_filename, report_filename

    try:
        out = _build_report_pdf(job_id, lang)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"The report PDF could not be built: {e}") from e
    facts = _sitrep_facts(_job_dir(job_id))[0]["facts"]
    name = report_filename(facts, job_id, lang)
    disposition = f"attachment; filename=\"{ascii_filename(facts, job_id, lang)}.pdf\"; filename*=UTF-8''{quote(name)}.pdf"
    return FileResponse(
        out, media_type="application/pdf", headers={"Content-Disposition": disposition}
    )


def _job_dir(job_id: str) -> Path:
    if not job_id.replace("_", "").isalnum():
        raise HTTPException(400, "bad job id")
    d = get_settings().jobs_dir / job_id
    if not d.is_dir():
        raise HTTPException(404, "job not found")
    return d


# Each settlement's road route to its hospital (before/after) is ~95 % of the settlements layer;
# the map loads the layer without them and fetches one route when a settlement is selected.
ROUTE_PROPS = ("hospital_path_pre", "hospital_path_post", "hospital_breaks")


def _settlements_slim(d: Path) -> Path:
    """web/settlements.geojson without the per-settlement routes (EPSG:4326), cached by mtime."""
    src = d / "web" / "settlements.geojson"
    out = d / "cache" / "settlements_slim.geojson"
    if out.exists() and out.stat().st_mtime_ns >= src.stat().st_mtime_ns:
        return out
    gj = json.loads(src.read_text())
    for ft in gj.get("features", []):
        props = ft.get("properties") or {}
        for k in ROUTE_PROPS[:2]:
            props.pop(k, None)
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(gj, separators=(",", ":")))
    tmp.replace(out)
    return out


@functools.lru_cache(maxsize=8)
def _route_index(
    path: str, mtime_ns: int
) -> tuple[dict[str, dict], list[tuple[float, float, dict]]]:
    """Routes by OSM id, plus (lon, lat, route) for a nearest-point lookup (EPSG:4326)."""
    gj = json.loads(Path(path).read_text())
    by_id: dict[str, dict] = {}
    points: list[tuple[float, float, dict]] = []
    for ft in gj.get("features", []):
        props = ft.get("properties") or {}
        route = {k: props.get(k) for k in ROUTE_PROPS}
        if props.get("osm_id"):
            by_id[str(props["osm_id"])] = route
        coords = (ft.get("geometry") or {}).get("coordinates")
        if coords:
            points.append((float(coords[0]), float(coords[1]), route))
    return by_id, points


@app.get("/api/jobs/{job_id}/settlements/route")
def settlement_route(
    job_id: str, osm_id: str = "", lon: float | None = None, lat: float | None = None
) -> dict:
    """One settlement's road route to its hospital before and after the event, as stored in the
    settlements layer (JSON-encoded [lon, lat] lists, EPSG:4326). Looked up by OSM id, else the
    settlement point nearest to lon/lat (EPSG:4326) within about 50 m."""
    d = _job_dir(job_id)
    src = d / "web" / "settlements.geojson"
    if not src.exists():
        raise HTTPException(404, "this job has no settlements layer")
    by_id, points = _route_index(str(src), src.stat().st_mtime_ns)
    if osm_id and osm_id in by_id:
        return by_id[osm_id]
    if lon is not None and lat is not None and points:
        x, y, route = min(points, key=lambda q: (q[0] - lon) ** 2 + (q[1] - lat) ** 2)
        if (x - lon) ** 2 + (y - lat) ** 2 < 0.0005**2:
            return route
    raise HTTPException(404, "settlement not found in this job")


def _gzipped(p: Path, d: Path) -> Path:
    """A gzip copy of a job file in <job>/cache, made once; layers never change after a job ends."""
    out = d / "cache" / "gz" / (str(p.relative_to(d.resolve())).replace("/", "__") + ".gz")
    if out.exists() and out.stat().st_mtime_ns >= p.stat().st_mtime_ns:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with p.open("rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    tmp.replace(out)
    return out


@app.get("/api/jobs/{job_id}/files/{name:path}")
def job_file(job_id: str, name: str, request: Request, slim: bool = False):
    d = _job_dir(job_id)
    p = (d / name).resolve()
    if d.resolve() not in p.parents:
        raise HTTPException(400, "bad path")
    ok = name in ALLOWED_FILES or (
        p.parent in (d.resolve() / "web", d.resolve()) and p.suffix in WEB_EXT
    )
    if not ok or not p.is_file():
        raise HTTPException(404, "file not available")
    if slim and name == "web/settlements.geojson":
        p = _settlements_slim(d)
    media = {
        ".geojson": "application/geo+json",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".tif": "image/tiff",
    }.get(p.suffix, "application/octet-stream")
    big_text = p.suffix in (".geojson", ".json") and p.stat().st_size > 256_000
    if big_text and "gzip" in request.headers.get("accept-encoding", ""):
        return FileResponse(
            _gzipped(p.resolve(), d),
            media_type=media,
            headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
        )
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
            "report_en.pdf",  # one-page report, once rendered via /report.pdf
            "report_ne.pdf",
            "provenance.json",
            "facts.json",
            "summary.json",
            "assistant_log.jsonl",
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


class SelectedEntityContext(BaseModel):
    id: str = ""
    kind: Literal[
        "settlement", "bridge", "hospital", "building", "road", "flow-settlement", "point"
    ]
    coordinates: tuple[float, float]


class ChatRequest(BaseModel):
    message: str = Field(description="Question about the disaster situation in any language")
    lang: str = Field(
        default="en", description="Target response language (e.g., 'en', 'ne', 'hi', 'es', 'fr')"
    )
    entity: SelectedEntityContext | None = Field(
        default=None, description="Selected map feature context for resolving same-name features"
    )


class ExplainAssetRequest(BaseModel):
    asset_type: Literal["bridge", "settlement", "road"] = Field(description="Type of OSM asset")
    asset_name: str = Field(description="Name or ID of the asset")
    lang: str = Field(default="en", description="Target explanation language")


@app.get("/api/jobs/{job_id}/assistant/suggested")
def assistant_suggested(job_id: str, lang: str = "en") -> list[dict]:
    """The assistant's suggested questions with their answers, computed from this job's facts
    (templates, no language model), for instant display."""
    from dhara.assistant.service import suggested_answers

    return suggested_answers(_job_dir(job_id), lang=lang)


@app.post("/api/jobs/{job_id}/chat")
def chat_with_assistant(job_id: str, req: ChatRequest) -> dict:
    """Ask analytical questions about the disaster in any language, grounded in verified facts."""
    from dhara.assistant.service import ask_assistant

    d = _job_dir(job_id)
    return ask_assistant(
        d,
        req.message,
        lang=req.lang,
        selected_entity=req.entity.model_dump() if req.entity else None,
    )


@app.post("/api/jobs/{job_id}/explain-asset")
def explain_asset(job_id: str, req: ExplainAssetRequest) -> dict:
    """Explain why a specific bridge or settlement is cut off or damaged using radar/terrain physics."""
    from dhara.assistant.service import explain_asset_impact

    d = _job_dir(job_id)
    return explain_asset_impact(d, req.asset_type, req.asset_name, lang=req.lang)


class _SpaFiles(StaticFiles):
    """The built frontend; unknown non-API paths (e.g. /labs/reports) get index.html so the
    client-side router can open them."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            if e.status_code != 404 or path.startswith("api/"):
                raise
            return await super().get_response("index.html", scope)


# Mount compiled frontend if available (educational prototype UI)
if _web_dist.exists() and (_web_dist / "index.html").exists():
    app.mount("/", _SpaFiles(directory=_web_dist, html=True), name="static")
