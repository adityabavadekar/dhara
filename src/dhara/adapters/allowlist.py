"""Single network gate for the pipeline.

Every outbound HTTP request in `dhara` must go through `guarded_request()`. It rejects any host,
collection, DEM instance or OSM timestamp outside the competition data rules.
"""

import re
import threading
from datetime import date
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from dhara.config import OSM_SNAPSHOT_MAX, get_settings

# Timeout raised by guarded_request(), for callers that retry elsewhere (no httpx import needed).
RequestTimeout = httpx.TimeoutException


class ComplianceError(RuntimeError):
    """Raised when a request would use data outside the allowed inputs."""


ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "stac.dataspace.copernicus.eu",  # CDSE STAC catalogue (scene discovery)
        "identity.dataspace.copernicus.eu",  # CDSE OAuth token
        "sh.dataspace.copernicus.eu",  # CDSE Sentinel Hub Process API (S1, S2, COP-DEM)
        "eodata.dataspace.copernicus.eu",  # CDSE S3: original Sentinel-1 GRD files only (see below)
        "api.ohsome.org",  # ohsome v1 (OSM history), snapshot pinned ≤ OSM_SNAPSHOT_MAX
        "api.heigit.org",  # ohsome v2 (path /ohsome-api/ only), snapshot pinned the same way
    }
)

# Assistant language-model hosts. Not data inputs: they only phrase answers from a finished job's
# facts. Each is accepted only when `assistant_llm` selects that provider (default "off").
LLM_HOSTS: dict[str, str] = {
    "api.groq.com": "groq",
    "generativelanguage.googleapis.com": "gemini",
    "api.openai.com": "openai",
    "openrouter.ai": "openrouter",
}

# Raster (GDAL) sources, opened only through `guarded_raster_open()`. Copernicus DEM GLO-30 COG
# tiles only: CDSE S3 (eodata) and the official open GLO-30 COG distribution on AWS Open Data.
ALLOWED_RASTER_PREFIXES: tuple[str, ...] = (
    "/vsis3/eodata/auxdata/CopDEM_COG/copernicus-dem-30m/",
    "/vsicurl/https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/",
    S1_GRD_RASTER_PREFIX := "/vsis3/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/",
)
# CDSE S3 objects readable over https: original Sentinel-1 IW GRD products (cloud-optimised SAFE),
# their measurement images and annotation XMLs only. Nothing else in the bucket.
S1_GRD_S3_PREFIX = "/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/"
_S1_GRD_FILE = re.compile(
    r"^/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/\d{4}/\d{2}/\d{2}/S1[A-D]_IW_GRDH_1S[DS][VH]_[0-9T_A-F]+_COG\.SAFE/"
    r"(manifest\.safe|annotation/(calibration/)?[a-z0-9-]+\.xml|measurement/[a-z0-9-]+\.tiff)$"
)

# CDSE STAC collection ids.
ALLOWED_STAC_COLLECTIONS: frozenset[str] = frozenset(
    {"sentinel-1-grd", "sentinel-1-slc", "sentinel-2-l2a", "sentinel-2-l1c"}
)
# Sentinel Hub data types.
ALLOWED_SH_TYPES: frozenset[str] = frozenset(
    {"sentinel-1-grd", "sentinel-2-l2a", "sentinel-2-l1c", "dem"}
)
ALLOWED_DEM_INSTANCES: frozenset[str] = frozenset({"COPERNICUS_30"})

_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _check_stac(url: str, payload: dict[str, Any] | None) -> None:
    path = urlparse(url).path
    cols: list[str] = []
    if payload and "collections" in payload:
        cols = list(payload["collections"])
    m = re.search(r"/collections/([^/]+)", path)
    if m:
        cols.append(m.group(1))
    if path.rstrip("/").endswith("/search") and not cols:
        raise ComplianceError("STAC search without explicit collections is not allowed")
    bad = [c for c in cols if c not in ALLOWED_STAC_COLLECTIONS]
    if bad:
        raise ComplianceError(f"STAC collection(s) not allowed: {bad}")


def _check_sh(url: str, payload: dict[str, Any] | None) -> None:
    path = urlparse(url).path
    if path.endswith("/process"):
        if not payload:
            raise ComplianceError("Sentinel Hub process request without payload")
        data = payload.get("input", {}).get("data", [])
        if not data:
            raise ComplianceError("Sentinel Hub process request without input data")
        for d in data:
            t = d.get("type")
            if t not in ALLOWED_SH_TYPES:
                raise ComplianceError(f"Sentinel Hub data type not allowed: {t!r}")
            if t == "dem":
                inst = d.get("dataFilter", {}).get("demInstance")
                if inst not in ALLOWED_DEM_INSTANCES:
                    raise ComplianceError(f"DEM instance not allowed: {inst!r}")
    elif "/catalog/" in path:
        if payload:
            bad = [c for c in payload.get("collections", []) if c not in ALLOWED_SH_TYPES]
            if bad:
                raise ComplianceError(f"Sentinel Hub catalog collection(s) not allowed: {bad}")
    else:
        raise ComplianceError(f"Sentinel Hub endpoint not allowed: {path}")


def check_ohsome_time(value: str) -> None:
    """Every date in an ohsome `time` parameter must be ≤ OSM_SNAPSHOT_MAX."""
    dates = _ISO_DATE.findall(value)
    if not dates or "latest" in value:
        raise ComplianceError(f"ohsome time must be explicit ISO date(s), got {value!r}")
    for d in dates:
        if date.fromisoformat(d) > OSM_SNAPSHOT_MAX:
            raise ComplianceError(
                f"ohsome timestamp {d} is after the allowed pre-event snapshot "
                f"{OSM_SNAPSHOT_MAX.isoformat()}"
            )


def _check_ohsome(
    url: str,
    params: dict[str, Any] | None,
    data: dict[str, Any] | None,
    body: dict[str, Any] | None,
) -> None:
    path = urlparse(url).path
    if urlparse(url).hostname == "api.heigit.org" and not path.startswith("/ohsome-api/"):
        raise ComplianceError(f"api.heigit.org path not allowed: {path}")
    times: list[str] = []
    for src in (params or {}, data or {}, body or {}):
        if "time" in src:
            t = src["time"]
            times += [str(v) for v in t.values()] if isinstance(t, dict) else [str(t)]
    times += parse_qs(urlparse(url).query).get("time", [])
    if not times and not path.rstrip("/").endswith(("/metadata", "/health")):
        raise ComplianceError("ohsome request without an explicit pinned `time` is not allowed")
    for t in times:
        check_ohsome_time(t)


def check_request(
    url: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """Raise ComplianceError unless the request targets an allowed source with allowed parameters."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ComplianceError(f"Only https is allowed: {url}")
    host = parsed.hostname or ""
    if host in LLM_HOSTS:
        settings = get_settings()
        selected = settings.llm_provider
        if LLM_HOSTS[host] != selected and LLM_HOSTS[host] not in settings.llm_fallback_providers:
            raise ComplianceError(
                f"Assistant LLM host {host!r} is blocked: the active provider is {selected!r}. "
                f"Set ASSISTANT_LLM={LLM_HOSTS[host]} and its API key to enable it."
            )
        return
    if host not in ALLOWED_HOSTS:
        raise ComplianceError(f"Host not in allowlist: {host!r}")
    if host == "stac.dataspace.copernicus.eu":
        _check_stac(url, json)
    elif host == "sh.dataspace.copernicus.eu":
        _check_sh(url, json)
    elif host in ("api.ohsome.org", "api.heigit.org"):
        _check_ohsome(url, params, data, json)
    elif host == "eodata.dataspace.copernicus.eu" and not _S1_GRD_FILE.match(parsed.path):
        raise ComplianceError(
            f"Only original Sentinel-1 IW GRD files may be read from CDSE S3: {parsed.path}"
        )


_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                timeout=get_settings().dhara_http_timeout_s,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
    return _client


def guarded_request(
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    timeout_s: float | None = None,
) -> httpx.Response:
    """The only way pipeline code may reach the network. Retries 429/5xx with backoff."""
    import time

    check_request(url, json=json, params=params, data=data)
    client = _get_client()
    last: httpx.Response | None = None
    for attempt in range(retries + 1):
        if timeout_s is not None:
            resp = client.request(
                method,
                url,
                json=json,
                params=params,
                data=data,
                headers=headers,
                timeout=timeout_s,
            )
        else:
            resp = client.request(method, url, json=json, params=params, data=data, headers=headers)
        if resp.status_code not in (429, 500, 502, 503, 504):
            return resp
        last = resp
        if attempt == retries:
            return resp
        wait = float(resp.headers.get("retry-after", 2 ** (attempt + 1)))
        time.sleep(min(wait, 30.0))
    assert last is not None
    return last


def check_raster_source(path: str) -> None:
    if not path.startswith(ALLOWED_RASTER_PREFIXES) or ".." in path:
        raise ComplianceError(f"Raster source not in allowlist: {path}")
    if path.startswith(S1_GRD_RASTER_PREFIX):
        if not _S1_GRD_FILE.match(path.removeprefix("/vsis3")) or "/measurement/" not in path:
            raise ComplianceError(f"Only Sentinel-1 GRD measurement images allowed: {path}")
        return
    if "Copernicus_DSM_COG_10_" not in path:
        raise ComplianceError(f"Only Copernicus DEM GLO-30 (10 arc-second) tiles allowed: {path}")


def guarded_raster_open(path: str):
    """Open a remote raster with rasterio after the allowlist check. Returns a dataset (context manager)."""
    import rasterio

    check_raster_source(path)
    return rasterio.open(path)
