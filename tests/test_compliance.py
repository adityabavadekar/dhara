"""Data-rule guardrails (PLAN.md §6). A failure here means a disqualification risk."""

import re
from datetime import date
from pathlib import Path

import pytest

from dhara.adapters import ohsome
from dhara.adapters.allowlist import ComplianceError, check_request
from dhara.config import OSM_SNAPSHOT_MAX

PKG = Path(__file__).resolve().parents[1] / "src" / "dhara"
PY_FILES = sorted(PKG.rglob("*.py"))

# Reference maps / disallowed sources. Word-boundary regexes, case-insensitive.
BLOCKED = [
    r"\bvalidation\b",
    r"emergency\.copernicus",
    r"\bemsr\d*",
    r"\bcems\b",
    r"unosat",
    r"hotosm",
    r"\bfair\.",
    r"global-flood",
    r"\bgfm\b",
    r"\bopera\b",
    r"dswx",
    r"\bdist-(s1|alert)",
    r"worldpop",
    r"worldcover",
    r"\bjrc\b",
    r"\bmerit\b",
    r"overpass",
    r"\bhand_merit",
]


def test_package_has_files():
    assert PY_FILES, "pipeline package not found"


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_reference_sources_in_pipeline(path: Path):
    text = path.read_text().lower()
    hits = [pat for pat in BLOCKED if re.search(pat, text)]
    assert not hits, f"{path} references blocked source(s): {hits}"


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(PKG)))
def test_network_only_via_allowlist(path: Path):
    if path.name == "allowlist.py":
        return
    text = path.read_text()
    pat = r"^\s*(import|from)\s+(httpx|requests|aiohttp|urllib3|urllib\.request|http\.client)\b"
    assert not re.search(pat, text, re.M), f"{path} does network I/O outside allowlist.py"


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_post_snapshot_dates_near_ohsome(path: Path):
    text = path.read_text()
    if "ohsome" not in text.lower():
        return
    for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", text):
        dt = date.fromisoformat(d)
        # The v1 shutdown date is mentioned in a docstring and is not a query timestamp.
        if dt > OSM_SNAPSHOT_MAX and d != "2026-11-30":
            pytest.fail(f"{path} contains date {d} after the OSM snapshot limit")


def test_snapshot_never_after_limit():
    assert ohsome.snapshot_date(date(2026, 8, 26)) == OSM_SNAPSHOT_MAX
    assert ohsome.snapshot_date(date(2030, 1, 1)) == OSM_SNAPSHOT_MAX
    assert ohsome.snapshot_date(date(2021, 7, 15)) == date(2021, 7, 14)


def test_fetch_layer_refuses_late_snapshot():
    with pytest.raises(ComplianceError):
        ohsome.fetch_layer("buildings", (85.3, 28.1, 85.31, 28.11), date(2026, 8, 1))


@pytest.mark.parametrize(
    "url,kw",
    [
        ("https://api.ohsome.org/v1/elements/geometry", {"data": {"time": "2026-08-27"}}),
        (
            "https://api.ohsome.org/v1/elements/geometry",
            {"data": {"time": "2026-07-01,2026-09-01"}},
        ),
        ("https://api.ohsome.org/v1/elements/geometry", {"data": {"filter": "building=*"}}),
        (
            "https://api.heigit.org/ohsome-api/v2-rc/extraction/features.parquet",
            {"json": {"time": "latest"}},
        ),
        (
            "https://api.heigit.org/ohsome-api/v2-rc/extraction/features.parquet",
            {"json": {"time": {"start": "2026-07-01", "end": "2026-08-30"}}},
        ),
    ],
)
def test_ohsome_late_or_missing_time_rejected(url, kw):
    with pytest.raises(ComplianceError):
        check_request(url, **kw)


@pytest.mark.parametrize(
    "url",
    [
        "https://mapping.emergency.copernicus.eu/activations/EMSR927",
        "https://unosat.org/products/",
        "https://hotosm.org",
        "https://overpass-api.de/api/interpreter",
        "https://global-flood.emergency.copernicus.eu/",
        "https://hub.worldpop.org/",
        "http://stac.dataspace.copernicus.eu/v1/search",
    ],
)
def test_blocked_hosts(url):
    with pytest.raises(ComplianceError):
        check_request(url)
