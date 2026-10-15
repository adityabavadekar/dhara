import pytest

from dhara.adapters.allowlist import ComplianceError, check_request

SH = "https://sh.dataspace.copernicus.eu/api/v1/process"
STAC = "https://stac.dataspace.copernicus.eu/v1/search"


def _sh(*data):
    return {"input": {"bounds": {}, "data": list(data)}, "evalscript": ""}


def test_allowed_requests_pass():
    check_request(STAC, json={"collections": ["sentinel-1-grd"]})
    check_request(SH, json=_sh({"type": "sentinel-1-grd"}))
    check_request(SH, json=_sh({"type": "dem", "dataFilter": {"demInstance": "COPERNICUS_30"}}))
    check_request("https://api.ohsome.org/v1/elements/geometry", data={"time": "2026-07-27"})
    check_request(
        "https://api.heigit.org/ohsome-api/v2-rc/extraction/features.parquet",
        json={"time": "2026-07-27T00:00:00Z"},
    )


@pytest.mark.parametrize(
    "url,payload",
    [
        (STAC, {"collections": ["opera-dswx"]}),
        (STAC, {}),
        (SH, _sh({"type": "landsat-ot-l2"})),
        (SH, _sh({"type": "dem", "dataFilter": {"demInstance": "MAPZEN"}})),
        (SH, _sh({"type": "dem"})),
        (SH, _sh()),
        ("https://sh.dataspace.copernicus.eu/api/v1/byoc/collections", {}),
        ("https://api.heigit.org/ors/v2/directions", {"time": "2026-07-27"}),
    ],
)
def test_disallowed_requests_fail(url, payload):
    with pytest.raises(ComplianceError):
        check_request(url, json=payload)


def test_llm_hosts_blocked_unless_selected(monkeypatch):
    from dhara.config import get_settings

    monkeypatch.setenv("ASSISTANT_LLM", "off")
    get_settings.cache_clear()
    try:
        with pytest.raises(ComplianceError, match="ASSISTANT_LLM"):
            check_request("https://api.groq.com/openai/v1/chat/completions")
        monkeypatch.setenv("ASSISTANT_LLM", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        get_settings.cache_clear()
        check_request("https://api.groq.com/openai/v1/chat/completions")
        with pytest.raises(ComplianceError):
            check_request("https://api.openai.com/v1/chat/completions")
        monkeypatch.setenv("ASSISTANT_LLM", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        get_settings.cache_clear()
        check_request("https://openrouter.ai/api/v1/chat/completions")
        with pytest.raises(ComplianceError):
            check_request("https://api.groq.com/openai/v1/chat/completions")
    finally:
        get_settings.cache_clear()


def test_zero_retries_returns_rate_limit_without_sleep(monkeypatch):
    import httpx

    from dhara.adapters import allowlist

    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(429, headers={"retry-after": "30"}))
    )
    monkeypatch.setattr(allowlist, "check_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(allowlist, "_get_client", lambda: client)
    monkeypatch.setattr("time.sleep", lambda *_: pytest.fail("unexpected retry delay"))
    try:
        response = allowlist.guarded_request(
            "POST", "https://api.groq.com/test", retries=0, timeout_s=1
        )
        assert response.status_code == 429
    finally:
        client.close()


S1_SAFE = (
    "/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/2026/08/28/"
    "S1D_IW_GRDH_1SDV_20260828T122141_20260828T122206_004326_007FA4_C73B_COG.SAFE/"
)


def test_cdse_s3_allows_only_sentinel1_grd_files():
    from dhara.adapters.allowlist import ComplianceError, check_raster_source, check_request

    host = "https://eodata.dataspace.copernicus.eu"
    name = "s1d-iw-grd-vv-20260828t122141-20260828t122206-004326-007fa4-001"
    for ok in (
        f"annotation/{name}.xml",
        f"annotation/calibration/calibration-{name}.xml",
        f"annotation/calibration/noise-{name}.xml",
        f"measurement/{name}.tiff",
    ):
        check_request(host + S1_SAFE + ok)
    check_raster_source("/vsis3" + S1_SAFE + f"measurement/{name}.tiff")
    for bad in (
        host + S1_SAFE + "../../../Sentinel-2/x.jp2",
        host + "/eodata/Sentinel-2/MSI/L2A/2026/08/28/x.SAFE/GRANULE/x.jp2",
        host + "/eodata/CEMS/EMSR927/footprint.json",
    ):
        with pytest.raises(ComplianceError):
            check_request(bad)
    with pytest.raises(ComplianceError):
        check_raster_source("/vsis3" + S1_SAFE + f"annotation/{name}.xml")
