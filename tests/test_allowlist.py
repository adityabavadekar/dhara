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
