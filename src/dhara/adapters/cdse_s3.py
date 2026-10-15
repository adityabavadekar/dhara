"""Read original Sentinel-1 GRD files from the CDSE S3 store (eodata.dataspace.copernicus.eu).

Small files (annotation XMLs) come over https with AWS Signature V4, through
`allowlist.guarded_request`; the measurement images are cloud-optimised GeoTIFFs read by GDAL
(`/vsis3/`, only the needed window) through `allowlist.guarded_raster_open`. Both accept only
Sentinel-1 IW GRD product files. Needs CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY (CDSE S3 keys
manager). No geodata is transformed here.
"""

import hashlib
import hmac
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from urllib.parse import quote, urlparse

from dhara.adapters.allowlist import guarded_raster_open, guarded_request
from dhara.config import get_settings

ENDPOINT = "eodata.dataspace.copernicus.eu"
REGION = "default"


class S3CredentialsMissing(RuntimeError):
    pass


def _keys() -> tuple[str, str]:
    s = get_settings()
    if not (s.cdse_s3_access_key and s.cdse_s3_secret_key):
        raise S3CredentialsMissing(
            "Reading original Sentinel-1 products needs CDSE S3 keys: create them at "
            "https://eodata-s3keysmanager.dataspace.copernicus.eu and set CDSE_S3_ACCESS_KEY and "
            "CDSE_S3_SECRET_KEY in .env (or use s1_source='sentinelhub')."
        )
    return s.cdse_s3_access_key.get_secret_value(), s.cdse_s3_secret_key.get_secret_value()


def gdal_env() -> dict[str, str]:
    """GDAL options for reading CDSE S3 with `/vsis3/`. rasterio refuses AWS credentials as Env
    options, so the keys go into the process environment (where GDAL reads them); everything else
    is returned for `rasterio.Env(**gdal_env())`."""
    access, secret = _keys()
    os.environ["AWS_ACCESS_KEY_ID"] = access
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret
    return {
        "AWS_S3_ENDPOINT": ENDPOINT,
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_HTTPS": "YES",
        "AWS_REGION": REGION,
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        # The store sometimes answers 500 / "DB scan query failed"; retry before giving up.
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
    }


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _signed_headers(path: str, extra: dict[str, str]) -> dict[str, str]:
    """AWS Signature V4 headers for a GET of `path` (path-style: /<bucket>/<key>) with no body."""
    access, secret = _keys()
    now = datetime.now(UTC)
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    payload = hashlib.sha256(b"").hexdigest()
    headers = {"host": ENDPOINT, "x-amz-content-sha256": payload, "x-amz-date": amz_date}
    headers |= {k.lower(): v for k, v in extra.items()}
    names = sorted(headers)
    canonical = "\n".join(
        [
            "GET",
            quote(path, safe="/-_.~"),
            "",
            "".join(f"{k}:{headers[k].strip()}\n" for k in names),
            ";".join(names),
            payload,
        ]
    )
    scope = f"{day}/{REGION}/s3/aws4_request"
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    k = _sign(_sign(_sign(_sign(f"AWS4{secret}".encode(), day), REGION), "s3"), "aws4_request")
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access}/{scope}, SignedHeaders={';'.join(names)}, "
        f"Signature={signature}"
    )
    headers.pop("host")
    return headers


def _path(s3_url: str) -> str:
    """s3://eodata/Sentinel-1/... -> /eodata/Sentinel-1/..."""
    u = urlparse(s3_url)
    if u.scheme != "s3" or u.netloc != "eodata":
        raise ValueError(f"Not a CDSE eodata S3 URL: {s3_url}")
    return f"/eodata{u.path}"


def get_bytes(s3_url: str) -> bytes:
    """Download one small product file (annotation XML, manifest)."""
    path = _path(s3_url)
    r = guarded_request(
        "GET", f"https://{ENDPOINT}{path}", headers=_signed_headers(path, {}), timeout_s=120
    )
    if r.status_code != 200:
        raise RuntimeError(f"CDSE S3 returned {r.status_code} for {path}: {r.text[:300]}")
    return r.content


@contextmanager
def open_measurement(s3_url: str):
    """Open a measurement GeoTIFF for windowed reads (GDAL /vsis3, range requests only)."""
    import rasterio

    with (
        rasterio.Env(**gdal_env(), CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff"),
        guarded_raster_open("/vsis3" + _path(s3_url)) as ds,
    ):
        yield ds
