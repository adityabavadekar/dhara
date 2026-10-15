"""Pre-event OpenStreetMap via the ohsome API. The snapshot is always ≤ OSM_SNAPSHOT_MAX.

Two routes, both through the allowlist:
- v2 (`api.heigit.org/ohsome-api/v2-rc`, needs a free OHSOME_API_KEY): GeoParquet snapshot extraction.
- v1 fallback (`api.ohsome.org/v1`, no key, shuts down 2026-11-30): `elements/geometry` returns
  403 now, so we use `elementsFullHistory/geometry` over [T-1d, T] and keep the versions still
  valid at T. This is slow.
"""

import io
import json
from datetime import UTC, date, datetime, time, timedelta

import geopandas as gpd
import pyarrow.parquet as pq
import shapely

from dhara.adapters.allowlist import ComplianceError, guarded_request
from dhara.config import OSM_SNAPSHOT_MAX, get_settings
from dhara.timeutil import event_day_utc, iso_utc

OHSOME_V2_URL = "https://api.heigit.org/ohsome-api/v2-rc/extraction/features.parquet"
OHSOME_V1_FH_URL = "https://api.ohsome.org/v1/elementsFullHistory/geometry"

# Feature layers used by the pipeline (ohsome filter language).
LAYERS: dict[str, str] = {
    "buildings": "building=* and building!=no and geometry:polygon",
    "roads": "highway=* and type:way and geometry:line",
    "places": "place in (city, town, village, hamlet, isolated_dwelling, locality, neighbourhood)",
    "health": "amenity in (hospital, clinic, doctors) or healthcare=*",
}


class OhsomeError(RuntimeError):
    pass


def snapshot_date(event: date | datetime) -> date:
    """OSM snapshot (00:00Z) for an event: the UTC day before the event, never after OSM_SNAPSHOT_MAX."""
    return min(event_day_utc(event) - timedelta(days=1), OSM_SNAPSHOT_MAX)


def _empty() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"osm_id": [], "tags": []}, geometry=[], crs="EPSG:4326")


def _fetch_v2(flt: str, bbox: tuple[float, float, float, float], snap: datetime, key: str):
    resp = guarded_request(
        "POST",
        OHSOME_V2_URL,
        json={"aoi": list(bbox), "filter": flt, "time": iso_utc(snap)},
        headers={"Authorization": key},
    )
    if resp.status_code != 200:
        raise OhsomeError(f"ohsome v2 request failed ({resp.status_code}): {resp.text[:300]}")
    if not resp.content or len(resp.content) < 4:
        return _empty()
    try:
        tbl = pq.read_table(io.BytesIO(resp.content))
    except Exception:
        return _fetch_v1(flt, bbox, snap)
    if tbl.num_rows == 0:
        return _empty()
    df = tbl.to_pandas()
    # Defence in depth: never keep a version uploaded after the snapshot.
    df = df[df["edit_timestamp"] <= snap]
    return gpd.GeoDataFrame(
        {
            "osm_id": df["osm_type"] + "/" + df["osm_id"].astype(str),
            "tags": [json.dumps(dict(t)) for t in df["tags"]],
        },
        geometry=shapely.from_wkb(df["geom"].to_numpy()),
        crs="EPSG:4326",
    )


def _fetch_v1(flt: str, bbox: tuple[float, float, float, float], snap: datetime):
    t_end = iso_utc(snap)
    t_start = iso_utc(snap - timedelta(days=1))
    resp = guarded_request(
        "POST",
        OHSOME_V1_FH_URL,
        data={
            "bboxes": ",".join(f"{v:.6f}" for v in bbox),
            "time": f"{t_start},{t_end}",
            "filter": flt,
            "properties": "tags",
        },
    )
    if resp.status_code != 200:
        raise OhsomeError(f"ohsome v1 request failed ({resp.status_code}): {resp.text[:300]}")
    feats = [f for f in resp.json().get("features", []) if f["properties"].get("@validTo") == t_end]
    if not feats:
        return _empty()
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    gdf["osm_id"] = gdf["@osmId"]
    tag_cols = [c for c in gdf.columns if not c.startswith("@") and c not in ("geometry", "osm_id")]
    gdf["tags"] = [
        json.dumps({k: v for k, v in row.items() if isinstance(v, str)})
        for row in gdf[tag_cols].to_dict("records")
    ]
    return gdf[["osm_id", "tags", "geometry"]]


def fetch_layer(
    layer: str, bbox: tuple[float, float, float, float], snapshot: date
) -> gpd.GeoDataFrame:
    """Fetch one OSM layer as it was at 00:00 UTC on `snapshot`. bbox and output: EPSG:4326."""
    if snapshot > OSM_SNAPSHOT_MAX:
        raise ComplianceError(f"OSM snapshot {snapshot} is after {OSM_SNAPSHOT_MAX}")
    if layer not in LAYERS:
        raise ValueError(f"unknown OSM layer {layer!r}")
    snap = datetime.combine(snapshot, time(0), tzinfo=UTC)
    key = get_settings().ohsome_api_key
    if key:
        try:
            return _fetch_v2(LAYERS[layer], bbox, snap, key.get_secret_value())
        except Exception:
            return _fetch_v1(LAYERS[layer], bbox, snap)
    return _fetch_v1(LAYERS[layer], bbox, snap)
