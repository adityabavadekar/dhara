"""Sentinel-2 L2A via CDSE: candidate dates (STAC), AOI cloud fraction, and AOI-only reflectance.

Cloud fraction is measured over the AOI (not the scene) from the L2A Scene Classification
(SCL) band, read at 60 m to stay cheap.
"""

import json
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
from pydantic import BaseModel

from dhara.adapters.cdse import Grid, sh_process, stac_search
from dhara.geo import AOI
from dhara.timeutil import event_day_utc, iso_utc

# SCL classes treated as not observable for optical evidence.
SCL_INVALID = (0, 1, 3, 8, 9, 10)  # no data, saturated, cloud shadow, cloud med/high, cirrus
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


class S2Scene(BaseModel):
    day: date
    scene_ids: list[str]
    aoi_cloud_fraction: float | None = None  # NaN-free fraction of AOI not observable
    # Same, over ground a flood or debris can reach inside the AOI (None: not measured).
    reach_cloud_fraction: float | None = None


class S2Pair(BaseModel):
    pre: S2Scene | None
    post: S2Scene | None
    candidates: list[S2Scene]


def _day_filter(day: date) -> dict:
    start = datetime.combine(day, time(0), tzinfo=UTC)
    return {
        "type": "sentinel-2-l2a",
        "dataFilter": {
            "timeRange": {"from": iso_utc(start), "to": iso_utc(start + timedelta(days=1))},
            "mosaickingOrder": "leastCC",
        },
    }


_EVAL_SCL = """//VERSION=3
function setup() { return {input: ["SCL", "dataMask"], output: {bands: 1, sampleType: "UINT8"}}; }
function evaluatePixel(s) { return [s.dataMask == 0 ? 0 : s.SCL]; }
"""

_EVAL_BANDS = """//VERSION=3
function setup() {
  return {input: [{bands: %s, units: %s}], output: {bands: %d, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) {
  if (s.dataMask == 0) { return [%s]; }
  return [%s, s.SCL];
}
"""


def candidate_days(aoi: AOI, event: date | datetime, window_days: int = 20) -> list[S2Scene]:
    """S2 L2A acquisition days intersecting the AOI within ±window_days of the event."""
    day = event_day_utc(event)
    start = datetime.combine(day - timedelta(days=window_days), time(0), tzinfo=UTC)
    end = datetime.combine(day + timedelta(days=window_days + 1), time(0), tzinfo=UTC)
    items = stac_search("sentinel-2-l2a", aoi.bounds, start, end)
    by_day: dict[date, list[str]] = {}
    for it in items:
        d = datetime.fromisoformat(it["properties"]["datetime"].replace("Z", "+00:00")).date()
        by_day.setdefault(d, []).append(it["id"])
    return [S2Scene(day=d, scene_ids=sorted(ids)) for d, ids in sorted(by_day.items())]


def cloud_mask(scene: S2Scene, grid60: Grid) -> np.ndarray:
    """True where a pixel is cloud / shadow / cirrus / no data, on the coarse grid `grid60`
    (projected, typically 60 m)."""
    scl = sh_process(grid60, [_day_filter(scene.day)], _EVAL_SCL, 1)[0]
    return np.isin(np.nan_to_num(scl, nan=0).astype(np.uint8), SCL_INVALID)


def aoi_cloud_fraction(scene: S2Scene, grid60: Grid, in_aoi: np.ndarray) -> float:
    """Fraction of AOI pixels that are cloud / shadow / cirrus / no data. grid60: coarse grid."""
    bad = cloud_mask(scene, grid60)
    return float(bad[in_aoi].mean()) if in_aoi.any() else 1.0


def choose_pair(
    candidates: list[S2Scene],
    event: date | datetime,
    *,
    max_cloud: float = 0.6,
    max_reach_cloud: float = 0.75,
    days_weight: float = 0.01,
) -> S2Pair:
    """Clearest pre (before the event day) and post (after) scenes; score = AOI cloud + 0.01·|days|.

    A scene is usable if its cloud over reachable ground is at most `max_reach_cloud` (when
    measured), else if its AOI cloud is at most `max_cloud`. Cloudy pixels are masked one by one
    when the images are used, so a scene whose valleys are partly clear still adds evidence there;
    over a large mountain area the AOI-wide limit would reject it for clouds on the ridges.
    """
    day = event_day_utc(event)

    def score(s: S2Scene) -> float:
        assert s.aoi_cloud_fraction is not None
        return s.aoi_cloud_fraction + days_weight * abs((s.day - day).days)

    def usable(s: S2Scene) -> bool:
        if s.aoi_cloud_fraction is None:
            return False
        if s.reach_cloud_fraction is not None:
            return s.reach_cloud_fraction <= max_reach_cloud
        return s.aoi_cloud_fraction <= max_cloud

    ok = [s for s in candidates if usable(s)]
    pre = [s for s in ok if s.day < day]
    post = [s for s in ok if s.day > day]
    return S2Pair(
        pre=min(pre, key=score) if pre else None,
        post=min(post, key=score) if post else None,
        candidates=candidates,
    )


def fetch_scene(scene: S2Scene, grid: Grid) -> dict[str, np.ndarray]:
    """Reflectance bands (B02,B03,B04,B08,B11,B12; 0–1) and SCL on `grid` (10 m, 20 m bands upsampled)."""
    nan = ", ".join(["NaN"] * (len(BANDS) + 1))
    vals = ", ".join(f"s.{b}" for b in BANDS)
    names = [*BANDS, "SCL", "dataMask"]
    units = ["REFLECTANCE"] * len(BANDS) + ["DN", "DN"]
    script = _EVAL_BANDS % (json.dumps(names), json.dumps(units), len(BANDS) + 1, nan, vals)
    data = [_day_filter(scene.day) | {"processing": {"upsampling": "BILINEAR"}}]
    arr = sh_process(grid, data, script, len(BANDS) + 1)
    out = {b: arr[i] for i, b in enumerate(BANDS)}
    out["SCL"] = arr[len(BANDS)]
    return out
