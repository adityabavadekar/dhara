"""Debris / scour member: bidirectional SAR backscatter change.

Debris-flow deposits and scoured channels often get *brighter* (rough boulder fields, new
double-bounce) or *darker* (fine wet sediment, ponding). This is unlike open water, which only gets
darker. The member flags strong change in either direction with split-based thresholding on |Δσ|,
measured in dB and averaged over VV and VH. It is uncalibrated; fusion applies the terrain prior.

Grid-agnostic: arrays on any projected grid.
"""

from dataclasses import dataclass, field

import numpy as np

from dhara.members.classical import split_based_threshold, to_db

FALLBACK_ABS_CHANGE_DB = 3.0


@dataclass
class DebrisResult:
    p_change: np.ndarray  # float32 [0,1], NaN = not observable
    abs_change_db: np.ndarray
    brighter: np.ndarray  # bool, change is an increase
    meta: dict = field(default_factory=dict)


def run(
    pre: np.ndarray,
    post: np.ndarray,
    observable: np.ndarray,
    *,
    tile: int,
    scale_db: float = 1.0,
    pre2: np.ndarray | None = None,
) -> DebrisResult:
    """pre/post: (2, H, W) linear γ⁰ [VV, VH]; observable: (H, W) bool.

    pre2: optional earlier pre-event pass on the same track used to calibrate
    the empirical null distribution under the placebo hypothesis (p99 false-alarm cap).
    """
    pre_db, post_db = to_db(pre), to_db(post)
    with np.errstate(invalid="ignore"):
        d = (post_db - pre_db).mean(axis=0)
    valid = observable & np.isfinite(d)
    a = np.abs(np.where(valid, d, 0.0))
    t, diag = split_based_threshold(a, valid, tile=tile, side="high")
    meta: dict = {"abs_change_threshold": diag}
    if t is None or t < 1.5:
        diag["fallback_db"] = FALLBACK_ABS_CHANGE_DB
        t = FALLBACK_ABS_CHANGE_DB

    # Placebo null calibration: if pre2 is available, calibrate against natural baseline noise
    if pre2 is not None:
        pre2_db = to_db(pre2)
        with np.errstate(invalid="ignore"):
            d_null = (pre_db - pre2_db).mean(axis=0)
        valid_null = observable & np.isfinite(d_null)
        null_shifts = np.abs(d_null[valid_null])
        if len(null_shifts) >= 200:
            p99 = float(np.percentile(null_shifts, 99.0))
            meta["placebo_null_p99_db"] = round(p99, 3)
            # Ensure detection threshold strictly exceeds 99% of null baseline noise
            t = max(float(t), p99)

    meta["abs_change_threshold_db"] = round(float(t), 3)
    p = (1.0 / (1.0 + np.exp(-(a - t) / scale_db))).astype(np.float32)
    p[~valid] = np.nan
    a = a.astype(np.float32)
    a[~valid] = np.nan
    return DebrisResult(p_change=p, abs_change_db=a, brighter=valid & (d > 0), meta=meta)
