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
    pre: np.ndarray, post: np.ndarray, observable: np.ndarray, *, tile: int, scale_db: float = 1.0
) -> DebrisResult:
    """pre/post: (2, H, W) linear γ⁰ [VV, VH]; observable: (H, W) bool."""
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
    meta["abs_change_threshold_db"] = round(float(t), 3)
    p = (1.0 / (1.0 + np.exp(-(a - t) / scale_db))).astype(np.float32)
    p[~valid] = np.nan
    a = a.astype(np.float32)
    a[~valid] = np.nan
    return DebrisResult(p_change=p, abs_change_db=a, brighter=valid & (d > 0), meta=meta)
