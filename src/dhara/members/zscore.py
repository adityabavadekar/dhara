"""Multi-date radar baseline member: per-pixel z-score of the post-event backscatter against the
pre-event passes of the same track.

Idea after the Global Flood Mapper (Tripathy & Malladi, 2022), reimplemented here: a pixel counts
as changed when the post-event value is far outside that pixel's own pre-event range. Slopes and
fields whose backscatter normally varies a lot get a wide range, so ordinary variation is not
flagged; that is where single-pair change detection raises most of its false alarms.

    z = (post_dB − mean(pre_dB)) / sd,   sd = sqrt(local mean of the pre-event variance + floor²)

With only two or three pre-event dates the per-pixel variance is noisy, so it is averaged over a
`pool_px` window and floored at `floor_db`. Values are 3x3 box-filtered in linear power first.

- debris/scour: |z| ≥ k in VV and VH with the same sign (brighter or darker);
- water: z ≤ −k in VV and VH and the post-event VV is water-dark.
Thresholds are soft (logistic, `scale` in z units) and uncalibrated; fusion applies the terrain prior.

Grid-agnostic: arrays on any projected grid (the job's local UTM grid in the pipeline).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter

from dhara.members.classical import to_db


@dataclass
class ZScoreResult:
    p_change: np.ndarray  # float32 [0,1]; NaN = not observable
    p_water: np.ndarray  # float32 [0,1]; NaN = not observable
    z: np.ndarray  # float32 (2, H, W): z of VV and VH
    meta: dict = field(default_factory=dict)


def _sig(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _box_db(x: np.ndarray, box: int) -> np.ndarray:
    return to_db(np.stack([uniform_filter(np.nan_to_num(b), box) for b in x]))


def run(
    pres: list[np.ndarray],
    post: np.ndarray,
    observable: np.ndarray,
    *,
    water_db: float,
    k: float = 3.0,
    floor_db: float = 0.75,
    box: int = 3,
    pool_px: int = 5,
    scale: float = 0.5,
) -> ZScoreResult:
    """pres: ≥ 2 pre-event passes, each (2, H, W) linear γ⁰ [VV, VH]; post: (2, H, W).
    observable: (H, W) bool. water_db: post-event VV level (dB) below which a pixel is water-dark
    (the classical member's automatic threshold)."""
    if len(pres) < 2:
        raise ValueError("the z-score baseline needs at least two pre-event passes")
    base = np.stack([_box_db(p, box) for p in pres])  # (n, 2, H, W)
    mu = base.mean(0)
    var = base.var(0, ddof=1)
    var = np.stack([uniform_filter(v, pool_px) for v in var])
    sd = np.sqrt(var + floor_db**2)
    post_db = _box_db(post, box)
    z = ((post_db - mu) / sd).astype(np.float32)
    valid = observable & np.isfinite(z).all(0) & np.isfinite(base).all((0, 1))

    same_sign = np.sign(z[0]) == np.sign(z[1])
    weakest = np.minimum(np.abs(z[0]), np.abs(z[1]))  # both polarisations must agree
    p_change = np.where(same_sign, _sig((weakest - k) / scale), 0.0).astype(np.float32)
    darker = -np.maximum(z[0], z[1])  # both must get darker
    p_dark = _sig((water_db - post_db[0]) / 1.0)
    p_water = (_sig((darker - k) / scale) * p_dark).astype(np.float32)
    p_change[~valid] = np.nan
    p_water[~valid] = np.nan
    z[:, ~valid] = np.nan
    meta = {
        "baseline_passes": len(pres),
        "k": k,
        "floor_db": floor_db,
        "box_px": box,
        "pool_px": pool_px,
        "changed_fraction": round(float((np.nan_to_num(p_change) >= 0.5)[valid].mean()), 4)
        if valid.any()
        else 0.0,
    }
    return ZScoreResult(p_change=p_change, p_water=p_water, z=z, meta=meta)
