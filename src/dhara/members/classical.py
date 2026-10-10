"""Classical SAR flood member: log-ratio change detection plus split-based Otsu thresholding.

Works on co-registered pre/post γ⁰ (linear power) arrays on one grid (any projected CRS).
The tile selection follows the split-based idea (Martinis et al.): pick tiles whose change
distribution is likely bimodal, then threshold the pooled pixels with Otsu.
"""

from dataclasses import dataclass, field

import numpy as np
from skimage.filters import threshold_otsu

# Fallbacks if no bimodal tiles are found; recorded in `meta` whenever they are used.
FALLBACK_DECREASE_DB = -3.0
FALLBACK_WATER_VV_DB = -18.0


@dataclass
class ClassicalResult:
    p_flood: np.ndarray  # float32 [0,1]; NaN = not observable
    log_ratio_db: np.ndarray  # float32, post − pre, mean of VV and VH, in dB
    permanent_water: np.ndarray  # bool; dark both before and after
    meta: dict = field(default_factory=dict)


def to_db(x: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.clip(x, 1e-6, None))).astype(np.float32)


def _ashman_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10 or b.size < 10:
        return 0.0
    return float(np.sqrt(2) * abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var() + 1e-9))


def split_based_threshold(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    tile: int = 128,
    side: str = "low",
    min_valid: float = 0.9,
    top_frac: float = 0.05,
    min_tiles: int = 5,
) -> tuple[float | None, dict]:
    """Otsu threshold on tiles likely to contain both classes.

    side="low": the target class is the low tail (darkening, water). Candidate tiles have
    above-median std and below-median mean. Returns (threshold or None, diagnostics).
    None means the pooled distribution was not bimodal (Ashman's D < 2).
    """
    h, w = values.shape
    stats = []
    for r in range(0, h - tile + 1, tile):
        for c in range(0, w - tile + 1, tile):
            v = valid[r : r + tile, c : c + tile]
            if v.mean() < min_valid:
                continue
            x = values[r : r + tile, c : c + tile][v]
            stats.append((r, c, float(x.mean()), float(x.std())))
    diag: dict = {"tiles_total": len(stats)}
    if len(stats) < min_tiles:
        diag["reason"] = "too few valid tiles"
        return None, diag
    arr = np.array([(s[2], s[3]) for s in stats])
    med_mean, med_std = np.median(arr[:, 0]), np.median(arr[:, 1])
    if side == "low":
        cand = [s for s in stats if s[3] > med_std and s[2] < med_mean]
    else:
        cand = [s for s in stats if s[3] > med_std and s[2] > med_mean]
    cand.sort(key=lambda s: -s[3])
    k = max(min_tiles, int(len(stats) * top_frac))
    cand = cand[:k]
    diag["tiles_used"] = len(cand)
    if len(cand) < min_tiles:
        diag["reason"] = "too few candidate tiles"
        return None, diag
    pooled = np.concatenate(
        [values[r : r + tile, c : c + tile][valid[r : r + tile, c : c + tile]] for r, c, *_ in cand]
    )
    t = float(threshold_otsu(pooled))
    d = _ashman_d(pooled[pooled < t], pooled[pooled >= t])
    diag["ashman_d"] = round(d, 3)
    if d < 2.0:
        diag["reason"] = "not bimodal"
        return None, diag
    return t, diag


def adaptive_tile(shape: tuple[int, ...], target_tiles: int = 100) -> int:
    """Tile size giving roughly `target_tiles` tiles, clamped to [32, 128] px."""
    h, w = shape[-2:]
    return int(np.clip(np.sqrt(h * w / target_tiles), 32, 128))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def run(
    pre: np.ndarray,
    post: np.ndarray,
    observable: np.ndarray,
    *,
    tile: int | None = None,
    scale_db: float = 1.0,
) -> ClassicalResult:
    """pre/post: (2, H, W) linear γ⁰ [VV, VH]. observable: (H, W) bool (False = layover/shadow/no data).

    Returns p_flood = P(decrease) · P(post dark), which is uncalibrated: logistic around the
    automatic thresholds with `scale_db` spread. Calibration happens in fusion (milestone 2).
    """
    pre_db, post_db = to_db(pre), to_db(post)
    valid = observable & np.isfinite(pre_db).all(0) & np.isfinite(post_db).all(0)
    with np.errstate(invalid="ignore"):
        lr = (post_db - pre_db).mean(axis=0)
    lr[~valid] = np.nan
    tile = tile or adaptive_tile(lr.shape)
    meta: dict = {"tile_px": tile}

    t_dec, d1 = split_based_threshold(np.nan_to_num(lr), valid, tile=tile, side="low")
    meta["decrease_threshold"] = d1
    if t_dec is None or t_dec > -1.0:
        meta["decrease_threshold"]["fallback_db"] = FALLBACK_DECREASE_DB
        t_dec = FALLBACK_DECREASE_DB
    meta["decrease_threshold_db"] = round(t_dec, 3)

    vv_post = post_db[0]
    t_w, d2 = split_based_threshold(np.nan_to_num(vv_post), valid, tile=tile, side="low")
    meta["water_threshold"] = d2
    if t_w is None or t_w > -12.0:
        meta["water_threshold"]["fallback_db"] = FALLBACK_WATER_VV_DB
        t_w = FALLBACK_WATER_VV_DB
    meta["water_threshold_db"] = round(t_w, 3)

    p_dec = _sigmoid((t_dec - lr) / scale_db)
    p_dark = _sigmoid((t_w - vv_post) / scale_db)
    p = p_dec * p_dark
    p[~valid] = np.nan
    perm = valid & (pre_db[0] < t_w) & (vv_post < t_w)
    p[perm] = 0.0  # already water before the event: not flood
    meta["observable_fraction"] = round(float(valid.mean()), 4)
    return ClassicalResult(p_flood=p, log_ratio_db=lr, permanent_water=perm, meta=meta)
