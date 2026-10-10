"""Optical members from a Sentinel-2 pre/post pair (used only where the sky is clear).

- Water: MNDWI = (G − SWIR1)/(G + SWIR1). Flood = water after but not before.
- Debris/scour: NDVI loss combined with a bare-soil index (BSI) gain.
  BSI = ((SWIR1 + R) − (NIR + B)) / ((SWIR1 + R) + (NIR + B)).
All thresholds are soft (logistic) and uncalibrated; fusion weights them.
"""

from dataclasses import dataclass, field

import numpy as np

from dhara.adapters.sentinel2 import SCL_INVALID


@dataclass
class OpticalResult:
    p_flood: np.ndarray  # NaN = cloudy / not observable
    p_debris: np.ndarray
    water_pre: np.ndarray  # bool
    observable: np.ndarray  # bool
    meta: dict = field(default_factory=dict)


def _sig(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _nd(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)


def indices(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    bsi_num = (s["B11"] + s["B04"]) - (s["B08"] + s["B02"])
    bsi_den = (s["B11"] + s["B04"]) + (s["B08"] + s["B02"])
    with np.errstate(invalid="ignore", divide="ignore"):
        bsi = bsi_num / bsi_den
    return {"mndwi": _nd(s["B03"], s["B11"]), "ndvi": _nd(s["B08"], s["B04"]), "bsi": bsi}


def clear(s: dict[str, np.ndarray], buffer_px: int = 5) -> np.ndarray:
    """Clear-sky pixels. SCL invalid classes are dilated by `buffer_px` (10 m px) because SCL
    misses thin cloud fringes and shadow edges, which otherwise look like vegetation loss."""
    from scipy.ndimage import binary_dilation

    scl = np.nan_to_num(s["SCL"], nan=0).astype(np.uint8)
    bad = np.isin(scl, SCL_INVALID) | ~np.isfinite(s["B03"]) | ~np.isfinite(s["B11"])
    if buffer_px > 0:
        bad = binary_dilation(bad, iterations=buffer_px)
    return ~bad


def run(pre: dict[str, np.ndarray] | None, post: dict[str, np.ndarray]) -> OpticalResult:
    ip = indices(post)
    ok = clear(post)
    p_water_post = _sig((ip["mndwi"] - 0.0) / 0.08)
    if pre is not None:
        ib = indices(pre)
        ok_pre = clear(pre)
        water_pre = ok_pre & (ib["mndwi"] > 0.05)
        p_flood = p_water_post * np.where(ok_pre, 1.0 - _sig((ib["mndwi"] - 0.0) / 0.08), 0.5)
        ndvi_loss = ib["ndvi"] - ip["ndvi"]
        bsi_gain = ip["bsi"] - ib["bsi"]
        p_debris = _sig((ndvi_loss - 0.15) / 0.05) * _sig((bsi_gain - 0.03) / 0.03)
        p_debris = np.where(ok_pre, p_debris, np.nan)
    else:
        water_pre = np.zeros_like(ok)
        p_flood = p_water_post * 0.5  # no baseline: cannot separate flood from normal water
        p_debris = np.full(ok.shape, np.nan, np.float32)
    p_flood = np.where(ok, p_flood, np.nan).astype(np.float32)
    p_debris = np.where(ok, p_debris, np.nan).astype(np.float32)
    return OpticalResult(
        p_flood=p_flood,
        p_debris=p_debris,
        water_pre=water_pre,
        observable=ok,
        meta={"clear_fraction_post": round(float(ok.mean()), 4), "has_pre": pre is not None},
    )
