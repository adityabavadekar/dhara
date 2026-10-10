"""Stage 4: fuse member evidence with a terrain prior into classes, probabilities and confidence.

Method (all on the 10 m analysis grid):
1. Each member gives per-pixel probabilities for `flood` and/or `debris` (NaN = cannot see).
2. Logarithmic opinion pool: the weighted *mean* of member logits over the members that observe
   the pixel. A mean, not a sum, so more members don't inflate confidence.
3. A physics prior in logit space from HAND and slope. It is soft: strong evidence can override
   it. Water cannot pond far above the drainage network or on steep slopes; debris surges reach
   higher, so the debris prior is looser.
4. Classes (flood takes precedence over debris, since the debris member responds to any strong
   change, including new water): not_observable (no member sees the pixel; never reported as "no damage"),
   permanent_water, flood, debris, no_change.
5. Confidence for positive pixels: High = P ≥ 0.8 and ≥ 2 members agree; Moderate = P ≥ 0.65 or
   ≥ 2 agree; otherwise Low. Patches not connected to the major drainage network are capped at
   Low (the hydraulic-connectivity prior).
"""

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

NOT_OBSERVABLE, NO_CHANGE, PERMANENT_WATER, FLOOD, DEBRIS, OUTSIDE = 0, 1, 2, 3, 4, 255
CLASS_NAMES = {
    NOT_OBSERVABLE: "not_observable",
    NO_CHANGE: "no_change",
    PERMANENT_WATER: "permanent_water",
    FLOOD: "flood",
    DEBRIS: "debris",
}
TIER_NONE, TIER_LOW, TIER_MOD, TIER_HIGH = 0, 1, 2, 3
TIER_NAMES = {TIER_NONE: "none", TIER_LOW: "low", TIER_MOD: "moderate", TIER_HIGH: "high"}


class FusionParams(BaseModel):
    # Soft HAND limits (m) where the prior crosses zero; logistic scale (m).
    flood_hand_m: float = 20.0
    debris_hand_m: float = 40.0
    hand_scale_m: float = 6.0
    # Slope (deg) above which the flood prior turns negative.
    flood_slope_deg: float = 15.0
    debris_slope_deg: float = 30.0
    slope_scale_deg: float = 4.0
    prior_weight: float = 1.0
    prior_clip: tuple[float, float] = (-4.0, 1.0)
    p_class: float = 0.5


@dataclass
class Evidence:
    name: str
    target: str  # "flood" | "debris"
    p: np.ndarray  # [0,1], NaN = not observed
    weight: float = 1.0
    sensor: str = "s1"


@dataclass
class FusionResult:
    p_flood: np.ndarray
    p_debris: np.ndarray
    classes: np.ndarray  # uint8
    confidence: np.ndarray  # uint8 tier
    n_observing: np.ndarray  # uint8
    n_agree: np.ndarray  # uint8
    meta: dict = field(default_factory=dict)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def terrain_prior(
    hand: np.ndarray, slope: np.ndarray, target: str, prm: FusionParams
) -> np.ndarray:
    """Prior logit from HAND (m) and slope (deg). NaN terrain → 0 (uninformative)."""
    h0 = prm.flood_hand_m if target == "flood" else prm.debris_hand_m
    s0 = prm.flood_slope_deg if target == "flood" else prm.debris_slope_deg
    lp = (h0 - np.nan_to_num(hand, nan=h0)) / prm.hand_scale_m
    lp = lp + np.minimum(0.0, (s0 - np.nan_to_num(slope, nan=0.0)) / prm.slope_scale_deg)
    return np.clip(lp, *prm.prior_clip).astype(np.float32)


def pool(evidence: list[Evidence], target: str, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted mean logit over observing members; returns (mean logit, n_obs, n_agree)."""
    num = np.zeros(shape, np.float64)
    den = np.zeros(shape, np.float64)
    n_obs = np.zeros(shape, np.uint8)
    n_agree = np.zeros(shape, np.uint8)
    for e in evidence:
        if e.target != target:
            continue
        ok = np.isfinite(e.p)
        num[ok] += e.weight * logit(e.p[ok])
        den[ok] += e.weight
        n_obs += ok
        n_agree += ok & (np.nan_to_num(e.p) >= 0.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, np.nan)
    return mean, n_obs, n_agree


def connected_to(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Pixels of `mask` whose 8-connected component touches `seeds`."""
    from scipy.ndimage import label

    lab, n = label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(mask)
    hit = np.unique(lab[seeds & mask])
    return np.isin(lab, hit[hit > 0])


def fuse(
    evidence: list[Evidence],
    hand: np.ndarray,
    slope: np.ndarray,
    in_aoi: np.ndarray,
    permanent_water: np.ndarray,
    prm: FusionParams | None = None,
    major_channels: np.ndarray | None = None,
) -> FusionResult:
    prm = prm or FusionParams()
    shape = in_aoi.shape
    out: dict[str, np.ndarray] = {}
    nobs_all = np.zeros(shape, np.uint8)
    nagree: dict[str, np.ndarray] = {}
    for target in ("flood", "debris"):
        m, n_obs, n_ag = pool(evidence, target, shape)
        prior = terrain_prior(hand, slope, target, prm)
        p = sigmoid(m + prm.prior_weight * prior)
        p[~np.isfinite(m)] = np.nan
        out[target] = p
        nobs_all = np.maximum(nobs_all, n_obs)
        nagree[target] = n_ag

    pf, pd = out["flood"], out["debris"]
    observable = in_aoi & (nobs_all > 0)
    cls = np.full(shape, NO_CHANGE, np.uint8)
    pf0, pd0 = np.nan_to_num(pf), np.nan_to_num(pd)
    # Water is the more specific explanation of a change: debris = strong change that is not water.
    is_flood = pf0 >= prm.p_class
    is_debris = (pd0 >= prm.p_class) & ~is_flood
    cls[is_flood] = FLOOD
    cls[is_debris] = DEBRIS
    # Pre-existing water must sit on/near the drainage network; radar-dark slopes are not water.
    near_drainage = np.nan_to_num(hand, nan=np.inf) <= prm.flood_hand_m
    cls[permanent_water & observable & near_drainage] = PERMANENT_WATER
    cls[~observable] = NOT_OBSERVABLE
    cls[~in_aoi] = OUTSIDE

    p_pos = np.where(cls == FLOOD, pf0, np.where(cls == DEBRIS, pd0, 0.0))
    ag = np.where(cls == FLOOD, nagree["flood"], np.where(cls == DEBRIS, nagree["debris"], 0))
    tier = np.full(shape, TIER_NONE, np.uint8)
    pos = (cls == FLOOD) | (cls == DEBRIS)
    tier[pos] = TIER_LOW
    tier[pos & ((p_pos >= 0.65) | (ag >= 2))] = TIER_MOD
    tier[pos & (p_pos >= 0.8) & (ag >= 2)] = TIER_HIGH
    n_capped = 0
    if major_channels is not None:
        # Hydraulic connectivity: a flood/debris patch must connect to the main drainage
        # network. Isolated patches (e.g. side-stream wetting under cloud edges) stay mapped
        # but are capped at low confidence ("unexplained change") instead of being deleted.
        linked = connected_to(pos, major_channels)
        capped = pos & ~linked & (tier > TIER_LOW)
        n_capped = int(capped.sum())
        tier[pos & ~linked] = TIER_LOW
    meta = {
        "params": prm.model_dump(),
        "px_capped_unconnected": n_capped,
        "members": [
            {
                "name": e.name,
                "target": e.target,
                "weight": e.weight,
                "sensor": e.sensor,
                "observed_fraction": round(float(np.isfinite(e.p)[in_aoi].mean()), 4),
            }
            for e in evidence
        ],
    }
    return FusionResult(
        pf, pd, cls, tier, nobs_all, np.maximum(nagree["flood"], nagree["debris"]), meta
    )
