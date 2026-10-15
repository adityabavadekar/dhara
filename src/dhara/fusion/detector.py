"""Stage 4b: turn fused per-pixel classes into a physically consistent flood / debris footprint.

All arrays on the 10 m analysis grid (local UTM). Height above the nearest drainage (HAND, m) and
slope (deg) come from Copernicus DEM GLO-30.

Fusion (fuse.py) decides each pixel on its own. A flood or debris flow is one connected body that
follows the valley, so this step applies four physical rules to the fused classes:

1. Terrain gate: flood water cannot stand far above the drainage or on steep slopes; debris surges
   reach higher but not up whole hillsides. Positives above those limits are removed.
2. Specks: isolated patches smaller than `min_patch_px` are radar noise at 10 m, not a flood.
3. Corridor: a debris patch must connect to the main drainage network (upstream area ≥ the
   pipeline's major-channel threshold). Hillside change unconnected to the river is ordinary
   change (harvest, soil moisture, slope wash), the main false-alarm source in mountain valleys.
   Unconnected water is kept (ponding on a plain), at the low confidence fusion gives it.
4. Valley-floor fill: in each reach (`block_px` square), the flood level is read from where the
   mapped footprint reaches, as a high percentile of its HAND. Ground below that level that is
   connected to the mapped footprint and not steep is filled, including ground the radar could
   not see (shadow) or where the change was too weak to pass on its own. Filled pixels keep low
   confidence: they are inferred from the level, not observed.

5. Swept channel: the river channel inside a reach the footprint covers was run over by the same
   flow (a debris flow fills the riverbed). Pre-event water connected to the footprint there is
   marked as debris, at low confidence: inferred, since radar cannot see debris on water.
"""

import numpy as np
from pydantic import BaseModel
from scipy import ndimage as ndi

from dhara.fusion.fuse import (
    DEBRIS,
    FLOOD,
    NO_CHANGE,
    NOT_OBSERVABLE,
    OUTSIDE,
    PERMANENT_WATER,
    TIER_LOW,
)


class DetectorParams(BaseModel):
    flood_max_hand_m: float = 25.0
    debris_max_hand_m: float = 55.0
    flood_max_slope_deg: float = 25.0
    debris_max_slope_deg: float = 50.0
    min_patch_px: int = 12
    require_corridor: bool = True
    corridor_keeps_water: bool = True  # False = the version frozen for the held-out test
    # Landslide scars: debris the terrain gate removed (steep, high above the river) is restored
    # when its patch runs continuously into the final footprint: source and deposit of one flow.
    runout_scars: bool = True
    # Local high-water mark: near observed footprint, fill only up to the highest observed
    # footprint HAND within this window (px); the reach level applies only where none is near.
    local_mark_px: int = 0
    fill: bool = True
    block_px: int = 50  # 500 m reaches
    level_percentile: float = 95.0
    min_block_px: int = 40  # mapped pixels needed in a reach to read its level
    fill_max_slope_deg: float = 30.0
    fill_max_hand_m: float = 50.0
    reach_radius_blocks: float = 3.0  # how far a reach's level carries to reaches with no footprint
    swept_channel: bool = True
    # Only trusted footprint sets a reach's level and seeds the fill: pixels at or above this
    # confidence tier, in connected seeds of at least `min_seed_px`.
    seed_min_tier: int = 2
    min_seed_px: int = 50
    # Filled ground must lie within this distance of observed footprint (m): the fill closes gaps
    # in the observed flow, it does not paint a wide valley floor the flow never reached.
    fill_max_dist_m: float = 250.0
    res_m: float = 10.0


def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def _touching(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    hit = np.unique(lab[seeds & mask])
    return np.isin(lab, hit[hit > 0])


def _level_field(pos: np.ndarray, hand: np.ndarray, prm: DetectorParams) -> np.ndarray:
    """Flood level (HAND, m) per pixel: per reach, a high percentile of the footprint's HAND;
    reaches without enough footprint take the value of the nearest reach that has one, within one
    reach, else NaN (no fill)."""
    b = prm.block_px
    H, W = pos.shape
    nb_r, nb_c = -(-H // b), -(-W // b)
    lev = np.full((nb_r, nb_c), np.nan, np.float32)
    for i in range(nb_r):
        for j in range(nb_c):
            sl = (slice(i * b, (i + 1) * b), slice(j * b, (j + 1) * b))
            h = hand[sl][pos[sl]]
            h = h[np.isfinite(h)]
            if h.size >= prm.min_block_px:
                lev[i, j] = np.percentile(h, prm.level_percentile)
    known = np.isfinite(lev)
    if not known.any():
        return np.full(pos.shape, np.nan, np.float32)
    # Nearest known reach, at most one reach away.
    dist, (ii, jj) = ndi.distance_transform_edt(~known, return_indices=True)
    filled = np.where(dist <= prm.reach_radius_blocks, lev[ii, jj], np.nan).astype(np.float32)
    return np.kron(filled, np.ones((b, b), np.float32))[:H, :W]


def refine(
    classes: np.ndarray,
    confidence: np.ndarray,
    p_flood: np.ndarray,
    p_debris: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
    major_channels: np.ndarray,
    prm: DetectorParams | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the rules above to fused `classes` / `confidence` (10 m grid, local UTM). Returns new
    classes, confidence and a summary of what each rule changed (pixels)."""
    prm = prm or DetectorParams()
    cls, conf = classes.copy(), confidence.copy()
    h = np.nan_to_num(hand, nan=np.inf)
    s = np.nan_to_num(slope, nan=0.0)
    stats: dict[str, int] = {}

    gated = ((cls == FLOOD) & ((h > prm.flood_max_hand_m) | (s > prm.flood_max_slope_deg))) | (
        (cls == DEBRIS) & ((h > prm.debris_max_hand_m) | (s > prm.debris_max_slope_deg))
    )
    stats["px_terrain_gate"] = int(gated.sum())
    gated_debris = gated & (cls == DEBRIS)
    cls[gated] = NO_CHANGE

    pos = (cls == FLOOD) | (cls == DEBRIS)
    kept = _drop_small(pos, prm.min_patch_px)
    stats["px_specks"] = int((pos & ~kept).sum())
    if prm.require_corridor:
        # Only debris must reach the river: unconnected water (ponding on a plain) stays, capped
        # at low confidence by fusion's connectivity rule.
        linked = _touching(kept, major_channels)
        if prm.corridor_keeps_water:
            linked |= kept & (cls == FLOOD)
        stats["px_off_corridor"] = int((kept & ~linked).sum())
        kept = linked
    cls[pos & ~kept] = NO_CHANGE
    conf[pos & ~kept] = 0
    pos = kept

    stats["px_filled"] = 0
    seed = _drop_small(pos & (conf >= prm.seed_min_tier), prm.min_seed_px)
    if prm.fill and seed.any():
        level = _level_field(seed, np.where(np.isfinite(hand), hand, np.nan), prm)
        if prm.local_mark_px > 0:
            hs = np.where(seed & np.isfinite(hand), hand, -np.inf).astype(np.float32)
            mark = ndi.maximum_filter(hs, size=prm.local_mark_px)
            level = np.where(np.isfinite(mark), np.minimum(level, mark), level)
        near = ndi.distance_transform_edt(~pos) * prm.res_m <= prm.fill_max_dist_m
        low = (
            near
            & np.isfinite(level)
            & (h <= level)
            & (h <= prm.fill_max_hand_m)
            & (s <= prm.fill_max_slope_deg)
            & ((cls == NO_CHANGE) | (cls == NOT_OBSERVABLE) | pos)
        )
        add = _touching(low | pos, seed) & ~pos & (cls != OUTSIDE)
        # The filled pixel takes the class that dominates the footprint around it.
        near_flood = ndi.uniform_filter((cls == FLOOD).astype(np.float32), 15)
        near_debris = ndi.uniform_filter((cls == DEBRIS).astype(np.float32), 15)
        cls[add & (near_flood > near_debris)] = FLOOD
        cls[add & (near_flood <= near_debris)] = DEBRIS
        conf[add] = TIER_LOW
        stats["px_filled"] = int(add.sum())
        if prm.swept_channel:
            pos = (cls == FLOOD) | (cls == DEBRIS)
            water = (cls == PERMANENT_WATER) & np.isfinite(level)
            swept = _touching(water | pos, seed) & water
            cls[swept] = DEBRIS
            conf[swept] = TIER_LOW
            stats["px_swept_channel"] = int(swept.sum())
    if prm.runout_scars and gated_debris.any():
        final = (cls == FLOOD) | (cls == DEBRIS)
        scars = _touching(gated_debris | final, final) & gated_debris
        cls[scars] = DEBRIS
        conf[scars] = TIER_LOW
        stats["px_runout_scars"] = int(scars.sum())
    return cls, conf, stats
