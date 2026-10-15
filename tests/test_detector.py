import numpy as np

from dhara.fusion import fuse as F
from dhara.fusion.detector import DetectorParams, refine


def _valley(n=200):
    """A straight valley along x: HAND rises 0.5 m per pixel away from the river row (10 m grid)."""
    rows = np.arange(n)[:, None]
    hand = np.abs(rows - n // 2) * 0.5 + np.zeros((n, n))
    slope = np.full((n, n), 5.0)
    major = np.zeros((n, n), bool)
    major[n // 2 - 3 : n // 2 + 4] = True  # the pipeline widens the channel by 3 px
    return hand.astype(np.float32), slope.astype(np.float32), major


def _run(cls, conf, hand, slope, major, **kw):
    p = np.zeros(cls.shape, np.float32)
    return refine(cls, conf, p, p, hand, slope, major, DetectorParams(**kw))


def test_fill_closes_gap_next_to_observed_debris():
    hand, slope, major = _valley()
    cls = np.full(hand.shape, F.NO_CHANGE, np.uint8)
    conf = np.zeros(hand.shape, np.uint8)
    cls[95:106, 20:80] = F.DEBRIS  # observed footprint, ±5 px around the river
    conf[95:106, 20:80] = F.TIER_HIGH
    cls[95:106, 110:180] = F.DEBRIS
    conf[95:106, 110:180] = F.TIER_HIGH
    out, oconf, st = _run(cls, conf, hand, slope, major)
    assert (out[100, 80:110] == F.DEBRIS).all()  # the 300 m gap between the two is filled
    assert (oconf[100, 85:105] == F.TIER_LOW).all()  # inferred, not observed
    assert st["px_filled"] > 0


def test_fill_stays_within_distance_of_footprint():
    hand, slope, major = _valley()
    hand[:] = 0.0  # a wide flat valley floor: everything is "low"
    cls = np.full(hand.shape, F.NO_CHANGE, np.uint8)
    conf = np.zeros(hand.shape, np.uint8)
    cls[94:107, 80:120] = F.DEBRIS  # enough footprint in each 500 m reach to read its level
    conf[94:107, 80:120] = F.TIER_HIGH
    out, _, _ = _run(cls, conf, hand, slope, major)
    filled = (out == F.DEBRIS) & (cls != F.DEBRIS)
    rr, cc = np.nonzero(filled)
    assert filled.any()
    # Distance (m) from each filled pixel to the observed rectangle stays within the 250 m cap.
    dr = np.clip(np.maximum(94 - rr, rr - 106), 0, None)
    dc = np.clip(np.maximum(80 - cc, cc - 119), 0, None)
    assert np.hypot(dr, dc).max() * 10 <= 250


def test_no_trusted_seed_no_fill_and_hillside_patch_removed():
    hand, slope, major = _valley()
    cls = np.full(hand.shape, F.NO_CHANGE, np.uint8)
    conf = np.zeros(hand.shape, np.uint8)
    cls[96:104, 50:60] = F.DEBRIS  # low-confidence only: no seed
    conf[96:104, 50:60] = F.TIER_LOW
    cls[10:20, 10:20] = F.DEBRIS  # far up the hillside (HAND 40–45 m) and not on the river
    conf[10:20, 10:20] = F.TIER_HIGH
    out, _, st = _run(cls, conf, hand, slope, major)
    assert st["px_filled"] == 0
    assert (out[10:20, 10:20] == F.NO_CHANGE).all()


def test_swept_channel_marks_river_inside_covered_reach():
    hand, slope, major = _valley()
    cls = np.full(hand.shape, F.NO_CHANGE, np.uint8)
    conf = np.zeros(hand.shape, np.uint8)
    cls[99:102, :] = F.PERMANENT_WATER
    cls[92:99, 40:160] = F.DEBRIS
    conf[92:99, 40:160] = F.TIER_HIGH
    out, oconf, st = _run(cls, conf, hand, slope, major)
    assert (out[100, 60:140] == F.DEBRIS).all()
    assert (oconf[100, 60:140] == F.TIER_LOW).all()
    assert st["px_swept_channel"] > 0
