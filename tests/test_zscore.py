"""Multi-date radar baseline: real change is flagged, ordinary variation on noisy ground is not."""

import numpy as np

from dhara.members import zscore


def _lin(db):
    return (10 ** (db / 10)).astype(np.float32)


def test_change_flagged_and_noisy_ground_ignored():
    rng = np.random.default_rng(0)
    h = w = 60
    calm = np.full((h, w), 0.3, np.float32)  # quiet ground: small date-to-date variation (dB)
    calm[:, 30:] = 3.0  # noisy slope: large natural variation
    base = -8.0
    pres = [
        _lin(np.stack([base + rng.normal(0, calm), base - 7 + rng.normal(0, calm)]))
        for _ in range(3)
    ]
    post_db = np.stack([base + rng.normal(0, calm), base - 7 + rng.normal(0, calm)])
    post_db[:, 10:20, 5:15] -= 9.0  # strong darkening on quiet ground (new water)
    post_db[:, 40:50, 40:50] += 4.0  # within the noisy slope's normal range
    r = zscore.run(pres, _lin(post_db), np.ones((h, w), bool), water_db=-14.0)
    assert np.nanmean(r.p_change[11:19, 6:14]) > 0.9
    assert np.nanmean(r.p_water[11:19, 6:14]) > 0.9
    assert np.nanmean(r.p_change[41:49, 41:49]) < 0.3
    assert np.nanmean(r.p_change[:, 20:30]) < 0.1  # untouched quiet ground


def test_needs_two_pre_passes():
    one = np.ones((2, 8, 8), np.float32)
    try:
        zscore.run([one], one, np.ones((8, 8), bool), water_db=-14.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
