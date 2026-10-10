import numpy as np

from dhara.members import classical


def _scene(seed: int = 0, flood: bool = True):
    """Synthetic γ⁰ pair (linear). Land ≈ -8 dB VV; a flooded block drops to ≈ -22 dB."""
    rng = np.random.default_rng(seed)
    h = w = 512
    land_vv = 10 ** (rng.normal(-8, 1.0, (h, w)) / 10)
    land_vh = 10 ** (rng.normal(-15, 1.0, (h, w)) / 10)
    pre = np.stack([land_vv, land_vh]).astype(np.float32)
    post = pre * 10 ** (rng.normal(0, 0.5, (2, h, w)) / 10)
    truth = np.zeros((h, w), bool)
    if flood:
        # Many 64-px patches so several 128-px tiles are bimodal.
        for r in range(0, h, 128):
            for c in range(0, w, 128):
                truth[r + 32 : r + 96, c + 32 : c + 96] = (r // 128 + c // 128) % 2 == 0
        post[0][truth] = 10 ** (rng.normal(-22, 1.0, truth.sum()) / 10)
        post[1][truth] = 10 ** (rng.normal(-28, 1.0, truth.sum()) / 10)
    return pre, post.astype(np.float32), truth


def test_detects_synthetic_flood():
    pre, post, truth = _scene()
    obs = np.ones(truth.shape, bool)
    res = classical.run(pre, post, obs)
    pred = res.p_flood >= 0.5
    iou = (pred & truth).sum() / (pred | truth).sum()
    assert iou > 0.9, res.meta
    assert "fallback_db" not in res.meta["decrease_threshold"]


def test_not_observable_stays_nan():
    pre, post, truth = _scene()
    obs = np.ones(truth.shape, bool)
    obs[:, :100] = False
    res = classical.run(pre, post, obs)
    assert np.isnan(res.p_flood[:, :100]).all()
    assert np.isfinite(res.p_flood[:, 100:]).all()


def test_no_flood_uses_fallback_and_finds_little():
    pre, post, _ = _scene(flood=False)
    res = classical.run(pre, post, np.ones(pre.shape[1:], bool))
    assert (res.p_flood >= 0.5).mean() < 0.01
