import numpy as np

from dhara.terrain import d8, hydro


def _dem(seed: int = 0, n: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    dem = 500.0 + 2.0 * y + 0.5 * x + rng.normal(0, 3, (n, n))  # tilted, noisy, with pits
    dem[5:9, 40:44] = np.nan
    return dem


def test_order_is_downstream_first():
    dem = _dem()
    cond, order = d8.priority_flood_eps(dem, 1e-3)
    fd = d8.flow_direction(cond)
    W = dem.shape[1]
    pos = np.full(dem.size, -1)
    pos[order] = np.arange(order.size)
    assert order.size == np.isfinite(dem).sum()
    for i in order:
        r, c = divmod(int(i), W)
        k = fd[r, c]
        if k >= 0:
            j = (r + d8.DR[k]) * W + (c + d8.DC[k])
            assert pos[j] < pos[i]  # the cell it drains to comes first


def test_accumulation_and_hand_match_sorted_reference():
    dem = _dem(1)
    cond, order = d8.priority_flood_eps(dem, 1e-3)
    fd = d8.flow_direction(cond)
    acc = d8.accumulation(cond, fd, order)
    # Reference: walk cells by descending elevation (the former implementation).
    H, W = dem.shape
    ref = np.where(np.isnan(cond), 0.0, 1.0)
    flat = np.where(np.isnan(cond), -np.inf, cond).ravel()
    for i in np.argsort(-flat, kind="stable"):
        r, c = divmod(int(i), W)
        if np.isnan(cond[r, c]):
            continue
        k = fd[r, c]
        if k >= 0:
            ref[r + d8.DR[k], c + d8.DC[k]] += ref[r, c]
    assert np.array_equal(acc, ref)
    channels = acc >= 30
    h = d8.hand(cond, fd, channels, order)
    assert np.all(h[channels] == 0)
    assert np.nanmin(h) >= 0


def test_slope_matches_numpy_gradient():
    dem = np.nan_to_num(_dem(2), nan=520.0)
    gy, gx = np.gradient(dem, 30.0)
    ref = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    assert np.allclose(hydro.slope_degrees(dem, 30.0), ref, atol=1e-4)
