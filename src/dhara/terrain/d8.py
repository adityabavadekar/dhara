"""D8 hydrology kernels (numba). Arrays are row-major (row 0 = north); CRS-agnostic.

- `priority_flood_eps`: depression filling with a tiny gradient on flats (Barnes et al. 2014), so
  every non-edge cell has a strictly lower neighbour. Also returns the order cells left the queue:
  lowest first, every cell after the neighbour it drains to, so the passes below need no sort.
- `flow_direction`: steepest descent on the conditioned DEM, encoded as index 0..7 into DR/DC
  (-1 = outlet or no data).
- `accumulation`: upstream cell count (including the cell itself), in reverse of that order.
- `hand`: height above the nearest drainage cell reached by following the flow path, in that order.
"""

import numba
import numpy as np

# Neighbour offsets: N, NE, E, SE, S, SW, W, NW
DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)
DIST = np.array([1.0, 1.4142135623730951] * 4, dtype=np.float64)


@numba.njit(cache=True)
def _heap_push(hz, hi, n, z, i):
    hz[n] = z
    hi[n] = i
    k = n
    while k > 0:
        p = (k - 1) // 2
        if hz[p] <= hz[k]:
            break
        hz[p], hz[k] = hz[k], hz[p]
        hi[p], hi[k] = hi[k], hi[p]
        k = p
    return n + 1


@numba.njit(cache=True)
def _heap_pop(hz, hi, n):
    z, i = hz[0], hi[0]
    n -= 1
    hz[0], hi[0] = hz[n], hi[n]
    k = 0
    while True:
        a, b, m = 2 * k + 1, 2 * k + 2, k
        if a < n and hz[a] < hz[m]:
            m = a
        if b < n and hz[b] < hz[m]:
            m = b
        if m == k:
            break
        hz[m], hz[k] = hz[k], hz[m]
        hi[m], hi[k] = hi[k], hi[m]
        k = m
    return z, i, n


@numba.njit(cache=True)
def priority_flood_eps(dem, eps):
    """Fill depressions; flats get a +eps slope away from the outlet. NaN cells act as outlets' edge.

    Returns the filled DEM and the flat indices of the non-NaN cells in the order they were
    finalised (ascending filled elevation)."""
    H, W = dem.shape
    out = dem.copy()
    closed = np.zeros((H, W), dtype=np.bool_)
    hz = np.empty(H * W, dtype=np.float64)
    hi = np.empty(H * W, dtype=np.int64)
    order = np.empty(H * W, dtype=np.int64)
    m = 0
    n = 0
    for r in range(H):
        for c in range(W):
            if np.isnan(dem[r, c]):
                closed[r, c] = True
                continue
            edge = r == 0 or c == 0 or r == H - 1 or c == W - 1
            if not edge:
                for k in range(8):
                    if np.isnan(dem[r + DR[k], c + DC[k]]):
                        edge = True
                        break
            if edge:
                closed[r, c] = True
                n = _heap_push(hz, hi, n, out[r, c], r * W + c)
    while n > 0:
        z, i, n = _heap_pop(hz, hi, n)
        order[m] = i
        m += 1
        r, c = i // W, i % W
        for k in range(8):
            rr, cc = r + DR[k], c + DC[k]
            if rr < 0 or cc < 0 or rr >= H or cc >= W or closed[rr, cc]:
                continue
            closed[rr, cc] = True
            nz = out[rr, cc]
            if nz <= z + eps:
                nz = z + eps
            out[rr, cc] = nz
            n = _heap_push(hz, hi, n, nz, rr * W + cc)
    return out, order[:m]


@numba.njit(cache=True)
def flow_direction(z):
    H, W = z.shape
    fd = np.full((H, W), -1, dtype=np.int8)
    for r in range(H):
        for c in range(W):
            if np.isnan(z[r, c]):
                continue
            best, bk = 0.0, -1
            for k in range(8):
                rr, cc = r + DR[k], c + DC[k]
                if rr < 0 or cc < 0 or rr >= H or cc >= W or np.isnan(z[rr, cc]):
                    continue
                s = (z[r, c] - z[rr, cc]) / DIST[k]
                if s > best:
                    best, bk = s, k
            fd[r, c] = bk
    return fd


@numba.njit(cache=True)
def accumulation(z, fd, order):
    """`order`: from `priority_flood_eps` (ascending); walked from the top down."""
    H, W = z.shape
    acc = np.ones((H, W), dtype=np.float64)
    for r in range(H):
        for c in range(W):
            if np.isnan(z[r, c]):
                acc[r, c] = 0.0
    for j in range(order.size - 1, -1, -1):
        i = order[j]
        r, c = i // W, i % W
        k = fd[r, c]
        if k >= 0:
            acc[r + DR[k], c + DC[k]] += acc[r, c]
    return acc


@numba.njit(cache=True)
def hand(z, fd, channels, order):
    """HAND on the conditioned DEM. Cells draining off-grid without meeting a channel get NaN.
    `order`: from `priority_flood_eps` (ascending: downstream cells first)."""
    H, W = z.shape
    drain = np.full((H, W), np.nan)
    for j in range(order.size):
        i = order[j]
        r, c = i // W, i % W
        if channels[r, c]:
            drain[r, c] = z[r, c]
            continue
        k = fd[r, c]
        if k >= 0:
            drain[r, c] = drain[r + DR[k], c + DC[k]]
    return (z - drain).astype(np.float32)


@numba.njit(cache=True)
def slope_deg(z, res):
    """Slope in degrees, same as numpy.gradient (central differences inside, one-sided at the
    edges) followed by arctan of the gradient length, in a single pass."""
    H, W = z.shape
    out = np.empty((H, W), dtype=np.float32)
    k = 180.0 / np.pi
    for r in range(H):
        r0 = r - 1 if r > 0 else r
        r1 = r + 1 if r < H - 1 else r
        dy = (r1 - r0) * res
        for c in range(W):
            c0 = c - 1 if c > 0 else c
            c1 = c + 1 if c < W - 1 else c
            gy = (z[r1, c] - z[r0, c]) / dy
            gx = (z[r, c1] - z[r, c0]) / ((c1 - c0) * res)
            out[r, c] = np.arctan(np.sqrt(gx * gx + gy * gy)) * k
    return out
