import numpy as np

from dhara.sar import grd


def test_zero_doppler_on_straight_orbit():
    t = np.linspace(-50, 50, 11)
    v = np.array([7000.0, 0.0, 0.0])
    pos = np.array([0.0, 0.0, 700_000.0]) + t[:, None] * v
    orbit = grd.Orbit(t, pos, np.tile(v, (len(t), 1)))
    target = np.array([[3500.0, 250_000.0, 0.0], [-7000.0, 300_000.0, 0.0]])
    tz, p, _ = grd.zero_doppler(orbit, target, np.zeros(2))
    np.testing.assert_allclose(tz, [0.5, -1.0], atol=1e-6)  # sensor abeam of each target
    np.testing.assert_allclose(p[:, 0], target[:, 0], atol=1e-3)


def test_lut_window_is_bilinear():
    lut = grd.Lut(
        lines=np.array([0.0, 100.0]),
        pixels=np.array([[0.0, 10.0], [0.0, 10.0]]),
        values=np.array([[1.0, 3.0], [5.0, 7.0]]),
    )
    w = lut.window(np.array([0.0, 50.0, 100.0]), np.array([0.0, 5.0, 10.0]))
    np.testing.assert_allclose(w, [[1, 2, 3], [3, 4, 5], [5, 6, 7]])


def test_lee_keeps_constant_image_and_nans():
    img = np.full((20, 20), 0.05, np.float32)
    img[3, 3] = np.nan
    out = grd.lee(img)
    assert np.isnan(out[3, 3])
    np.testing.assert_allclose(out[np.isfinite(out)], 0.05, rtol=1e-5)


def test_slant_to_pixel_uses_srgr_polynomial():
    ann = grd.Annotation(
        epoch=np.datetime64("2026-08-28T00:00:00", "ns"),
        line_dt=0.001,
        range_spacing=10.0,
        azimuth_spacing=10.0,
        n_lines=100,
        n_samples=100,
        orbit_t=np.zeros(1),
        orbit_pos=np.zeros((1, 3)),
        orbit_vel=np.zeros((1, 3)),
        srgr_t=np.array([0.0, 10.0]),
        srgr_sr0=np.array([800_000.0, 800_000.0]),
        srgr_coef=np.array([[0.0, 2.0], [0.0, 2.0]]),  # ground = 2 × (slant − sr0)
        geo={},
    )
    px = grd.slant_to_pixel(ann, np.array([5.0]), np.array([800_050.0]))
    np.testing.assert_allclose(px, [10.0])  # 100 m ground / 10 m pixels
