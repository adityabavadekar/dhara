"""Our own Sentinel-1 IW GRD processing chain: original product → calibrated, terrain-corrected
backscatter on the job grid. Replaces the server-side processing of the Sentinel Hub route
(`adapters.sentinel1.fetch_pass`) with the same outputs, so the rest of the pipeline is unchanged.

Input: the original GRD product on the CDSE S3 store (cloud-optimised SAFE). Only the job's window
of the measurement images is read, plus the annotation XMLs. Output: arrays on `grid` (local UTM,
10 m), bands [VV, VH] (+ localIncidenceAngle in degrees, shadowMask non-zero = layover/shadow).

Steps, per polarisation and slice:
1. Thermal noise removal: noise = range LUT × azimuth LUT (IPF ≥ 2.9 annotation), DN² − noise.
2. Radiometric calibration: β⁰ = (DN² − noise) / B², σ⁰ = (DN² − noise) / S² with the betaNought /
   sigmaNought LUTs, bilinearly interpolated in (line, pixel).
3. Speckle filter: Lee 5×5 in radar geometry (equivalent number of looks of IW GRDH ≈ 4.4).
4. Terrain correction (Range-Doppler, backward geocoding): for every output pixel, the target
   point from its UTM position and Copernicus DEM height (+ the geoid offset, below) is converted
   to Earth-fixed coordinates; the zero-Doppler time is solved on the orbit (state vectors from the
   annotation, polynomial fit; Newton iteration); slant range → ground range with the annotation's
   SRGR polynomials gives the GRD pixel, the time gives the line; values are sampled bilinearly.
5. Geoid offset: Copernicus DEM heights are on EGM2008, the geometry needs ellipsoidal heights.
   The offset (constant over a job area) is found by geocoding a subsample with trial offsets and
   keeping the one where log β⁰ best correlates with a DEM simulation of the terrain's brightness
   (slopes facing the radar are bright). This also absorbs small orbit and timing biases.
6. Terrain flattening (Small 2011, area-based): γ⁰_T = β⁰ · A_β / A_γ, with A_β the radar pixel's
   area (azimuth × ground-range spacing × sin θ_ellipsoid) and A_γ the sum of the DEM facets that
   fall in that radar pixel, each projected perpendicular to the look direction. Layover:
   foreslope steeper than the incidence angle; shadow: backslope steeper than its complement.

Radiometry names follow the Sentinel Hub route: SIGMA0_ELLIPSOID (σ⁰ from the LUT, as Kuro Siwo),
GAMMA0_TERRAIN (γ⁰_T). Pixels outside every slice are NaN.
"""

from __future__ import annotations

import functools
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from scipy import ndimage as ndi

from dhara.adapters import cdse_s3
from dhara.adapters.allowlist import guarded_request
from dhara.adapters.cdse import Grid

STAC_ITEM_URL = "https://stac.dataspace.copernicus.eu/v1/collections/sentinel-1-grd/items/{}"
GRD_ENL = 4.4
_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


# ----------------------------------------------------------------------------- annotation


def _t(text: str) -> np.datetime64:
    return np.datetime64(text.strip(), "ns")


def _floats(text: str | None) -> np.ndarray:
    return np.array((text or "").split(), dtype=np.float64)


@dataclass
class Annotation:
    epoch: np.datetime64  # productFirstLineUtcTime
    line_dt: float  # s per line
    range_spacing: float  # m per ground-range pixel
    azimuth_spacing: float  # m per line
    n_lines: int
    n_samples: int
    orbit_t: np.ndarray  # s after epoch
    orbit_pos: np.ndarray  # (n, 3) ECEF m
    orbit_vel: np.ndarray  # (n, 3) m/s
    srgr_t: np.ndarray  # s after epoch
    srgr_sr0: np.ndarray
    srgr_coef: np.ndarray  # (n, k)
    geo: dict[str, np.ndarray]  # geolocation grid: line, pixel, lat, lon, height, inc

    def secs(self, t: np.datetime64) -> float:
        return float((t - self.epoch) / np.timedelta64(1, "ns")) * 1e-9


def parse_annotation(xml: bytes) -> Annotation:
    r = ET.fromstring(xml)
    info = r.find("imageAnnotation/imageInformation")
    assert info is not None
    epoch = _t(info.findtext("productFirstLineUtcTime", ""))

    def secs(text: str) -> float:
        return float((_t(text) - epoch) / np.timedelta64(1, "ns")) * 1e-9

    orb = r.findall("generalAnnotation/orbitList/orbit")
    ot = np.array([secs(o.findtext("time", "")) for o in orb])
    pos = np.array([[float(o.findtext(f"position/{a}", "nan")) for a in "xyz"] for o in orb])
    vel = np.array([[float(o.findtext(f"velocity/{a}", "nan")) for a in "xyz"] for o in orb])
    cc = r.findall("coordinateConversion/coordinateConversionList/coordinateConversion")
    st = np.array([secs(c.findtext("azimuthTime", "")) for c in cc])
    sr0 = np.array([float(c.findtext("sr0", "nan")) for c in cc])
    coef = np.array([_floats(c.findtext("srgrCoefficients")) for c in cc])
    pts = r.findall("geolocationGrid/geolocationGridPointList/geolocationGridPoint")
    geo = {
        k: np.array([float(p.findtext(tag, "nan")) for p in pts])
        for k, tag in (
            ("line", "line"),
            ("pixel", "pixel"),
            ("lat", "latitude"),
            ("lon", "longitude"),
            ("height", "height"),
            ("inc", "incidenceAngle"),
        )
    }
    return Annotation(
        epoch=epoch,
        line_dt=float(info.findtext("azimuthTimeInterval", "nan")),
        range_spacing=float(info.findtext("rangePixelSpacing", "nan")),
        azimuth_spacing=float(info.findtext("azimuthPixelSpacing", "nan")),
        n_lines=int(info.findtext("numberOfLines", "0")),
        n_samples=int(info.findtext("numberOfSamples", "0")),
        orbit_t=ot,
        orbit_pos=pos,
        orbit_vel=vel,
        srgr_t=st,
        srgr_sr0=sr0,
        srgr_coef=coef,
        geo=geo,
    )


@dataclass
class Lut:
    lines: np.ndarray  # (n,)
    pixels: np.ndarray  # (n, m)
    values: np.ndarray  # (n, m)

    def window(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Bilinear values on the (rows × cols) window (radar geometry)."""
        per_vec = np.stack(
            [np.interp(cols, self.pixels[i], self.values[i]) for i in range(len(self.lines))]
        )
        i = np.clip(np.searchsorted(self.lines, rows) - 1, 0, len(self.lines) - 2)
        l0, l1 = self.lines[i], self.lines[i + 1]
        w = np.clip((rows - l0) / np.maximum(l1 - l0, 1e-9), 0.0, 1.0)[:, None]
        return ((1 - w) * per_vec[i] + w * per_vec[i + 1]).astype(np.float32)


def _vectors(r: ET.Element, path: str, value_tag: str) -> Lut:
    vs = r.findall(path)
    lines = np.array([float(v.findtext("line", "nan")) for v in vs])
    pixels = [_floats(v.findtext("pixel")) for v in vs]
    values = [_floats(v.findtext(value_tag)) for v in vs]
    m = min(len(p) for p in pixels)  # all vectors share one pixel list in practice
    return Lut(lines, np.array([p[:m] for p in pixels]), np.array([v[:m] for v in values]))


def parse_calibration(xml: bytes) -> dict[str, Lut]:
    r = ET.fromstring(xml)
    path = "calibrationVectorList/calibrationVector"
    return {k: _vectors(r, path, k) for k in ("sigmaNought", "betaNought")}


def noise_window(xml: bytes, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Thermal noise power (DN²) on the window: range LUT × azimuth LUT (IPF ≥ 2.9)."""
    r = ET.fromstring(xml)
    rng = _vectors(r, "noiseRangeVectorList/noiseRangeVector", "noiseRangeLut").window(rows, cols)
    az = np.ones_like(rng)
    for b in r.findall("noiseAzimuthVectorList/noiseAzimuthVector"):
        l0, l1 = int(b.findtext("firstAzimuthLine", "0")), int(b.findtext("lastAzimuthLine", "0"))
        c0, c1 = int(b.findtext("firstRangeSample", "0")), int(b.findtext("lastRangeSample", "0"))
        lines, lut = _floats(b.findtext("line")), _floats(b.findtext("noiseAzimuthLut"))
        ri = (rows >= l0) & (rows <= l1)
        ci = (cols >= c0) & (cols <= c1)
        if not ri.any() or not ci.any():
            continue
        vals = np.interp(rows[ri], lines, lut) if len(lines) > 1 else np.full(ri.sum(), lut[0])
        az[np.ix_(ri, ci)] = vals[:, None]
    return rng * az


# ----------------------------------------------------------------------------- geometry


class Orbit:
    """Polynomial fit of the annotation state vectors (ECEF), position and velocity."""

    def __init__(self, t: np.ndarray, pos: np.ndarray, vel: np.ndarray):
        self.t0, self.ts = float(t.mean()), float(max(t.std(), 1.0))
        deg = min(8, len(t) - 1)
        x = (t - self.t0) / self.ts
        self.pp = [np.polynomial.Polynomial.fit(x, pos[:, k], deg) for k in range(3)]
        self.vp = [np.polynomial.Polynomial.fit(x, vel[:, k], deg) for k in range(3)]
        self.ap = [p.deriv() for p in self.vp]

    def state(self, t: np.ndarray):
        x = (t - self.t0) / self.ts
        p = np.stack([f(x) for f in self.pp], -1)
        v = np.stack([f(x) for f in self.vp], -1)
        a = np.stack([f(x) for f in self.ap], -1) / self.ts
        return p, v, a


def zero_doppler(orbit: Orbit, target: np.ndarray, t_init: np.ndarray, iters: int = 8):
    """Azimuth time where the sensor velocity is perpendicular to the line of sight (Newton)."""
    t = t_init.copy()
    for _ in range(iters):
        p, v, a = orbit.state(t)
        d = p - target
        f = np.einsum("ij,ij->i", d, v)
        fp = np.einsum("ij,ij->i", v, v) + np.einsum("ij,ij->i", d, a)
        dt = f / fp
        t = t - dt
        if np.nanmax(np.abs(dt)) < 1e-7:  # converged (sub-millimetre along track)
            break
    p, v, _ = orbit.state(t)
    return t, p, v


def slant_to_pixel(ann: Annotation, t: np.ndarray, slant: np.ndarray) -> np.ndarray:
    """GRD range pixel of slant range `slant` (m) at azimuth time `t` (SRGR polynomials, linearly
    interpolated between annotation records)."""
    i = np.clip(np.searchsorted(ann.srgr_t, t) - 1, 0, len(ann.srgr_t) - 2)
    w = np.clip((t - ann.srgr_t[i]) / (ann.srgr_t[i + 1] - ann.srgr_t[i]), 0, 1)[:, None]
    coef = (1 - w) * ann.srgr_coef[i] + w * ann.srgr_coef[i + 1]
    sr0 = (1 - w[:, 0]) * ann.srgr_sr0[i] + w[:, 0] * ann.srgr_sr0[i + 1]
    d = slant - sr0
    ground = np.zeros_like(d)
    for k in range(coef.shape[1] - 1, -1, -1):  # Horner
        ground = ground * d + coef[:, k]
    return ground / ann.range_spacing


def lee(img: np.ndarray, size: int = 5, enl: float = GRD_ENL) -> np.ndarray:
    """Lee filter on intensity (radar geometry). NaN-safe: NaNs stay NaN."""
    ok = np.isfinite(img)
    x = np.where(ok, img, 0.0).astype(np.float32)  # float32 halves memory on large windows
    n = ndi.uniform_filter(ok.astype(np.float32), size)
    m = ndi.uniform_filter(x, size) / np.maximum(n, 1e-9)
    m2 = ndi.uniform_filter(x * x, size) / np.maximum(n, 1e-9)
    var = np.maximum(m2 - m * m, 0.0)
    cu2 = 1.0 / enl
    var_x = np.maximum((var - m * m * cu2) / (1 + cu2), 0.0)
    k = np.where(var > 0, var_x / np.maximum(var, 1e-30), 0.0)
    out = m + k * (x - m)
    return np.where(ok, out, np.nan).astype(np.float32)


# ----------------------------------------------------------------------------- per slice


@dataclass
class Geometry:
    line: np.ndarray  # fractional GRD line per output pixel (NaN = outside the slice)
    pixel: np.ndarray
    cos_loc: np.ndarray  # cos local incidence
    cos_psi: np.ndarray  # cos projection angle
    lia_deg: np.ndarray
    layover_shadow: np.ndarray  # bool
    area_gamma: np.ndarray  # facet area projected perpendicular to the look direction (m²)
    sin_ell: np.ndarray  # sin of the ellipsoid incidence angle


def _enu_basis(lon: np.ndarray, lat: np.ndarray):
    lo, la = np.radians(lon), np.radians(lat)
    e = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    n = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    u = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    return e, n, u


def dem_gradients(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(d/dy, d/dx, fill height) of the DEM on the grid (m/m); computed once per pass."""
    fill = float(np.nanmedian(dem))
    gy_, gx_ = np.gradient(np.where(np.isfinite(dem), dem, fill).astype(np.float32), res)
    return gy_, gx_, fill


def geocode(
    ann: Annotation,
    orbit: Orbit,
    grid: Grid,
    dem: np.ndarray,
    offset_m: float,
    step: int = 1,
    rows_range: tuple[int, int] | None = None,
    grads: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> Geometry:
    """Range-Doppler geometry of the output grid (every `step`-th pixel; rows `rows_range` only, to
    bound memory on large areas). `dem`: heights on `grid` (m, EGM2008); `offset_m` added to get
    ellipsoidal heights."""
    r_a, r_b = rows_range or (0, grid.height)
    gy_, gx_, fill_h = grads or dem_gradients(dem, grid.res)
    rows = np.arange(r_a, r_b, step)
    cols = np.arange(0, grid.width, step)
    cc, rr = np.meshgrid(cols, rows)
    x, y = grid.to_world(cc + 0.5, rr + 0.5)
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(np.ravel(x), np.ravel(y))
    h = np.ravel(dem[r_a:r_b:step, ::step]).astype(np.float64)
    valid = np.isfinite(h)
    h = np.where(valid, h, fill_h) + offset_m
    tx, ty, tz = _ECEF.transform(lon, lat, h)
    target = np.stack([tx, ty, tz], -1)
    # Initial azimuth time from the geolocation grid (line as a function of position).
    g = ann.geo
    gx, gy = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(
        g["lon"], g["lat"]
    )
    A = np.c_[gx, gy, np.ones_like(gx)]
    cl, *_ = np.linalg.lstsq(A, g["line"], rcond=None)
    line0 = np.c_[np.ravel(x), np.ravel(y), np.ones(x.size)] @ cl
    t, p, v = zero_doppler(orbit, target, line0 * ann.line_dt)
    look = p - target
    slant = np.linalg.norm(look, axis=-1)
    lhat = look / slant[:, None]
    line = t / ann.line_dt
    pixel = slant_to_pixel(ann, t, slant)
    inside = (line >= 0) & (line <= ann.n_lines - 1) & (pixel >= 0) & (pixel <= ann.n_samples - 1)

    dzdx = np.ravel(gx_[r_a:r_b:step, ::step]).astype(np.float64)
    dzdy = -np.ravel(gy_[r_a:r_b:step, ::step]).astype(np.float64)  # rows run south
    return _terrain_geometry(
        np.asarray(lon), np.asarray(lat), line, pixel, lhat, v, dzdx, dzdy,
        inside & valid, rr.shape, grid.res * step,
    )  # fmt: skip


def _terrain_geometry(lon, lat, line, pixel, lhat, v, dzdx, dzdy, ok, shp, cell_m) -> Geometry:
    """Local incidence, projection angle, layover/shadow and facet areas from the look vector
    (ECEF, target→sensor), sensor velocity and DEM slopes (east, north; m/m)."""
    e, nn, u = _enu_basis(lon, lat)
    normal = -dzdx[:, None] * e - dzdy[:, None] * nn + u
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    cos_loc = np.einsum("ij,ij->i", normal, lhat)
    img_n = np.cross(v / np.linalg.norm(v, axis=-1, keepdims=True), lhat)
    img_n /= np.linalg.norm(img_n, axis=-1, keepdims=True)
    cos_psi = np.abs(np.einsum("ij,ij->i", normal, img_n))
    # Layover / shadow from the slope along the horizontal look direction.
    l_e, l_n, l_u = (np.einsum("ij,ij->i", lhat, b) for b in (e, nn, u))
    theta = np.arccos(np.clip(l_u, -1, 1))  # ellipsoid incidence
    hz = np.hypot(l_e, l_n)
    rise_toward_sensor = (dzdx * l_e + dzdy * l_n) / np.maximum(hz, 1e-9)
    alpha = np.arctan(rise_toward_sensor)  # > 0: terrain rises toward the sensor (backslope)
    layover = -alpha > theta
    shadow = alpha > (np.pi / 2 - theta)
    nan = np.nan
    return Geometry(
        line=np.where(ok, line, nan).reshape(shp),
        pixel=np.where(ok, pixel, nan).reshape(shp),
        cos_loc=cos_loc.reshape(shp),
        cos_psi=cos_psi.reshape(shp),
        lia_deg=np.degrees(np.arccos(np.clip(cos_loc, -1, 1))).reshape(shp),
        layover_shadow=(layover | shadow | (cos_loc <= 0)).reshape(shp),
        # Facet area (cell² / cos slope) projected onto the plane perpendicular to the look vector.
        area_gamma=(
            cell_m**2
            / np.clip(np.einsum("ij,ij->i", normal, u), 0.05, None)
            * np.clip(cos_loc, 0.0, None)
        ).reshape(shp),
        sin_ell=np.sin(theta).reshape(shp),
    )


class LatticeGeocoder:
    """Range-Doppler geometry solved exactly on a coarse lattice (every `step` pixels) at two
    ellipsoidal heights spanning the area's terrain, then interpolated to every pixel: smoothly in
    position, linearly in height (a height change moves a target's radar position almost linearly;
    checked against the exact
    solution on a real 28 Aug 2026 scene: within 0.07 pixel, identical layover/shadow mask). Same results as `geocode`, far less
    work on large areas."""

    def __init__(
        self,
        ann: Annotation,
        orbit: Orbit,
        grid: Grid,
        dem: np.ndarray,
        offset_m: float,
        step: int = 8,
    ):
        self.ann, self.grid, self.step, self.offset = ann, grid, step, offset_m
        hmin, hmax = float(np.nanmin(dem)), float(np.nanmax(dem))
        self.h0, self.h1 = hmin + offset_m - 100.0, hmax + offset_m + 100.0
        rows = np.arange(0, grid.height + step, step)
        cols = np.arange(0, grid.width + step, step)
        cc, rr = np.meshgrid(cols, rows)
        x, y = grid.to_world(cc + 0.5, rr + 0.5)
        lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(
            np.ravel(x), np.ravel(y)
        )
        g = ann.geo
        gx, gy = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(
            g["lon"], g["lat"]
        )
        cl, *_ = np.linalg.lstsq(np.c_[gx, gy, np.ones_like(gx)], g["line"], rcond=None)
        t_init = (np.c_[np.ravel(x), np.ravel(y), np.ones(x.size)] @ cl) * ann.line_dt
        shp = rr.shape
        fields: dict[str, np.ndarray] = {
            "lon": np.asarray(lon).reshape(shp),
            "lat": np.asarray(lat).reshape(shp),
        }
        for tag, h in (("0", self.h0), ("1", self.h1)):
            tx, ty, tz = _ECEF.transform(lon, lat, np.full(len(t_init), h))
            target = np.stack([tx, ty, tz], -1)
            t, p, v = zero_doppler(orbit, target, t_init)
            look = p - target
            slant = np.linalg.norm(look, axis=-1)
            fields["line" + tag] = (t / ann.line_dt).reshape(shp)
            fields["pixel" + tag] = slant_to_pixel(ann, t, slant).reshape(shp)
            for k in range(3):
                fields[f"l{k}_{tag}"] = (look[:, k] / slant).reshape(shp)
                fields[f"v{k}_{tag}"] = v[:, k].reshape(shp)
        self.fields = fields

    def block(self, r_a: int, r_b: int, dem: np.ndarray, grads) -> Geometry:
        """Full-resolution geometry of rows r_a..r_b (local UTM grid)."""
        gy_, gx_, fill_h = grads
        rows, cols = np.arange(r_a, r_b), np.arange(self.grid.width)
        cc, rr = np.meshgrid(cols, rows)
        coords = np.stack([np.ravel(rr) / self.step, np.ravel(cc) / self.step])

        def at(name: str) -> np.ndarray:
            return ndi.map_coordinates(self.fields[name], coords, order=1, mode="nearest")

        h = np.ravel(dem[r_a:r_b]).astype(np.float64)
        valid = np.isfinite(h)
        w = (np.where(valid, h, fill_h) + self.offset - self.h0) / (self.h1 - self.h0)
        line = at("line0") + w * (at("line1") - at("line0"))
        pixel = at("pixel0") + w * (at("pixel1") - at("pixel0"))
        lhat = np.stack([at(f"l{k}_0") + w * (at(f"l{k}_1") - at(f"l{k}_0")) for k in range(3)], -1)
        lhat /= np.linalg.norm(lhat, axis=-1, keepdims=True)
        v = np.stack([at(f"v{k}_0") for k in range(3)], -1)
        ann = self.ann
        inside = (
            (line >= 0) & (line <= ann.n_lines - 1) & (pixel >= 0) & (pixel <= ann.n_samples - 1)
        )
        dzdx = np.ravel(gx_[r_a:r_b]).astype(np.float64)
        dzdy = -np.ravel(gy_[r_a:r_b]).astype(np.float64)
        return _terrain_geometry(
            at("lon"),
            at("lat"),
            line,
            pixel,
            lhat,
            v,
            dzdx,
            dzdy,
            inside & valid,
            rr.shape,
            self.grid.res,
        )


def splat_area(acc: np.ndarray, win, line: np.ndarray, pixel: np.ndarray, area: np.ndarray):
    """Add facet areas bilinearly into the radar-window accumulator (Small 2011)."""
    r0, _, c0, _ = win
    ok = np.isfinite(line) & np.isfinite(area)
    li, pi_, a = line[ok] - r0, pixel[ok] - c0, area[ok]
    i0, j0 = np.floor(li).astype(int), np.floor(pi_).astype(int)
    fi, fj = li - i0, pi_ - j0
    for di, dj, w in (
        (0, 0, (1 - fi) * (1 - fj)),
        (1, 0, fi * (1 - fj)),
        (0, 1, (1 - fi) * fj),
        (1, 1, fi * fj),
    ):
        ii, jj = np.clip(i0 + di, 0, acc.shape[0] - 1), np.clip(j0 + dj, 0, acc.shape[1] - 1)
        np.add.at(acc, (ii, jj), a * w)


def flattening_area(geom: Geometry, win) -> np.ndarray:
    """Small (2011) area-based terrain flattening: the γ-projected areas of all DEM facets that
    fall in each radar pixel, summed (bilinear splat on the radar window, 3×3 smoothing against
    aliasing), then read back at every output pixel."""
    r0, r1, c0, c1 = win
    acc = np.zeros((r1 - r0 + 1, c1 - c0 + 1), np.float64)
    ok = np.isfinite(geom.line)
    li, pi_, a = geom.line[ok] - r0, geom.pixel[ok] - c0, geom.area_gamma[ok]
    i0, j0 = np.floor(li).astype(int), np.floor(pi_).astype(int)
    fi, fj = li - i0, pi_ - j0
    for di, dj, w in (
        (0, 0, (1 - fi) * (1 - fj)),
        (1, 0, fi * (1 - fj)),
        (0, 1, (1 - fi) * fj),
        (1, 1, fi * fj),
    ):
        ii, jj = np.clip(i0 + di, 0, acc.shape[0] - 1), np.clip(j0 + dj, 0, acc.shape[1] - 1)
        np.add.at(acc, (ii, jj), a * w)
    acc = ndi.uniform_filter(acc, 3)
    return _sample(acc.astype(np.float32), r0, c0, geom.line, geom.pixel)


def _sample(img: np.ndarray, row0: int, col0: int, line: np.ndarray, pixel: np.ndarray):
    ok = np.isfinite(line)
    coords = np.stack([np.where(ok, line - row0, 0), np.where(ok, pixel - col0, 0)])
    out = ndi.map_coordinates(np.nan_to_num(img, nan=0.0), coords, order=1, mode="constant", cval=0)
    valid = ndi.map_coordinates(np.isfinite(img).astype(np.float32), coords, order=1, cval=0)
    return np.where(ok & (valid > 0.99), out, np.nan).astype(np.float32)


@dataclass
class Slice:
    ann: Annotation
    orbit: Orbit
    assets: dict[str, str]


@functools.lru_cache(maxsize=32)
def _slice(scene_id: str) -> Slice:
    r = guarded_request("GET", STAC_ITEM_URL.format(scene_id), timeout_s=60)
    if r.status_code != 200:
        raise RuntimeError(f"STAC item {scene_id} not found ({r.status_code})")
    assets = {
        k: v["href"] for k, v in r.json()["assets"].items() if v.get("href", "").startswith("s3://")
    }
    ann = parse_annotation(cdse_s3.get_bytes(assets["schema-product-vv"]))
    return Slice(ann, Orbit(ann.orbit_t, ann.orbit_pos, ann.orbit_vel), assets)


def _window(geom: Geometry, ann: Annotation, margin: int = 8):
    ok = np.isfinite(geom.line)
    if not ok.any():
        return None
    r0 = max(int(np.floor(np.nanmin(geom.line))) - margin, 0)
    r1 = min(int(np.ceil(np.nanmax(geom.line))) + margin, ann.n_lines - 1)
    c0 = max(int(np.floor(np.nanmin(geom.pixel))) - margin, 0)
    c1 = min(int(np.ceil(np.nanmax(geom.pixel))) + margin, ann.n_samples - 1)
    return r0, r1, c0, c1


CAL_CHUNK = 1024  # radar lines per calibration chunk (memory bound on large windows)
LEE_HALO = 2  # half of the 5×5 Lee window: overlap between chunks, so the seams are exact


def _calibrated(sl: Slice, pol: str, win, decim: int = 1) -> dict[str, np.ndarray]:
    """Noise-removed, calibrated, Lee-filtered σ⁰ and β⁰ on the radar window, processed in chunks
    of CAL_CHUNK lines (with the Lee window's overlap) so large windows fit in memory. `decim` > 1
    reads the product's overviews instead (every `decim`-th line and pixel)."""
    r0, r1, c0, c1 = win
    cal = parse_calibration(cdse_s3.get_bytes(sl.assets[f"schema-calibration-{pol}"]))
    noise_xml = cdse_s3.get_bytes(sl.assets[f"schema-noise-{pol}"])
    half = (decim - 1) / 2

    def chunk(dn: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> dict[str, np.ndarray]:
        noise = noise_window(noise_xml, rows, cols)
        power = np.maximum(dn * dn - noise, 0.0)
        power[dn == 0] = np.nan  # no-data border
        del noise
        out = {}
        for name, key in (("sigma0", "sigmaNought"), ("beta0", "betaNought")):
            a = cal[key].window(rows, cols)
            out[name] = lee(np.maximum(power / (a * a), 1e-6))
        return out

    with cdse_s3.open_measurement(sl.assets[pol]) as ds:
        if decim > 1:
            shp = (-(-(r1 + 1 - r0) // decim), -(-(c1 + 1 - c0) // decim))
            dn = ds.read(1, window=((r0, r1 + 1), (c0, c1 + 1)), out_shape=shp).astype(np.float32)
            rows = r0 + np.arange(dn.shape[0], dtype=np.float64) * decim + half
            cols = c0 + np.arange(dn.shape[1], dtype=np.float64) * decim + half
            return chunk(dn, rows, cols)
        n_rows, n_cols = r1 + 1 - r0, c1 + 1 - c0
        cols = c0 + np.arange(n_cols, dtype=np.float64)
        res = {k: np.empty((n_rows, n_cols), np.float32) for k in ("sigma0", "beta0")}
        for a in range(0, n_rows, CAL_CHUNK):
            b = min(a + CAL_CHUNK, n_rows)
            ra, rb = max(a - LEE_HALO, 0), min(b + LEE_HALO, n_rows)
            dn = ds.read(1, window=((r0 + ra, r0 + rb), (c0, c1 + 1))).astype(np.float32)
            part = chunk(dn, r0 + np.arange(ra, rb, dtype=np.float64), cols)
            for k, v in part.items():
                res[k][a:b] = v[a - ra : a - ra + (b - a)]
        return res


OFFSET_DECIM = 1  # overview reads (2, 4) bias the offset by 5–10 m in tests; keep full res


def estimate_offset(sl: Slice, grid: Grid, dem: np.ndarray, step: int = 4) -> tuple[float, float]:
    """Geoid + bias offset (m) maximising corr(log β⁰, terrain brightness simulation).

    A height change moves a target's radar position almost linearly over ±100 m, so the geometry
    is solved at two offsets and interpolated for every trial offset; β⁰ is read at a quarter of
    full resolution (the product's overviews). Both are ample for a correlation search."""
    grads = dem_gradients(dem, grid.res)
    lo, hi = -80.0, 40.0
    g0 = geocode(sl.ann, sl.orbit, grid, dem, lo, step, grads=grads)
    g1 = geocode(sl.ann, sl.orbit, grid, dem, hi, step, grads=grads)
    win = _window(g0, sl.ann, margin=120)
    if win is None:
        return 0.0, float("nan")
    r0, _, c0, _ = win
    decim = OFFSET_DECIM
    cal_vv = _calibrated(sl, "vv", win, decim=decim)
    if decim == 1:
        _vv_cache[(id(sl), grid.bounds)] = (win, cal_vv)  # reused by the pass (no second read)
    beta = cal_vv["beta0"]
    half = (decim - 1) / 2
    simlog = np.log(np.clip(g0.cos_loc, 0.05, None) / np.clip(g0.cos_psi, 0.05, None))
    shadow = g0.layover_shadow

    def corr(off: float) -> float:
        w = (off - lo) / (hi - lo)
        line = g0.line + w * (g1.line - g0.line)
        pixel = g0.pixel + w * (g1.pixel - g0.pixel)
        b = _sample(beta, 0, 0, (line - r0 - half) / decim, (pixel - c0 - half) / decim)
        ok = np.isfinite(b) & np.isfinite(simlog) & ~shadow
        if ok.sum() < 500:
            return -1.0
        return float(np.corrcoef(np.log(b[ok]), simlog[ok])[0, 1])

    coarse = np.arange(lo, hi + 1.0, 8.0)
    c = [corr(o) for o in coarse]
    best = float(coarse[int(np.argmax(c))])
    fine = np.arange(best - 8.0, best + 8.1, 1.0)
    f = [corr(o) for o in fine]
    return float(fine[int(np.argmax(f))]), float(max(f))


# The offset search's calibrated VV window, reused when processing the same slice and grid.
_vv_cache: dict[tuple, tuple] = {}


def _crop(cal: dict[str, np.ndarray], big, win) -> dict[str, np.ndarray] | None:
    """`cal` (on window `big`) cut to `win`, or None when `big` does not contain it."""
    b0, b1, bc0, bc1 = big
    r0, r1, c0, c1 = win
    if r0 < b0 or r1 > b1 or c0 < bc0 or c1 > bc1:
        return None
    return {k: v[r0 - b0 : r1 - b0 + 1, c0 - bc0 : c1 - bc0 + 1] for k, v in cal.items()}


# Height offset per (slice, grid): the same pass is processed for γ⁰ and σ⁰, fit it once.
_offsets: dict[tuple, tuple[float, float]] = {}
_offset_lock = threading.Lock()


# The chain holds a pass's radar window and geometry in memory; at most two passes at once so a
# large area cannot exhaust memory when the pipeline fetches all passes in parallel.
_pass_slots = threading.BoundedSemaphore(2)
_slot_guard = threading.Lock()
BLOCK_ROWS = 256
LARGE_GRID_PX = 6_000_000  # ~600 km² at 10 m


def process_pass(
    scene_ids: list[str],
    grid: Grid,
    dem: np.ndarray,
    radiometry: str,
    with_geometry_bands: bool = False,
) -> tuple[np.ndarray, dict]:
    """Backscatter of one pass (all its slices) on `grid`: (bands, H, W) float32 linear, and a
    provenance dict (offset, correlation, slices)."""
    both, meta = process_pass_all(scene_ids, grid, dem)
    arr = both[radiometry]
    return (arr if with_geometry_bands else arr[:2]), meta | {"radiometry": radiometry}


def process_pass_all(
    scene_ids: list[str], grid: Grid, dem: np.ndarray
) -> tuple[dict[str, np.ndarray], dict]:
    """Both radiometries of one pass in one go (they share reads, calibration and geometry):
    {"GAMMA0_TERRAIN" | "SIGMA0_ELLIPSOID": (VV, VH, localIncidenceAngle, shadowMask)}.
    Geocoded in row blocks; at most two passes at a time (memory bounded)."""
    # A large area takes both slots: it runs alone.
    slots = 2 if grid.height * grid.width > LARGE_GRID_PX else 1
    with _slot_guard:  # take all needed slots at once: two large passes cannot each hold one
        for _ in range(slots):
            _pass_slots.acquire()
    try:
        return _process_pass(scene_ids, grid, dem)
    finally:
        for _ in range(slots):
            _pass_slots.release()


def _process_pass(scene_ids, grid, dem):
    shape = (grid.height, grid.width)
    out = {
        r: np.full((2, *shape), np.nan, np.float32) for r in ("GAMMA0_TERRAIN", "SIGMA0_ELLIPSOID")
    }
    lia = np.full(shape, np.nan, np.float32)
    mask = np.full(shape, np.nan, np.float32)
    meta: dict = {"slices": []}
    grads = dem_gradients(dem, grid.res)
    for sid in scene_ids:
        sl = _slice(sid)
        key = (sid, grid.bounds, grid.res)
        with _offset_lock:
            cached = _offsets.get(key)
        if cached is None:
            cached = estimate_offset(sl, grid, dem)
            with _offset_lock:
                _offsets[key] = cached
        offset, r = cached
        meta["slices"].append({"id": sid, "height_offset_m": offset, "terrain_corr": round(r, 3)})
        win = _window(geocode(sl.ann, sl.orbit, grid, dem, offset, 8, grads=grads), sl.ann, 40)
        if win is None:
            continue
        r0, r1, c0, c1 = win
        line = np.full(shape, np.nan, np.float32)
        pixel = np.full(shape, np.nan, np.float32)
        sin_ell = np.full(shape, np.nan, np.float32)
        acc = np.zeros((r1 - r0 + 1, c1 - c0 + 1), np.float32)
        lattice = LatticeGeocoder(sl.ann, sl.orbit, grid, dem, offset)
        for b0 in range(0, grid.height, BLOCK_ROWS):
            b1 = min(b0 + BLOCK_ROWS, grid.height)
            g = lattice.block(b0, b1, dem, grads)
            line[b0:b1], pixel[b0:b1], sin_ell[b0:b1] = g.line, g.pixel, g.sin_ell
            got = np.isfinite(g.line) & np.isnan(lia[b0:b1])
            lia[b0:b1][got] = g.lia_deg[got]
            mask[b0:b1][got] = g.layover_shadow[got].astype(np.float32)
            splat_area(acc, win, g.line, g.pixel, g.area_gamma)
        a_gamma = _sample(ndi.uniform_filter(acc, 3).astype(np.float32), r0, c0, line, pixel)
        del acc
        a_beta = sl.ann.azimuth_spacing * sl.ann.range_spacing * sin_ell
        for b, pol in enumerate(("vv", "vh")):
            if pol not in sl.assets:
                continue
            cal = None
            if pol == "vv" and (hit := _vv_cache.pop((id(sl), grid.bounds), None)):
                cal = _crop(hit[1], hit[0], win)
            cal = cal or _calibrated(sl, pol, win)
            beta = _sample(cal["beta0"], r0, c0, line, pixel)
            vals = {
                "GAMMA0_TERRAIN": np.where(
                    a_gamma > 1.0, beta * a_beta / np.maximum(a_gamma, 1.0), np.nan
                ),
                "SIGMA0_ELLIPSOID": _sample(cal["sigma0"], r0, c0, line, pixel),
            }
            del cal, beta
            for rad, val in vals.items():
                o = out[rad][b]
                fill = np.isnan(o) & np.isfinite(val)
                o[fill] = val[fill]
    both = {r: np.concatenate([a, lia[None], mask[None]]) for r, a in out.items()}
    return both, meta
