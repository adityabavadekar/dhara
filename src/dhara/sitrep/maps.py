"""Static report maps (PNG, base64) for the situation report.

All drawing happens in the job's local UTM CRS (the CRS of classes.tif); vector layers are
reprojected to it. Settlement and repair markers carry the rank numbers used in the report's
tables instead of names, because matplotlib cannot shape Devanagari text.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from shapely.geometry import box, shape  # noqa: E402

from dhara.fusion import fuse as F  # noqa: E402

STATUS_COLOR = {
    "cut_off": "#b91c1c",
    "hospital_cut_off": "#ea580c",
    "detour": "#d97706",
    "connected": "#16a34a",
    "no_baseline_route": "#94a3b8",
    "not_on_network": "#cbd5e1",
}
BRIDGE_COLOR = {
    "likely_lost": "#b91c1c",
    "at_risk": "#ea580c",
    "likely_standing": "#334155",
    "not_observable": "#a8a29e",
}


@dataclass
class Layers:
    cls: np.ndarray
    extent: tuple[float, float, float, float]  # left, right, bottom, top (UTM m)
    crs: object
    aoi: gpd.GeoSeries | None
    roads: gpd.GeoDataFrame | None
    failed: gpd.GeoDataFrame | None
    settlements: gpd.GeoDataFrame | None
    hospitals: gpd.GeoDataFrame | None
    bridges: gpd.GeoDataFrame | None
    buildings: gpd.GeoDataFrame | None


def _read(p: Path, crs) -> gpd.GeoDataFrame | None:
    if not p.exists():
        return None
    g = gpd.read_parquet(p)
    if not len(g):
        return None
    return g.to_crs(crs) if g.crs is not None and g.crs != crs else g


def load(job_dir: Path, max_px: int = 1600) -> Layers:
    with rasterio.open(job_dir / "classes.tif") as ds:
        step = max(1, int(np.ceil(max(ds.width, ds.height) / max_px)))
        cls = ds.read(1)[::step, ::step]
        b = ds.bounds
        crs = ds.crs
    aoi = None
    prov = job_dir / "provenance.json"
    if prov.exists():
        geo = json.loads(prov.read_text()).get("inputs", {}).get("aoi", {}).get("geojson")
        if geo:
            aoi = gpd.GeoSeries([shape(geo)], crs=4326).to_crs(crs)
    return Layers(
        cls=cls,
        extent=(b.left, b.right, b.bottom, b.top),
        crs=crs,
        aoi=aoi,
        roads=_read(job_dir / "roads.parquet", crs),
        failed=_read(job_dir / "failed_edges.parquet", crs),
        settlements=_read(job_dir / "settlements.parquet", crs),
        hospitals=_read(job_dir / "hospitals.parquet", crs),
        bridges=_read(job_dir / "bridges.parquet", crs),
        buildings=_read(job_dir / "buildings.parquet", crs),
    )


def _hazard_rgba(cls: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 4), np.float32)
    lut[F.PERMANENT_WATER] = (0.49, 0.83, 0.99, 0.9)
    lut[F.FLOOD] = (0.11, 0.31, 0.85, 1.0)
    lut[F.DEBRIS] = (0.76, 0.47, 0.01, 1.0)
    lut[F.NOT_OBSERVABLE] = (0.47, 0.44, 0.42, 0.45)
    return lut[cls.astype(np.uint8)]


def _grow(cls: np.ndarray, k: int) -> np.ndarray:
    """Dilate small hazard patches so they stay visible after downsampling to page size."""
    if k <= 0:
        return cls
    from scipy.ndimage import binary_dilation

    out = cls.copy()
    for target in (F.FLOOD, F.DEBRIS):
        g = binary_dilation(cls == target, iterations=k)
        out[g & (out == F.NO_CHANGE)] = target
    return out


def _scalebar(ax, extent) -> None:
    left, right, bottom, top = extent
    width = right - left
    target = width / 5
    pow10 = 10 ** np.floor(np.log10(target))
    length = max(v for v in (1, 2, 5, 10) if v * pow10 <= target) * pow10
    x0, y0 = left + width * 0.04, bottom + (top - bottom) * 0.05
    ax.plot([x0, x0 + length], [y0, y0], color="#0f172a", lw=2.5, solid_capstyle="butt", zorder=20)
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    ax.text(
        x0 + length / 2,
        y0 + (top - bottom) * 0.015,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        color="#0f172a",
        zorder=20,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
    )
    nx, ny = right - width * 0.04, top - (top - bottom) * 0.1
    ax.annotate(
        "N",
        xy=(nx, ny),
        xytext=(nx, ny - (top - bottom) * 0.07),
        ha="center",
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "#0f172a", "lw": 1.2},
        zorder=20,
    )


def _numbered(ax, x: float, y: float, n: int, color: str, square: bool = False) -> None:
    ax.text(
        x,
        y,
        str(n),
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color="white",
        zorder=30,
        clip_on=True,
        bbox={
            "boxstyle": "square,pad=0.25" if square else "circle,pad=0.25",
            "fc": color,
            "ec": "white",
            "lw": 0.8,
        },
    )


def _png(fig) -> str:
    for ax in fig.axes:  # geopandas adds CRS axis labels; the scale bar carries the units
        ax.set_xlabel("")
        ax.set_ylabel("")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _size(extent, width: float = 7.2, max_h: float = 3.9) -> tuple[float, float]:
    """Figure size (in) keeping the extent's aspect, no taller than `max_h`."""
    ratio = (extent[3] - extent[2]) / max(extent[1] - extent[0], 1e-9)
    h = min(width * ratio, max_h)
    return (h / ratio, h)


def _base(L: Layers, extent, figsize, grow: int = 0):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f8fafc")
    ax.imshow(_hazard_rgba(_grow(L.cls, grow)), extent=L.extent, interpolation="nearest", zorder=2)
    if L.roads is not None:
        L.roads.plot(ax=ax, color="#94a3b8", linewidth=0.35, zorder=3)
    if L.aoi is not None:
        L.aoi.boundary.plot(ax=ax, color="#0284c7", linewidth=1.0, linestyle="--", zorder=4)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cbd5e1")
    return fig, ax


def _settlements(ax, L: Layers, ranked_ids: list[str], size: float = 9) -> None:
    s = L.settlements
    if s is None:
        return
    for status, color in STATUS_COLOR.items():
        sub = s[s["status"] == status]
        if len(sub):
            sub.plot(
                ax=ax,
                color=color,
                markersize=size if status in ("cut_off", "hospital_cut_off") else size * 0.55,
                edgecolor="white",
                linewidth=0.3,
                zorder=6,
            )
    where = {
        str(o): (g, str(st)) for o, g, st in zip(s["osm_id"], s.geometry, s["status"], strict=True)
    }
    for i, oid in enumerate(ranked_ids, start=1):
        if oid in where:
            g, st = where[oid]
            _numbered(ax, g.x, g.y, i, STATUS_COLOR.get(st, "#b91c1c"))


def _hospitals(ax, L: Layers) -> None:
    if L.hospitals is None:
        return
    xs, ys = L.hospitals.geometry.x, L.hospitals.geometry.y
    ax.scatter(xs, ys, s=38, c="white", edgecolors="#dc2626", linewidths=1.2, zorder=7)
    ax.scatter(xs, ys, s=16, marker="+", c="#dc2626", linewidths=1.2, zorder=8)


def overview(L: Layers, cut_ids: list[str]) -> str:
    """Whole area: hazard, failed links, settlements by access, hospitals; cut-off ranks numbered."""
    fig, ax = _base(L, L.extent, _size(L.extent), grow=2)
    if L.failed is not None:
        L.failed.plot(ax=ax, color="#dc2626", linewidth=0.9, zorder=5)
    _settlements(ax, L, cut_ids)
    _hospitals(ax, L)
    _scalebar(ax, L.extent)
    return _png(fig)


def hazard_window(L: Layers, km: float = 6.0) -> tuple[float, float, float, float] | None:
    """The square window (UTM extent) with the most flood and debris pixels, or None."""
    hz = np.isin(L.cls, (F.FLOOD, F.DEBRIS)).astype(np.float32)
    if hz.sum() == 0:
        return None
    left, right, bottom, top = L.extent
    px = (right - left) / L.cls.shape[1]
    w = max(3, int(km * 1000 / px))
    if w >= min(L.cls.shape):
        return L.extent
    ii = np.pad(hz.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = ii[w:, w:] - ii[:-w, w:] - ii[w:, :-w] + ii[:-w, :-w]
    r, c = np.unravel_index(int(np.argmax(sums)), sums.shape)
    x0, y1 = left + c * px, top - r * px
    return (x0, x0 + w * px, y1 - w * px, y1)


def hazard_zoom(L: Layers, window: tuple[float, float, float, float], bridge_ids: list[str]) -> str:
    """Close-up of the most affected window: hazard at full resolution, buildings hit, bridges."""
    fig, ax = _base(L, window, _size(window, max_h=3.9))
    clip = box(window[0], window[2], window[1], window[3])
    if L.buildings is not None:
        b = L.buildings[L.buildings.intersects(clip)]
        hit = b[b["status"] == "hit"]
        b[b["status"] != "hit"].plot(ax=ax, color="#cbd5e1", edgecolor="none", zorder=5)
        if len(hit):
            hit.plot(ax=ax, color="#b45309", edgecolor="#451a03", linewidth=0.2, zorder=6)
    if L.failed is not None:
        L.failed[L.failed.intersects(clip)].plot(ax=ax, color="#dc2626", linewidth=1.2, zorder=6)
    if L.bridges is not None:
        br = L.bridges[L.bridges.intersects(clip)]
        for status, color in BRIDGE_COLOR.items():
            sub = br[br["status"] == status]
            if len(sub):
                sub.plot(ax=ax, color=color, linewidth=3, zorder=7)
        ids = list(L.bridges["osm_id"])
        for i, oid in enumerate(bridge_ids, start=1):
            if oid in ids:
                g = L.bridges.iloc[ids.index(oid)].geometry.centroid
                if clip.contains(g):
                    _numbered(ax, g.x, g.y, i, "#0f172a", square=True)
    _settlements(ax, L, [], size=14)
    _hospitals(ax, L)
    _scalebar(ax, window)
    return _png(fig)


def access(L: Layers, cut_ids: list[str], repair_ids: list[str]) -> str:
    """Road access: failed links, settlements by access status, hospitals, top repairs numbered."""
    fig, ax = _base(L, L.extent, _size(L.extent, max_h=4.8), grow=1)
    if L.failed is not None:
        L.failed.plot(ax=ax, color="#dc2626", linewidth=1.1, zorder=5)
    _settlements(ax, L, cut_ids[:20])
    _hospitals(ax, L)
    if L.failed is not None and repair_ids:
        f = L.failed.drop_duplicates("osm_id").set_index("osm_id")
        for i, oid in enumerate(repair_ids, start=1):
            if oid in f.index:
                g = f.loc[oid].geometry.centroid
                _numbered(ax, g.x, g.y, i, "#16a34a", square=True)
    _scalebar(ax, L.extent)
    return _png(fig)


def flowpath(L: Layers | None, fp_dir: Path) -> str:
    """One traced flood path: corridor, path, numbered settlements; hazard underneath if mapped."""
    path = gpd.read_file(fp_dir / "flowpath.geojson")
    corr = gpd.read_file(fp_dir / "corridor.geojson")
    sett = gpd.read_file(fp_dir / "settlements.geojson")
    crs = L.crs if L is not None else path.estimate_utm_crs()
    path, corr = path.to_crs(crs), corr.to_crs(crs)
    sett = sett.to_crs(crs) if len(sett) else sett
    x0, y0, x1, y1 = corr.total_bounds
    pad = 0.06 * max(x1 - x0, y1 - y0)
    ext = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
    figsize = _size(ext, max_h=3.4)
    if L is not None:
        fig, ax = _base(L, ext, figsize, grow=1)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#f8fafc")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([])
        ax.set_yticks([])
    corr.plot(ax=ax, color="#7c3aed", alpha=0.1, zorder=3)
    path.plot(ax=ax, color="#7c3aed", linewidth=2, zorder=4)
    start = path.geometry.iloc[0].coords[0]
    ax.scatter(
        [start[0]], [start[1]], s=60, c="#7c3aed", edgecolors="white", linewidths=1.5, zorder=9
    )
    for _, r in sett.iterrows():
        _numbered(ax, r.geometry.x, r.geometry.y, int(r["order"]), "#7c3aed")
    _scalebar(ax, ext)
    return _png(fig)
