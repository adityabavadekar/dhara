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
    dem_hs: np.ndarray | None = None
    tier: np.ndarray | None = None  # confidence tier per pixel (0 none, 1 low, 2 moderate, 3 high)


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

    dem_hs = None
    dem_p = job_dir / "dem.tif"
    if dem_p.exists():
        try:
            with rasterio.open(dem_p) as ds_dem:
                dem_arr = ds_dem.read(1)[::step, ::step].astype(np.float32)
                from matplotlib.colors import LightSource

                dem_hs = LightSource(azdeg=315, altdeg=45).hillshade(dem_arr, vert_exag=1.5)
        except Exception:
            dem_hs = None

    tier = None
    if (job_dir / "confidence.tif").exists():
        with rasterio.open(job_dir / "confidence.tif") as ds_t:
            tier = ds_t.read(1)[::step, ::step]

    return Layers(
        tier=tier,
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
        dem_hs=dem_hs,
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


def _scalebar(ax, extent, k: float = 1.0) -> None:
    """Scale bar and north arrow. `k` = map units per ground metre (1 in UTM, 1/cos φ in 3857)."""
    left, right, bottom, top = extent
    width = right - left
    target = width / k / 5
    pow10 = 10 ** np.floor(np.log10(target))
    length = max(v for v in (1, 2, 5, 10) if v * pow10 <= target) * pow10
    x0, y0 = left + width * 0.04, bottom + (top - bottom) * 0.05
    label = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    length *= k
    ax.plot([x0, x0 + length], [y0, y0], color="#0f172a", lw=2.5, solid_capstyle="butt", zorder=20)
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


def _png(fig, dpi: int = 150) -> str:
    for ax in fig.axes:  # geopandas adds CRS axis labels; the scale bar carries the units
        ax.set_xlabel("")
        ax.set_ylabel("")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _size(extent, width: float = 7.2, max_h: float = 3.9) -> tuple[float, float]:
    """Figure size (in) keeping the extent's aspect, no taller than `max_h`."""
    ratio = (extent[3] - extent[2]) / max(extent[1] - extent[0], 1e-9)
    h = min(width * ratio, max_h)
    return (h / ratio, h)


def _base(L: Layers, extent, figsize, grow: int = 0, all_buildings: bool = True):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f1f5f9")
    if L.dem_hs is not None:
        ax.imshow(L.dem_hs, extent=L.extent, cmap="gray", alpha=0.30, zorder=1)
    ax.imshow(_hazard_rgba(_grow(L.cls, grow)), extent=L.extent, interpolation="nearest", zorder=2)
    if L.roads is not None:
        L.roads.plot(ax=ax, color="#64748b", linewidth=0.75, zorder=3)
    if L.buildings is not None:
        if all_buildings:  # off on whole-area maps: tens of thousands of dots hide everything else
            L.buildings.plot(ax=ax, color="#94a3b8", alpha=0.6, markersize=1.8, zorder=4)
        if "status" in L.buildings:
            hit = L.buildings[L.buildings["status"] == "hit"]
            if len(hit):
                hit.plot(ax=ax, color="#b45309", markersize=3.0, zorder=5)
    if L.failed is not None:
        L.failed.plot(ax=ax, color="#dc2626", linewidth=1.8, zorder=6)
    if L.bridges is not None:
        for status, color in BRIDGE_COLOR.items():
            sub = L.bridges[L.bridges["status"] == status]
            if len(sub):
                sub.plot(ax=ax, color=color, linewidth=2.4, zorder=7)
    if L.aoi is not None:
        L.aoi.boundary.plot(ax=ax, color="#0284c7", linewidth=1.0, linestyle="--", zorder=8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cbd5e1")
    return fig, ax


def _settlements(ax, L: Layers, ranked_ids: list[str], size: float = 14) -> None:
    s = L.settlements
    if s is None:
        return
    for status, color in STATUS_COLOR.items():
        sub = s[s["status"] == status]
        if len(sub):
            sub.plot(
                ax=ax,
                color=color,
                markersize=size
                if status in ("cut_off", "hospital_cut_off", "detour")
                else size * 0.5,
                edgecolor="white",
                linewidth=0.5,
                zorder=9,
            )
    # Only the ranked settlements are marked, with the rank numbers of the report's table (named
    # labels for every settlement made whole-area maps unreadable).
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
    ax.scatter(xs, ys, s=38, c="white", edgecolors="#dc2626", linewidths=1.2, zorder=10)
    ax.scatter(xs, ys, s=16, marker="+", c="#dc2626", linewidths=1.2, zorder=11)


def overview(L: Layers, cut_ids: list[str]) -> str:
    """Whole area, drawn to stay readable at page size (local UTM): terrain shading, mapped flood
    and debris (solid where corroborated, faint where low confidence), thin roads with impassable
    sections in red, communities that lost access, health facilities as small crosses, and the
    report table's top communities numbered. Every settlement and every building drawn at full
    weight hid the map on large areas."""
    # The road network extends past the analysed area, so ranked communities can lie outside it.
    # Widen the frame to include nearby ones, but by at most half the area's size: communities
    # further away are pinned to the frame edge with an arrow (the table gives their distance),
    # so a far-off village does not shrink the analysed area to a corner of the map.
    ext = list(L.extent)
    reach = 0.5 * max(ext[1] - ext[0], ext[3] - ext[2])
    lim = [ext[0] - reach, ext[1] + reach, ext[2] - reach, ext[3] + reach]
    if L.settlements is not None and cut_ids:
        pts = L.settlements[L.settlements["osm_id"].astype(str).isin(cut_ids)].geometry
        if len(pts):
            m = 600.0
            ext = [
                max(lim[0], min(ext[0], float(pts.x.min()) - m)),
                min(lim[1], max(ext[1], float(pts.x.max()) + m)),
                max(lim[2], min(ext[2], float(pts.y.min()) - m)),
                min(lim[3], max(ext[3], float(pts.y.max()) + m)),
            ]
    # Pad the shorter side to the map box's proportions (one report column, up to 4.4 in tall),
    # so the image fills the box instead of leaving empty bands beside it.
    W, H = 3.9, 4.4
    w, h = ext[1] - ext[0], ext[3] - ext[2]
    if h / w > H / W:
        grow = (h * W / H - w) / 2
        ext[0], ext[1] = ext[0] - grow, ext[1] + grow
    extent = (ext[0], ext[1], ext[2], ext[3])
    # Drawn at about its printed size and print resolution, so the page does not rescale it and
    # labels stay crisp.
    fig, ax = plt.subplots(figsize=_size(extent, width=W, max_h=H), dpi=300)
    ax.set_facecolor("#f8fafc")
    if L.dem_hs is not None:
        ax.imshow(L.dem_hs, extent=L.extent, cmap="gray", vmin=0, vmax=1, alpha=0.35, zorder=1)
    rgba = _hazard_rgba(L.cls)
    if L.tier is not None and L.tier.shape == L.cls.shape:
        low = np.isin(L.cls, (F.FLOOD, F.DEBRIS)) & (L.tier <= 1)
        rgba[low, 3] = 0.28
    ax.imshow(rgba, extent=L.extent, interpolation="nearest", zorder=2)
    if L.roads is not None:
        L.roads.plot(ax=ax, color="#64748b", linewidth=0.25, alpha=0.55, zorder=3)
    if L.failed is not None:
        L.failed.plot(ax=ax, color="#dc2626", linewidth=0.8, zorder=5)
    if L.aoi is not None:
        L.aoi.boundary.plot(ax=ax, color="#0284c7", linewidth=0.8, linestyle="--", zorder=6)
    if L.settlements is not None:
        st = L.settlements
        for status in ("hospital_cut_off", "cut_off"):
            sub = st[st["status"] == status]
            if len(sub):
                sub.plot(
                    ax=ax, color=STATUS_COLOR[status], markersize=7, edgecolor="white",
                    linewidth=0.3, zorder=8,
                )  # fmt: skip
    if L.hospitals is not None and len(L.hospitals):
        ax.scatter(
            L.hospitals.geometry.x, L.hospitals.geometry.y, s=9, marker="+", c="#991b1b",
            linewidths=0.7, zorder=9,
        )  # fmt: skip
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
    _settlements_numbered(ax, L, cut_ids)
    _scalebar(ax, extent)
    return _png(fig, dpi=300)


def _settlements_numbered(ax, L: Layers, ranked_ids: list[str]) -> None:
    """Rank numbers of the report's table at their settlements, drawn above everything else.
    Communities close together get their labels spread apart, with a leader line to each."""
    if L.settlements is None:
        return
    s = L.settlements
    where = {
        str(o): (g, str(st)) for o, g, st in zip(s["osm_id"], s.geometry, s["status"], strict=True)
    }
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    gap = 0.055 * max(x1 - x0, y1 - y0)  # minimum distance between two labels
    placed: list[tuple[float, float]] = []
    for i, oid in enumerate(ranked_ids, start=1):
        if oid not in where:
            continue
        g, st = where[oid]
        color = STATUS_COLOR.get(st, "#b91c1c")
        # Beyond the frame: pin the marker inside the edge and point an arrow towards the place.
        size = max(x1 - x0, y1 - y0)
        pin = 0.075 * size
        outside = not (x0 <= g.x <= x1 and y0 <= g.y <= y1)
        ex = min(max(g.x, x0 + pin), x1 - pin)
        ey = min(max(g.y, y0 + pin), y1 - pin)
        if outside:
            d = np.hypot(g.x - ex, g.y - ey)
            ux, uy = (g.x - ex) / d, (g.y - ey) / d
            ax.annotate(
                "", xy=(ex + ux * pin * 0.9, ey + uy * pin * 0.9),
                xytext=(ex + ux * pin * 0.3, ey + uy * pin * 0.3),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.0, "mutation_scale": 8},
                zorder=29,
            )  # fmt: skip
            g = type(g)(ex, ey)
        lx, ly = g.x, g.y
        step = 0
        while any(np.hypot(lx - px, ly - py) < gap for px, py in placed) and step < 12:
            step += 1  # walk outwards on alternating sides, away from the frame edge
            side = 1 if (g.x - x0) < (x1 - g.x) else -1
            lx = g.x + side * gap * (1 + (step - 1) // 2) * 0.9
            ly = g.y + (gap if step % 2 else -gap) * ((step + 1) // 2) * 0.6
        inset = 0.03 * max(x1 - x0, y1 - y0)
        lx = min(max(lx, x0 + inset), x1 - inset)  # a moved label stays inside the frame
        ly = min(max(ly, y0 + inset), y1 - inset)
        placed.append((lx, ly))
        if (lx, ly) != (g.x, g.y):
            ax.plot([g.x, lx], [g.y, ly], color=color, linewidth=0.6, zorder=29)
            ax.scatter([g.x], [g.y], s=6, c=color, edgecolors="white", linewidths=0.3, zorder=29)
        # A true circle of fixed size with the number centred on it (a text box shaped as a
        # circle came out uneven and blurred once the map was scaled onto the page).
        ax.scatter([lx], [ly], s=62, c=color, edgecolors="white", linewidths=0.9, zorder=30)
        ax.text(
            lx, ly, str(i), ha="center", va="center_baseline", fontsize=6.2, fontweight="bold",
            color="white", zorder=31,
        )  # fmt: skip


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


def change_pair(job_dir: Path, half_km: float = 1.6) -> str | None:
    """Before/after Sentinel-2 true colour, side by side, around the river station that widened
    most, with the measured channel width on each date (white before, cyan after) and the station
    ranks of the report's table. Drawn in EPSG:3857, the CRS of the web images; the scale bar is
    corrected for latitude. None when the job has no optical pair or no stations."""
    import matplotlib.image as mpimg
    from pyproj import Transformer

    m = json.loads((job_dir / "layers.json").read_text())
    img = m.get("imagery")
    path = job_dir / "web" / "change.geojson"
    if not img or img.get("kind") != "optical" or not path.exists():
        return None
    g = gpd.read_file(path).to_crs(3857)
    st = g[g["kind"] == "station"].sort_values("width_change_m", ascending=False).head(8)
    if not len(st):
        return None
    c = img["before"]["coordinates"]  # EPSG:4326 corners: TL, TR, BR, BL
    t = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y1 = t.transform(*c[0])
    x1, y0 = t.transform(*c[2])
    lat = t.transform(st.geometry.iloc[0].x, st.geometry.iloc[0].y, direction="INVERSE")[1]
    k = 1 / np.cos(np.radians(lat))
    cx, cy = st.geometry.iloc[0].x, st.geometry.iloc[0].y
    h = half_km * 1000 * k
    win = (cx - h, cx + h, cy - h, cy + h)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, side, color in ((axes[0], "before", "#ffffff"), (axes[1], "after", "#22d3ee")):
        ax.imshow(mpimg.imread(job_dir / img[side]["url"]), extent=(x0, x1, y0, y1))
        lines = g[g["kind"] == f"width_{side}"]
        lines.plot(ax=ax, color="#0f172a", linewidth=4.4, zorder=3)
        lines.plot(ax=ax, color=color, linewidth=2.4, zorder=4)
        for i, (_, r) in enumerate(st.iterrows(), start=1):
            x, y = r.geometry.x, r.geometry.y
            if win[0] < x < win[1] and win[2] < y < win[3]:
                _numbered(ax, x, y, i, "#0f172a", square=True)
                ax.annotate(
                    f"{int(r[f'width_{side}_m'])} m",
                    (x, y),
                    xytext=(9, -3),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=color if side == "after" else "white",
                    zorder=31,
                    annotation_clip=True,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "fc": "#0f172a",
                        "ec": "none",
                        "alpha": 0.8,
                    },
                )
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(img[side]["date"], fontsize=8, loc="left", color="#0f172a")
    _scalebar(axes[0], win, k)
    fig.tight_layout(w_pad=0.5)
    return _png(fig)


def satellite_pair(job_dir: Path) -> str | None:
    """Side-by-side thumbnail pair of pre- and post-event satellite imagery (optical or SAR) in base64."""
    layers_p = job_dir / "layers.json"
    if not layers_p.exists():
        return None
    try:
        layers = json.loads(layers_p.read_text())
    except Exception:
        return None
    img = layers.get("imagery")
    if not img or not (job_dir / img["before"]["url"]).exists():
        img = layers.get("imagery_radar")
    if not img or not (job_dir / img["before"]["url"]).exists():
        return None

    b_path = job_dir / img["before"]["url"]
    a_path = job_dir / img["after"]["url"]
    if not b_path.exists() or not a_path.exists():
        return None
    try:
        import matplotlib.image as mpimg

        b_img = mpimg.imread(b_path)
        a_img = mpimg.imread(a_path)
        # The web images are in Web Mercator, which stretches north-south by 1/cos(latitude);
        # drawing pixels with aspect cos(latitude) restores true ground proportions.
        lats = [c[1] for c in img["before"]["coordinates"]]
        k = float(np.cos(np.radians((max(lats) + min(lats)) / 2)))
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9), dpi=200)
        for ax, data, d_str in zip(
            axes, [b_img, a_img], [img["before"]["date"], img["after"]["date"]], strict=True
        ):
            ax.imshow(data, aspect=k)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#cbd5e1")
                spine.set_linewidth(0.8)
            ax.text(
                0.04,
                0.08,
                d_str,
                transform=ax.transAxes,
                fontsize=6.5,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.18", fc="#0f172a", ec="none", alpha=0.75),
            )
        fig.tight_layout(pad=0.1, w_pad=0.2)
        return _png(fig, dpi=220)
    except Exception:
        return None
