"""Job orchestration: AOI + event → fused flood/debris map → damage → isolation → outputs + provenance.

CRS: rasters and GeoParquet in the AOI's local UTM zone (analysis); GeoJSON in EPSG:4326 (API/UI).
Caching: raw inputs (S1 passes, S2 scenes, DEM/terrain, OSM layers) are cached per AOI hash under
data/cache/<aoi_hash>/, so a re-run on the same AOI only recomputes the analysis.
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
from pydantic import BaseModel
from rasterio.features import geometry_mask
from rasterio.warp import Resampling
from scipy.ndimage import binary_dilation
from shapely import box

from dhara.adapters import ohsome
from dhara.adapters import sentinel2 as s2
from dhara.adapters.cdse import Grid, write_cog
from dhara.adapters.dem import fetch_dem
from dhara.adapters.sentinel1 import Radiometry, S1Pass, fetch_pass
from dhara.config import get_settings
from dhara.damage.overlay import bridge_check, is_bridge, sample_features, tag
from dhara.fusion import fuse as F
from dhara.fusion.detector import DetectorParams, refine
from dhara.geo import AOI, reproject
from dhara.jobs.exports import export_layers
from dhara.jobs.provenance import Provenance
from dhara.jobs.scene_selection import S1Pair, find_s1_pairs
from dhara.members import classical, debris, optical, zscore
from dhara.network import graph as netgraph
from dhara.osm_names import english_name
from dhara.sar import grd
from dhara.terrain import hydro
from dhara.timeutil import EventTime, iso_utc

RES_M = 10.0
log = logging.getLogger(__name__)

TERRAIN_BUFFER_M = 10_000.0
# Road network and targets are analysed this far beyond the AOI so routes to hospitals and towns
# outside the AOI exist; hazard is only known inside the AOI.
NETWORK_BUFFER_M = 15_000.0
# Height above nearest drainage (m) below which a pixel counts as valley floor.
VALLEY_HAND_M = 30.0
# Upstream area defining the major drainage network for the hydraulic-connectivity prior.
MAJOR_CHANNEL_KM2 = 10.0

Progress = Callable[[str, float], None]


S1Source = Literal["grd", "sentinelhub"]


class JobSpec(BaseModel):
    aoi: AOI
    # ISO 8601 date, or datetime with offset (stored as UTC). A time resolves same-day passes.
    event: EventTime
    relative_orbit: int | None = None  # force a single track
    use_s2: bool = True
    use_ml: bool = True
    # Multi-date radar baseline (members.zscore); needs a second pre-event pass on the track.
    # Off by default: in our accuracy tests it halved recall without separating the flood run from
    # a pre-flood control.
    use_zscore: bool = False
    # Fit the radar change thresholds on the valley floor (HAND ≤ VALLEY_HAND_M) instead of the
    # whole scene.
    valley_thresholds: bool = False
    # Latest Sentinel-2 day used as the "after" image, in days after the event. A short window keeps
    # later changes out; pre-flood control runs set it so that no image after the flood is used.
    s2_max_post_days: int = 20
    # Physical footprint rules after fusion (fusion.detector): terrain gate, corridor, valley-floor
    # fill and swept channel.
    use_detector: bool = True
    # Sentinel-1 processing: our own chain from the original GRD product (default) or Sentinel
    # Hub's server-side processing.
    s1_source: S1Source = "grd"


class JobResult(BaseModel):
    job_id: str
    job_dir: Path
    summary: dict


def make_grid(aoi: AOI) -> Grid:
    """10 m grid in the AOI's local UTM zone, snapped to whole metres."""
    crs = aoi.utm_crs()
    return Grid.snapped(crs, reproject(aoi.geometry, "EPSG:4326", crs).bounds, RES_M)


def aoi_mask(aoi: AOI, grid: Grid) -> np.ndarray:
    """True inside the AOI polygon, on `grid`."""
    return ~geometry_mask(
        [reproject(aoi.geometry, "EPSG:4326", grid.crs)], (grid.height, grid.width), grid.transform
    )


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read()


def _cached_pass(
    cache: Path,
    p: S1Pass,
    grid: Grid,
    radiometry: Radiometry,
    geom: bool,
    source: S1Source = "grd",
) -> np.ndarray:
    """One pass on `grid` (local UTM), cached. `grd`: our own processing chain from the original
    product (sar.grd); `sentinelhub`: server-side processing. If our chain fails on a pass, the
    pass is taken from Sentinel Hub instead and the sidecar records why (reported in provenance)."""
    suffix = "_grd" if source == "grd" else ""
    path = cache / f"s1_{p.key}_{radiometry}{'_geom' if geom else ''}{suffix}.tif"
    if _usable_pass(path):
        return _read(path)
    # One pass at a time per cache: the other radiometry of the same pass waits, then reads the
    # file this run writes instead of processing the pass again.
    with _pass_locks_guard:
        lock = _pass_locks.setdefault((str(cache), p.key, source), threading.Lock())
    with lock:
        if _usable_pass(path):
            return _read(path)
        return _process_and_cache(cache, p, grid, radiometry, geom, source, path)


def _usable_pass(path: Path) -> bool:
    """A cached pass is used only if this pipeline wrote it: a regular file with its sidecar.
    A link or a sidecar-less file may hold another processing chain's output, and comparing
    passes from two chains turns their calibration and geolocation differences into fake
    change. Such a file is removed so the pass is processed again."""
    if not (path.exists() or path.is_symlink()):
        return False
    if path.is_file() and not path.is_symlink() and path.with_suffix(".json").exists():
        return True
    log.warning("reprocessing %s: not written by this pipeline (link or no sidecar)", path.name)
    path.unlink(missing_ok=True)
    return False


_pass_locks: dict[tuple, threading.Lock] = {}
_pass_locks_guard = threading.Lock()


def _process_and_cache(cache, p, grid, radiometry, geom, source, path) -> np.ndarray:
    note: dict = {"pass": p.key, "radiometry": radiometry, "route": source}
    if source == "grd":
        try:
            dem, _ = fetch_dem(grid)
            both, meta = grd.process_pass_all(p.scene_ids, grid, dem)
            # One run gives both radiometries: cache all four variants so the other is a hit.
            for rad, a4 in both.items():
                for g4, a in ((True, a4), (False, a4[:2])):
                    pth = cache / f"s1_{p.key}_{rad}{'_geom' if g4 else ''}_grd.tif"
                    if pth != path and not _usable_pass(pth):
                        write_cog(str(pth), a, grid, nodata=float("nan"))
                        pth.with_suffix(".json").write_text(
                            json.dumps(note | meta | {"radiometry": rad}, default=str)
                        )
            arr = both[radiometry] if geom else both[radiometry][:2]
            note |= meta
        except Exception as e:  # keep the job alive; the fallback is recorded, not hidden
            arr, _ = fetch_pass(p, grid, radiometry, with_geometry_bands=geom)
            note |= {"route": "sentinelhub (fallback)", "error": f"{type(e).__name__}: {e}"[:300]}
    else:
        arr, _ = fetch_pass(p, grid, radiometry, with_geometry_bands=geom)
    write_cog(str(path), arr, grid, nodata=float("nan"))
    path.with_suffix(".json").write_text(json.dumps(note, default=str))
    return arr


def _cached_s2(cache: Path, scene: s2.S2Scene, grid: Grid) -> dict[str, np.ndarray]:
    path = cache / f"s2_{scene.day.isoformat()}.tif"
    names = [*s2.BANDS, "SCL"]
    if path.exists():
        a = _read(path)
    else:
        d = s2.fetch_scene(scene, grid)
        a = np.stack([d[n] for n in names])
        write_cog(str(path), a, grid, nodata=float("nan"))
    return {n: a[i] for i, n in enumerate(names)}


def cached_osm(
    cache: Path, layer: str, bbox: tuple[float, float, float, float], snap: date, key: str
) -> gpd.GeoDataFrame:
    """OSM layer at `snap` for a lon/lat bbox (EPSG:4326), cached under `key`."""
    path = cache / f"osm_{key}_{layer}_{snap.isoformat()}.parquet"
    if path.exists():
        return gpd.read_parquet(path)
    gdf = ohsome.fetch_layer(layer, bbox, snap)
    gdf.to_parquet(path)
    return gdf


def cached_terrain(cache: Path, grid: Grid) -> tuple[hydro.Terrain, dict]:
    tgrid = hydro.terrain_grid(grid, TERRAIN_BUFFER_M)
    tdir = cache / "terrain"
    tdir.mkdir(exist_ok=True)
    meta_path = tdir / "meta.json"
    t = hydro.load(tdir, tgrid)
    if t is not None and meta_path.exists():
        return t, json.loads(meta_path.read_text())
    dem, meta = fetch_dem(tgrid)
    t = hydro.compute(dem, tgrid)
    hydro.save(t, tdir)
    meta_path.write_text(json.dumps(meta))
    return t, meta


def _buffered(b: tuple[float, float, float, float], m: float) -> tuple[float, float, float, float]:
    return (b[0] - m, b[1] - m, b[2] + m, b[3] + m)


def _counts(gdf: gpd.GeoDataFrame) -> dict:
    out: dict = {"count": len(gdf)}
    out |= {f"status_{k}": int(v) for k, v in gdf["status"].value_counts().items()}
    if "confidence" in gdf:
        hit = gdf[gdf["status"].isin(["hit", "likely_lost"])]
        out |= {f"confidence_{k}": int(v) for k, v in hit["confidence"].value_counts().items()}
    return out


def select_tracks(pairs: list[S1Pair], forced: int | None) -> list[S1Pair]:
    """Best pair per orbit direction (ascending and descending fill each other's shadow/layover)."""
    if forced is not None:
        sel = [p for p in pairs if p.relative_orbit == forced]
        if not sel:
            avail = [p.relative_orbit for p in pairs]
            raise ValueError(f"No pair on relative orbit {forced}; available: {avail}")
        return sel
    out: dict[str, S1Pair] = {}
    for p in pairs:  # already ranked
        out.setdefault(p.orbit_state, p)
    return list(out.values())


def avg_logit(ps: list[np.ndarray]) -> np.ndarray:
    """Combine one member's maps from several tracks: mean logit over the observing tracks."""
    num = np.zeros(ps[0].shape)
    den = np.zeros(ps[0].shape)
    for p in ps:
        ok = np.isfinite(p)
        num[ok] += F.logit(p[ok])
        den[ok] += 1
    out = F.sigmoid(num / np.maximum(den, 1))
    out[den == 0] = np.nan
    return out


def run_job(
    spec: JobSpec, job_id: str | None = None, progress: Progress | None = None
) -> JobResult:
    s = get_settings()
    job_id = job_id or uuid.uuid4().hex[:12]
    job_dir = s.jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    cache = s.cache_dir / spec.aoi.hash
    cache.mkdir(parents=True, exist_ok=True)
    prov = Provenance(job_id, json.loads(spec.model_dump_json()))
    try:
        summary = _run(spec, cache, job_dir, prov, progress or (lambda *_: None))
    finally:
        prov.write(job_dir / "provenance.json")
    return JobResult(job_id=job_id, job_dir=job_dir, summary=summary)


def _scenes(spec: JobSpec, prov: Provenance) -> list[S1Pair]:
    with prov.stage("scene_selection") as st:
        pairs, notes = find_s1_pairs(spec.aoi, spec.event)
        tracks = select_tracks(pairs, spec.relative_orbit)
        st["candidates"] = [json.loads(p.model_dump_json()) for p in pairs]
        st["chosen_relative_orbits"] = [p.relative_orbit for p in tracks]
        prov.doc["notes"] += notes
        prov.doc["data_sources"]["sentinel1"] = [
            {
                "relative_orbit": p.relative_orbit,
                "orbit_state": p.orbit_state,
                "pre": p.pre.scene_ids,
                "pre2": p.pre2.scene_ids if p.pre2 else None,
                "post": p.post.scene_ids,
                "pre_start": p.pre.start,
                "post_start": p.post.start,
                "baseline_days": round(p.baseline_days, 2),
            }
            for p in tracks
        ]
        prov.doc["parameters"]["s1_route"] = (
            "Own processing chain from the original GRD product on CDSE S3 (sar.grd): thermal "
            "noise removal, calibration, Lee 5x5, Range-Doppler terrain correction and area-based "
            "terrain flattening on Copernicus DEM GLO-30"
            if spec.s1_source == "grd"
            else "CDSE Sentinel Hub Process API; orthorectified with COPERNICUS_30, LEE 5x5"
        ) + "; GAMMA0_TERRAIN for change members, SIGMA0_ELLIPSOID for the ML member"
    return tracks


def _s1_members(
    tracks,
    cache,
    grid,
    in_aoi,
    prov,
    progress,
    use_zscore: bool = False,
    fit_mask=None,
    source: S1Source = "grd",
):
    evidence: list[F.Evidence] = []
    perm_water = np.zeros(in_aoi.shape, bool)
    with prov.stage("s1_members") as st:
        water_maps, change_maps, zc_maps, zw_maps = [], [], [], []
        obs_any = np.zeros(in_aoi.shape, bool)
        for k, pair in enumerate(tracks):
            pre = _cached_pass(cache, pair.pre, grid, "GAMMA0_TERRAIN", False, source)
            post_all = _cached_pass(cache, pair.post, grid, "GAMMA0_TERRAIN", True, source)
            post, shadow = post_all[:2], post_all[3]
            obs = in_aoi & np.isfinite(shadow) & (np.nan_to_num(shadow) == 0)
            obs_any |= obs
            pre2 = (
                _cached_pass(cache, pair.pre2, grid, "GAMMA0_TERRAIN", False, source)
                if pair.pre2 is not None
                else None
            )
            cr = classical.run(pre, post, obs, fit_mask=fit_mask)
            dr = debris.run(pre, post, obs, tile=cr.meta["tile_px"], pre2=pre2, fit_mask=fit_mask)
            water_maps.append(cr.p_flood)
            change_maps.append(dr.p_change)
            perm_water |= cr.permanent_water
            zr = None
            if use_zscore and pre2 is not None:
                zr = zscore.run(
                    [pre2[:2], pre[:2]], post, obs, water_db=cr.meta["water_threshold_db"]
                )
                zc_maps.append(zr.p_change)
                zw_maps.append(np.where(cr.permanent_water, 0.0, zr.p_water).astype(np.float32))
            st[f"R{pair.relative_orbit:03d}"] = {
                "observable_fraction": round(float(obs[in_aoi].mean()), 4),
                "classical": cr.meta,
                "debris": dr.meta,
                "zscore": zr.meta if zr is not None else "skipped: no second pre-event pass",
            }
            progress("sentinel1", 0.2 + 0.25 * (k + 1) / len(tracks))
        evidence.append(F.Evidence("s1_water_logratio", "flood", avg_logit(water_maps), 1.0, "s1"))
        evidence.append(
            F.Evidence("s1_change_bidirectional", "debris", avg_logit(change_maps), 1.0, "s1")
        )
        if zc_maps:
            evidence.append(F.Evidence("s1_zscore_change", "debris", avg_logit(zc_maps), 1.0, "s1"))
            evidence.append(F.Evidence("s1_zscore_water", "flood", avg_logit(zw_maps), 1.0, "s1"))
        st["observable_fraction_any_track"] = round(float(obs_any[in_aoi].mean()), 4)
    return evidence, perm_water


def _ml_member(tracks, cache, grid, in_aoi, prov, source: S1Source = "grd") -> F.Evidence | None:
    from dhara.members import ml_seg

    with prov.stage("ml_member") as st:
        model = ml_seg.load_model()
        if model is None:
            st["skipped"] = f"no trained model at {ml_seg.MODEL_DIR}"
            return None
        maps = []
        for pair in tracks:
            post = _cached_pass(cache, pair.post, grid, "SIGMA0_ELLIPSOID", False, source)
            pre = _cached_pass(cache, pair.pre, grid, "SIGMA0_ELLIPSOID", False, source)
            pre2 = (
                _cached_pass(cache, pair.pre2, grid, "SIGMA0_ELLIPSOID", False, source)
                if pair.pre2
                else pre
            )
            geom = _cached_pass(cache, pair.post, grid, "GAMMA0_TERRAIN", True, source)
            obs = in_aoi & (np.nan_to_num(geom[3], nan=1) == 0)
            p = ml_seg.predict(model, post, pre, pre2).p_flood
            maps.append(np.where(obs, p, np.nan).astype(np.float32))
        st["model"] = model.card_summary
        return F.Evidence("ml_unet_kurosiwo", "flood", avg_logit(maps), 1.5, "s1")


# Fewest 60 m pixels of reachable ground to judge cloud on (about 1.8 km²); below this the whole
# AOI is used.
MIN_CLOUD_REACH_PX = 500


def s2_cloud_masks(spec: JobSpec, cache: Path, grid: Grid) -> list[tuple[s2.S2Scene, np.ndarray]]:
    """Candidate S2 days around the event with their 60 m cloud masks (on the AOI's snapped
    60 m grid in local UTM). Masks are cached per AOI, so re-runs spend no quota."""
    cands = s2.candidate_days(spec.aoi, spec.event)
    ev_day = spec.event.date() if isinstance(spec.event, datetime) else spec.event
    cands = [c for c in cands if (c.day - ev_day).days <= spec.s2_max_post_days]
    g60 = Grid.snapped(grid.crs, grid.bounds, 60.0)
    d = cache / "s2_cloud60"
    d.mkdir(parents=True, exist_ok=True)

    def mask(c: s2.S2Scene) -> np.ndarray:
        name = hashlib.sha256(f"{c.day}|{','.join(sorted(c.scene_ids))}".encode()).hexdigest()[:16]
        p = d / f"{c.day.isoformat()}_{name}.npy"
        if p.exists():
            return np.load(p)
        m = s2.cloud_mask(c, g60)
        np.save(p, m)
        return m

    with ThreadPoolExecutor(max_workers=6) as pool:
        masks = list(pool.map(mask, cands))
    return list(zip(cands, masks, strict=True))


def select_s2(
    spec: JobSpec,
    cache: Path,
    grid: Grid,
    cands: list[tuple[s2.S2Scene, np.ndarray]],
    reach: np.ndarray | None,
    st: dict,
) -> s2.S2Pair:
    """Choose the clearest S2 pair and download it into the cache. grid: the job's 10 m UTM grid;
    reach: ground a flood or debris can reach, on that grid.

    Scenes are ranked by cloud over the whole AOI, as before, but whether a scene is usable at
    all is judged on reachable ground (see sentinel2.choose_pair): over a large mountain area,
    cloud on the ridges would otherwise reject the only image whose valleys are partly clear.
    """
    g60 = Grid.snapped(grid.crs, grid.bounds, 60.0)
    in60 = aoi_mask(spec.aoi, g60)
    region = None
    if reach is not None:
        r60 = hydro.resample_to(reach.astype(np.float32), grid, g60) >= 0.5
        if int((in60 & r60).sum()) >= MIN_CLOUD_REACH_PX:
            region = in60 & r60
    st["s2_usable_judged_on"] = "reachable ground" if region is not None else "whole AOI"
    for c, m in cands:
        c.aoi_cloud_fraction = round(float(m[in60].mean()), 4) if in60.any() else 1.0
        if region is not None:
            c.reach_cloud_fraction = round(float(m[region].mean()), 4)
    pair = s2.choose_pair([c for c, _ in cands], spec.event)
    scenes = [sc for sc in (pair.pre, pair.post) if sc is not None]
    with ThreadPoolExecutor(max_workers=2) as pool:  # before and after are independent
        list(pool.map(lambda sc: _cached_s2(cache, sc, grid), scenes))
    return pair


def _s2_member(pair: s2.S2Pair, cache, grid, in_aoi, prov):
    with prov.stage("s2_member") as st:
        st["candidates"] = [c.model_dump(mode="json") for c in pair.candidates]
        if pair.post is None:
            st["skipped"] = "no post-event Sentinel-2 scene with ≤60% AOI cloud within ±20 days"
            return [], None, None
        post_s = _cached_s2(cache, pair.post, grid)
        pre_s = _cached_s2(cache, pair.pre, grid) if pair.pre else None
        orr = optical.run(pre_s, post_s)
        ev = [F.Evidence("s2_mndwi", "flood", np.where(in_aoi, orr.p_flood, np.nan), 1.0, "s2")]
        if pair.pre is not None:
            ev.append(
                F.Evidence(
                    "s2_ndvi_bsi", "debris", np.where(in_aoi, orr.p_debris, np.nan), 1.0, "s2"
                )
            )
        st.update(orr.meta)
        used = {
            "pre": pair.pre.model_dump(mode="json") if pair.pre else None,
            "post": pair.post.model_dump(mode="json"),
        }
        return ev, orr.water_pre, used


def _fetch_inputs(spec, cache, grid, tracks, net_bbox, snap, st) -> dict:
    """Download every remote input concurrently into the cache (S1, DEM/terrain, OSM, S2).

    S2 is optional: its failure is recorded, not raised. Everything else must succeed.
    """
    from dhara.members import ml_seg

    s1_jobs: list[tuple[S1Pass, Radiometry, bool]] = []
    for p in tracks:
        s1_jobs += [(p.pre, "GAMMA0_TERRAIN", False), (p.post, "GAMMA0_TERRAIN", True)]
        if spec.use_ml and ml_seg.model_available():
            s1_jobs += [(p.post, "SIGMA0_ELLIPSOID", False), (p.pre, "SIGMA0_ELLIPSOID", False)]
            if p.pre2 is not None:
                s1_jobs.append((p.pre2, "SIGMA0_ELLIPSOID", False))
    layers = ("roads", "buildings", "places", "health")
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    def timed(name, fn, *a):
        t = time.perf_counter()
        r = fn(*a)
        timings[name] = round(time.perf_counter() - t, 2)
        return r

    with ThreadPoolExecutor(max_workers=8) as pool:
        f_s1 = [
            pool.submit(
                timed, f"s1:{p.key}:{r}", _cached_pass, cache, p, grid, r, g, spec.s1_source
            )
            for p, r, g in s1_jobs
        ]
        f_ter = pool.submit(timed, "terrain", cached_terrain, cache, grid)
        f_osm = {
            ly: pool.submit(timed, f"osm:{ly}", cached_osm, cache, ly, net_bbox, snap, "net")
            for ly in layers
        }
        f_s2 = (
            pool.submit(timed, "s2_clouds", s2_cloud_masks, spec, cache, grid)
            if spec.use_s2
            else None
        )
        for f in f_s1:
            f.result()
        out: dict = {"terrain": f_ter.result(), "osm": {ly: f.result() for ly, f in f_osm.items()}}
        if f_s2 is not None:
            try:
                out["s2"] = f_s2.result()
            except Exception as e:  # optical is optional
                out["s2"] = f"{type(e).__name__}: {e}"
    st["timings_s"] = timings
    st["wall_s"] = round(time.perf_counter() - t0, 2)
    return out


def _run(spec: JobSpec, cache: Path, job_dir: Path, prov: Provenance, progress: Progress) -> dict:
    aoi = spec.aoi
    with prov.stage("validate_aoi") as st:
        aoi.check_size()
        st["area_km2"] = round(aoi.area_km2, 2)
    progress("scene_selection", 0.02)
    tracks = _scenes(spec, prov)

    grid = make_grid(aoi)
    in_aoi = aoi_mask(aoi, grid)
    prov.doc["parameters"]["grid"] = {
        "crs": grid.crs.to_string(),
        "res_m": grid.res,
        "bounds": grid.bounds,
        "shape": [grid.height, grid.width],
    }
    progress("fetch_inputs", 0.08)
    net_bounds_utm = _buffered(grid.bounds, NETWORK_BUFFER_M)
    net_bbox = reproject(box(*net_bounds_utm), grid.crs, "EPSG:4326").bounds
    snap = ohsome.snapshot_date(spec.event)
    with prov.stage("fetch_inputs") as st:
        fetched = _fetch_inputs(spec, cache, grid, tracks, net_bbox, snap, st)
    # How each pass was processed (route, geoid offset, terrain fit, or why it fell back).
    keys = {p.key for pair in tracks for p in (pair.pre, pair.post, pair.pre2) if p is not None}
    prov.doc["data_sources"]["s1_processing"] = [
        json.loads(f.read_text())
        for f in sorted(cache.glob("s1_*.json"))
        if any(k in f.name for k in keys) and (("_grd" in f.name) == (spec.s1_source == "grd"))
    ]

    progress("terrain", 0.15)
    with prov.stage("terrain") as st:
        terr, dem_meta = fetched["terrain"]
        hand = hydro.resample_to(terr.hand, terr.grid, grid)
        slope = hydro.resample_to(terr.slope_deg, terr.grid, grid)
        acc_km2 = terr.acc * (terr.grid.res**2) / 1e6
        major = hydro.resample_to(
            (acc_km2 >= MAJOR_CHANNEL_KM2).astype(np.float32), terr.grid, grid, Resampling.max
        )
        major = binary_dilation(np.nan_to_num(major) > 0, iterations=3)
        st["major_channel_km2"] = MAJOR_CHANNEL_KM2
        prov.doc["data_sources"]["dem"] = dem_meta
        st["channel_area_km2"] = hydro.CHANNEL_AREA_KM2
        st["terrain_buffer_m"] = TERRAIN_BUFFER_M
    write_cog(str(job_dir / "hand.tif"), hand, grid, nodata=float("nan"))
    dem_10m = hydro.resample_to(terr.dem, terr.grid, grid)
    write_cog(str(job_dir / "dem.tif"), dem_10m, grid, nodata=float("nan"))
    write_cog(str(job_dir / "slope.tif"), slope, grid, nodata=float("nan"))

    progress("sentinel1", 0.2)
    valley = None
    if spec.valley_thresholds:
        valley = np.nan_to_num(hand, nan=np.inf) <= VALLEY_HAND_M
    evidence, perm_water = _s1_members(
        tracks, cache, grid, in_aoi, prov, progress, spec.use_zscore, valley, spec.s1_source
    )
    if spec.use_ml:
        ml = _ml_member(tracks, cache, grid, in_aoi, prov, spec.s1_source)
        if ml is not None:
            evidence.append(ml)

    progress("sentinel2", 0.5)
    s2_used = None
    if spec.use_s2 and isinstance(fetched.get("s2"), list):
        try:
            with prov.stage("s2_selection") as st:
                reach = np.nan_to_num(hand, nan=np.inf) <= DetectorParams().debris_max_hand_m
                fetched["s2"] = select_s2(spec, cache, grid, fetched["s2"], reach, st)
        except Exception as e:  # optical is optional
            fetched["s2"] = f"{type(e).__name__}: {e}"
    if spec.use_s2 and isinstance(pair := fetched.get("s2"), s2.S2Pair):
        try:
            ev2, water_pre, s2_used = _s2_member(pair, cache, grid, in_aoi, prov)
            evidence += ev2
            if water_pre is not None:
                perm_water |= water_pre
        except Exception as e:  # optical is optional: record honestly and continue with SAR
            prov.doc["notes"].append(f"Sentinel-2 member skipped: {type(e).__name__}: {e}")
    elif spec.use_s2:
        prov.doc["notes"].append(f"Sentinel-2 member skipped: {fetched.get('s2')}")
    prov.doc["data_sources"]["sentinel2"] = s2_used

    progress("fusion", 0.6)
    with prov.stage("fusion") as st:
        fr = F.fuse(evidence, hand, slope, in_aoi, perm_water, major_channels=major)
        if spec.use_detector:
            prm = DetectorParams()
            fr.classes, fr.confidence, det = refine(
                fr.classes, fr.confidence, fr.p_flood, fr.p_debris, hand, slope, major, prm
            )
            st["detector"] = {"params": prm.model_dump(), "changed_px": det}
        px_km2 = grid.res**2 / 1e6
        st["area_km2"] = {
            n: round(float((fr.classes == c).sum()) * px_km2, 3) for c, n in F.CLASS_NAMES.items()
        }
        st["positive_km2_by_tier"] = {
            F.TIER_NAMES[t]: round(float((fr.confidence == t).sum()) * px_km2, 3) for t in (1, 2, 3)
        }
        st.update(fr.meta)
        nan = float("nan")
        write_cog(str(job_dir / "classes.tif"), fr.classes, grid, nodata=F.OUTSIDE)
        write_cog(str(job_dir / "confidence.tif"), fr.confidence, grid, nodata=None)
        for name, arr in (("p_flood", fr.p_flood), ("p_debris", fr.p_debris)):
            write_cog(
                str(job_dir / f"{name}.tif"),
                np.where(in_aoi, arr, nan).astype(np.float32),
                grid,
                nodata=nan,
            )
        write_cog(str(job_dir / "n_observing.tif"), fr.n_observing, grid, nodata=None)
        write_cog(str(job_dir / "perm_water.tif"), perm_water.astype(np.uint8), grid, nodata=None)
        for e in evidence:
            write_cog(
                str(job_dir / f"member_{e.name}.tif"),
                np.where(in_aoi, e.p, nan).astype(np.float32),
                grid,
                nodata=nan,
            )

    summary: dict = {
        "event": iso_utc(spec.event)
        if isinstance(spec.event, datetime)
        else spec.event.isoformat(),
        "area_km2": prov.doc["stages"]["fusion"]["area_km2"],
        "positive_km2_by_tier": prov.doc["stages"]["fusion"]["positive_km2_by_tier"],
        "tracks": [f"R{p.relative_orbit:03d} {p.orbit_state}" for p in tracks],
        "members": [e.name for e in evidence],
    }

    progress("damage", 0.7)
    snap_iso = f"{snap.isoformat()}T00:00:00Z"
    prov.doc["data_sources"]["osm"] = {
        "snapshot": snap_iso,
        "route": "ohsome API",
        "network_bbox_wgs84": [round(v, 5) for v in net_bbox],
    }
    summary["osm_snapshot"] = snap_iso
    osm = fetched["osm"]
    p_any = np.fmax(fr.p_flood, fr.p_debris).astype(np.float32)
    p_any[~in_aoi] = np.nan
    tier = np.where(in_aoi, fr.confidence, 0)

    with prov.stage("damage_overlay") as st:
        aoi_box = box(*aoi.bounds)
        for layer in ("buildings", "roads"):
            gdf = osm[layer]
            gdf = gdf[gdf.intersects(aoi_box)] if len(gdf) else gdf
            if len(gdf) == 0:
                st[layer] = {"count": 0}
                summary[layer] = st[layer]
                continue
            out = sample_features(
                gdf, p_any, grid, p_flood=fr.p_flood, p_debris=fr.p_debris, tier=tier
            )
            out = out[out["status"] != "outside"]
            out.to_parquet(job_dir / f"{layer}.parquet")
            st[layer] = _counts(out)
            summary[layer] = st[layer]

    with prov.stage("bridges") as st:
        roads = osm["roads"]
        bridges = roads[[is_bridge(t) for t in roads["tags"]]] if len(roads) else roads
        bridges = bridges[bridges.intersects(aoi_box)] if len(bridges) else bridges
        vv_pairs = []
        for pair in tracks:
            pre = _cached_pass(cache, pair.pre, grid, "GAMMA0_TERRAIN", False, spec.s1_source)
            post = _cached_pass(cache, pair.post, grid, "GAMMA0_TERRAIN", True, spec.s1_source)
            vv_pairs.append((pre[0], post[0]))
        if len(bridges):
            bres = bridge_check(bridges, grid, p_any, vv_pairs)
            bres["name"] = [
                english_name(
                    json.loads(t) if isinstance(t, str) else (t or {}),
                    "bridge:name:en",
                    "bridge:name",
                )
                or (f"{tag(t, 'ref')} bridge" if tag(t, "ref") else "")
                for t in bres["tags"]
            ]
            bres["ref"] = [tag(t, "ref") or "" for t in bres["tags"]]
            bres["bridge_type"] = [tag(t, "bridge") or "" for t in bres["tags"]]
            bres["highway"] = [tag(t, "highway") or "" for t in bres["tags"]]
            bres.to_parquet(job_dir / "bridges.parquet")
            st.update(_counts(bres))
        else:
            bres = gpd.GeoDataFrame(bridges.assign(p_lost=np.zeros(0)))
            st["count"] = 0
        summary["bridges"] = {k: v for k, v in st.items() if k not in ("status", "seconds")}

    progress("network", 0.8)
    with prov.stage("isolation") as st:
        net = netgraph.build(osm["roads"], grid.crs)
        netgraph.edge_hazard(net, p_any, grid, in_aoi, tier=tier)
        netgraph.apply_bridges(net, bres)
        iso = netgraph.analyse(
            net, osm["places"], osm["health"], osm["buildings"], net_bounds_utm, grid.crs
        )
        iso.settlements["in_aoi"] = iso.settlements.within(
            reproject(aoi.geometry, "EPSG:4326", grid.crs)
        )
        iso.settlements.to_parquet(job_dir / "settlements.parquet")
        iso.failed_edges.to_parquet(job_dir / "failed_edges.parquet")
        iso.hospitals.to_parquet(job_dir / "hospitals.parquet")
        (job_dir / "repairs.json").write_text(json.dumps(iso.repairs, indent=1))
        st.update(iso.meta)
        st["settlements"] = iso.settlements["status"].value_counts().to_dict()
        summary["network"] = iso.meta
        summary["settlements"] = st["settlements"]
        summary["top_repairs"] = iso.repairs[:5]

    progress("outputs", 0.92)
    with prov.stage("exports"):
        export_layers(
            job_dir,
            grid,
            in_aoi,
            cache,
            prov.doc.get("data_sources", {}).get("sentinel1"),
            prov.doc.get("data_sources", {}).get("sentinel2"),
        )
    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    prov.write(job_dir / "provenance.json")  # the sitrep reads both
    progress("sitrep", 0.96)
    with prov.stage("sitrep") as st:
        from dhara.sitrep.render import write as write_sitrep

        st.update(write_sitrep(job_dir))
    progress("done", 1.0)

    (job_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
