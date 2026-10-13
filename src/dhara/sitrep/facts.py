"""Facts table for the situation report: every number in the sitrep comes from here.

Facts are read only from a finished job's outputs (summary.json, provenance.json, settlements,
bridges, repairs). Each fact carries its value, unit and the exact display text the template
prints. The verifier (verify.py) checks rendered text against these display strings.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

DEVANAGARI = str.maketrans("0123456789", "०१२३४५६७८९")
ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


@dataclass
class Fact:
    key: str
    value: object
    unit: str
    text: str
    source: str


def fmt_num(v: float | int, decimals: int = 0) -> str:
    if decimals == 0:
        return f"{round(float(v)):,}"
    return f"{float(v):,.{decimals}f}"


class Facts:
    def __init__(self) -> None:
        self.items: dict[str, Fact] = {}
        self.tables: dict[str, list[dict]] = {}
        # OSM ids in table order, for numbering map markers; not shown, so not part of the ledger.
        self.ids: dict[str, list[str]] = {}

    def add(self, key: str, value, unit: str = "", *, decimals: int = 0, source: str, text=None):
        if text is None:
            text = fmt_num(value, decimals) if isinstance(value, int | float) else str(value)
        self.items[key] = Fact(key, value, unit, text, source)
        return text

    def t(self, key: str) -> str:
        return self.items[key].text

    def to_json(self) -> dict:
        return {"facts": {k: asdict(v) for k, v in self.items.items()}, "tables": self.tables}


def _read_gdf(p: Path) -> gpd.GeoDataFrame | None:
    return gpd.read_parquet(p) if p.exists() else None


def build(job_dir: Path) -> Facts:
    summary = json.loads((job_dir / "summary.json").read_text())
    prov = json.loads((job_dir / "provenance.json").read_text())
    f = Facts()
    S, P = "summary.json", "provenance.json"

    f.add("job_id", prov["job_id"], source=P)
    f.add("generated_utc", prov["created_utc"], source=P)
    f.add("event", summary["event"], source=S)
    f.add("aoi_area_km2", prov["stages"]["validate_aoi"]["area_km2"], "km²", decimals=1, source=P)
    f.add("resolution_m", prov["parameters"]["grid"]["res_m"], "m", source=P)
    f.add("osm_snapshot", summary["osm_snapshot"], source=S)
    tracks = prov["data_sources"].get("sentinel1") or []
    f.add("s1_track_count", len(tracks), source=P)
    f.tables["s1"] = [
        {
            "track": f"R{t['relative_orbit']:03d} {t['orbit_state']}",
            "pre": str(t["pre_start"])[:10],
            "post": str(t["post_start"])[:10],
        }
        for t in tracks
    ]
    s2 = prov["data_sources"].get("sentinel2")
    f.add(
        "s2_pair",
        f"{s2['pre']['day'] if s2.get('pre') else 'none'} → {s2['post']['day']}"
        if s2
        else "none (cloud)",
        source=P,
    )
    f.add("members", ", ".join(summary.get("members", [])), source=S)

    a = summary["area_km2"]
    for k in ("flood", "debris", "not_observable", "permanent_water"):
        f.add(f"area_{k}_km2", a.get(k, 0.0), "km²", decimals=2, source=S)
    for k, v in summary.get("positive_km2_by_tier", {}).items():
        f.add(f"area_tier_{k}_km2", v, "km²", decimals=2, source=S)
    obs = 100.0 * (1 - a.get("not_observable", 0.0) / max(sum(a.values()), 1e-9))
    f.add("observable_pct", obs, "%", decimals=1, source=S)

    b = summary.get("buildings", {})
    f.add("buildings_total", b.get("count", 0), source=S)
    f.add("buildings_hit", b.get("status_hit", 0), source=S)
    f.add("buildings_unobserved", b.get("status_not_observable", 0), source=S)
    for tier in ("high", "moderate", "low"):
        f.add(f"buildings_hit_{tier}", b.get(f"confidence_{tier}", 0), source=S)
    r = summary.get("roads", {})
    f.add("road_ways_hit", r.get("status_hit", 0), source=S)
    n = summary.get("network", {})
    f.add("network_km", n.get("km_network", 0.0), "km", decimals=1, source=S)
    f.add("network_failed_km", n.get("km_failed", 0.0), "km", decimals=1, source=S)
    unobs = n.get("edges_unobserved", 0) / max(n.get("edges", 1), 1) * 100
    f.add("network_unobserved_pct", unobs, "%", decimals=0, source=S)
    f.add("mc_draws", n.get("mc_draws", 0), source=S)
    f.add("exposure_radius_m", n.get("exposure_radius_m", 0), "m", source=S)

    br = summary.get("bridges", {})
    bdf = _read_gdf(job_dir / "bridges.parquet")
    if (
        bdf is not None
        and len(bdf)
        and bdf.crs is not None
        and getattr(bdf.crs, "to_epsg", lambda: None)() != 4326
    ):
        bdf = bdf.to_crs(4326)
    f.add("bridges_total", br.get("count", 0), source=S)
    f.add("bridges_lost", br.get("status_likely_lost", 0), source=S)
    f.add("bridges_lost_high", br.get("confidence_high", 0), source=S)
    f.add("bridges_at_risk", br.get("status_at_risk", 0), source=S)

    st = summary.get("settlements", {})
    st_src = S
    sdf = _read_gdf(job_dir / "settlements.parquet")
    cut = pd.DataFrame()
    if (
        sdf is not None
        and len(sdf)
        and sdf.crs is not None
        and getattr(sdf.crs, "to_epsg", lambda: None)() != 4326
    ):
        sdf = sdf.to_crs(4326)
    if sdf is not None and len(sdf) and "status" in sdf:
        # Count from the settlements file itself so the ledger, the sitrep and the map layers
        # can never disagree (summary.json goes stale if the network stage is re-run).
        st = {str(k): int(v) for k, v in sdf["status"].value_counts().items()}
        st_src = "settlements.parquet"
    if sdf is not None and len(sdf):
        cut = sdf[sdf["status"].isin(["cut_off", "hospital_cut_off"])].copy()
        cut = cut.sort_values(["buildings_500m", "p_cut_off"], ascending=False)
    f.add("settlements_total", int(sum(st.values())), source=st_src)
    f.add("settlements_cut_off", st.get("cut_off", 0), source=st_src)
    f.add("settlements_hospital_cut_off", st.get("hospital_cut_off", 0), source=st_src)
    f.add("settlements_detour", st.get("detour", 0), source=st_src)
    f.add("settlements_connected", st.get("connected", 0), source=st_src)
    f.add(
        "buildings_in_cut_off",
        int(cut["buildings_500m"].sum()) if len(cut) else 0,
        source="settlements.parquet",
    )
    if sdf is not None and len(sdf) and "delay_hours_target" in sdf:
        delays = sdf[sdf["delay_hours_target"] > 0]["delay_hours_target"]
        max_d = float(delays.max()) if len(delays) else 0.0
        avg_d = float(delays.mean()) if len(delays) else 0.0
        f.add("max_detour_hours", max_d, "h", decimals=1, source="settlements.parquet")
        f.add("avg_detour_hours", avg_d, "h", decimals=1, source="settlements.parquet")
        f.add("settlements_with_detour", len(delays), source="settlements.parquet")
    else:
        f.add("max_detour_hours", 0.0, "h", decimals=1, source=S)
        f.add("avg_detour_hours", 0.0, "h", decimals=1, source=S)
        f.add("settlements_with_detour", 0, source=S)

    hdf = _read_gdf(job_dir / "hospitals.parquet")
    if (
        hdf is not None
        and len(hdf)
        and hdf.crs is not None
        and getattr(hdf.crs, "to_epsg", lambda: None)() != 4326
    ):
        hdf = hdf.to_crs(4326)
    n_hosp = len(hdf) if hdf is not None else 0
    n_hosp_iso = (
        int((hdf["is_isolated"] == True).sum())  # noqa: E712
        if hdf is not None and "is_isolated" in hdf
        else 0
    )
    n_hosp_acc = n_hosp - n_hosp_iso
    f.add("hospitals_total", n_hosp, source="hospitals.parquet" if hdf is not None else S)
    f.add("hospitals_accessible", n_hosp_acc, source="hospitals.parquet" if hdf is not None else S)
    f.add("hospitals_isolated", n_hosp_iso, source="hospitals.parquet" if hdf is not None else S)

    def _coords(row) -> tuple[str, str, str, str]:
        geom = getattr(row, "geometry", None)
        if geom is None:
            return "", "", "", ""
        pt = geom.centroid if hasattr(geom, "centroid") else geom
        lat_v, lon_v = float(pt.y), float(pt.x)
        lat_s, lon_s = f"{lat_v:.4f}", f"{lon_v:.4f}"
        coords_s = f"{lat_s}, {lon_s}"
        url_s = f"https://www.google.com/maps?q={lat_v:.6f},{lon_v:.6f}"
        return lat_s, lon_s, coords_s, url_s

    hosp_rows = []
    if hdf is not None:
        for _, row in hdf.head(25).iterrows():
            lat_s, lon_s, coords_s, url_s = _coords(row)
            hosp_rows.append(
                {
                    "name": str(row.get("name") or "Unnamed health facility"),
                    "amenity": str(row.get("amenity") or "clinic"),
                    "status": "Isolated" if row.get("is_isolated") else "Accessible",
                    "osm_id": str(row.get("osm_id") or "n/a"),
                    "coords": coords_s or "n/a",
                    "lat": lat_s,
                    "lon": lon_s,
                    "gmaps_url": url_s,
                }
            )
    f.tables["hospitals"] = hosp_rows

    cut_rows = []
    f.ids["cut_off"] = [str(o) for o in cut.head(40)["osm_id"]] if len(cut) else []
    for _, row in cut.head(40).iterrows():
        lat_s, lon_s, coords_s, url_s = _coords(row)
        cut_rows.append(
            {
                "name": row["name"] or "(unnamed)",
                "place": row["place"],
                "status": row["status"].replace("_", " "),
                "buildings": fmt_num(row["buildings_500m"]),
                "p_cut": fmt_num(100 * row["p_cut_off"], 0) + "%"
                if pd.notna(row["p_cut_off"])
                else "n/a",
                "km_pre": fmt_num(row["km_to_hospital_pre"], 1)
                if pd.notna(row["km_to_hospital_pre"])
                else "n/a",
                "hospital": str(row.get("hospital_name") or "unnamed"),
                "delay": f"+{fmt_num(row['delay_hours_target'], 1)}h"
                if pd.notna(row.get("delay_hours_target"))
                and float(row.get("delay_hours_target") or 0) > 0
                else "none",
                "coords": coords_s or "n/a",
                "lat": lat_s,
                "lon": lon_s,
                "gmaps_url": url_s,
            }
        )
    f.tables["cut_off"] = cut_rows

    def _osm_meta(osm_id_val: str | None) -> tuple[str, str]:
        if not osm_id_val or str(osm_id_val).lower() == "n/a":
            return "", ""
        s = str(osm_id_val).strip()
        if s.startswith("way/") or s.startswith("node/") or s.startswith("relation/"):
            return f"https://www.openstreetmap.org/{s}", f"OSM #{s.split('/', 1)[1]}"
        return f"https://www.openstreetmap.org/way/{s}", f"OSM #{s}"

    repairs = (
        json.loads((job_dir / "repairs.json").read_text())
        if (job_dir / "repairs.json").exists()
        else []
    )
    f.tables["repairs"] = [
        {
            "rank": fmt_num(i + 1),
            "what": ("Bridge" if r.get("is_bridge") else "Road")
            + (f" {r['name']}" if r.get("name") else ""),
            "highway": r.get("highway") or "roadway",
            "settlements": fmt_num(r.get("settlements_reconnected", 0)),
            "buildings": fmt_num(r.get("buildings_reconnected", 0)),
            "osm_id": r.get("osm_id") or "",
            "osm_url": _osm_meta(r.get("osm_id"))[0],
            "osm_label": _osm_meta(r.get("osm_id"))[1],
        }
        for i, r in enumerate(repairs[:10])
    ]
    f.ids["repairs"] = [str(r.get("osm_id")) for r in repairs[:10]]
    bridge_rep = [r for r in repairs if r.get("is_bridge")]
    f.add("bridge_repairs_count", len(bridge_rep), source="repairs.json")
    f.tables["bridge_repairs"] = [
        {
            "rank": fmt_num(i + 1),
            "what": r.get("name") or r.get("highway") or "Bridge",
            "settlements": fmt_num(r.get("settlements_reconnected", 0)),
            "buildings": fmt_num(r.get("buildings_reconnected", 0)),
            "osm_id": r.get("osm_id") or "",
            "osm_url": _osm_meta(r.get("osm_id"))[0],
            "osm_label": _osm_meta(r.get("osm_id"))[1],
        }
        for i, r in enumerate(bridge_rep[:3])
    ]

    bridge_rows = []
    if bdf is not None and len(bdf):
        rank = {"likely_lost": 0, "at_risk": 1, "not_observable": 2, "likely_standing": 3}
        bdf = bdf.assign(_r=bdf["status"].map(rank).fillna(9), _p=bdf["p_lost"].fillna(-1))
        bdf = bdf.sort_values(["_r", "_p"], ascending=[True, False])
        shown = bdf[bdf["status"].isin(["likely_lost", "at_risk"])].head(15)
        f.ids["bridges"] = [str(o) for o in shown["osm_id"]]
        for _, row in shown.iterrows():
            lat_s, lon_s, coords_s, url_s = _coords(row)
            osm_u, osm_l = _osm_meta(row.get("osm_id"))
            bridge_rows.append(
                {
                    "name": str(row.get("name") or row.get("highway") or "Bridge"),
                    "status": str(row.get("status") or "nominal").replace("_", " "),
                    "confidence": str(row.get("confidence") or "nominal"),
                    "p_lost": fmt_num(100 * row.get("p_lost"), 0) + "%"
                    if pd.notna(row.get("p_lost"))
                    else "n/a",
                    "drop_db": fmt_num(row.get("bright_drop_db"), 1)
                    if pd.notna(row.get("bright_drop_db"))
                    else "n/a",
                    "length_m": fmt_num(row.get("length_m"), 0)
                    if pd.notna(row.get("length_m"))
                    else "n/a",
                    "osm_id": str(row.get("osm_id") or "n/a"),
                    "osm_url": osm_u,
                    "osm_label": osm_l,
                    "coords": coords_s or "n/a",
                    "lat": lat_s,
                    "lon": lon_s,
                    "gmaps_url": url_s,
                }
            )
    f.tables["bridges"] = bridge_rows
    for i, r in enumerate(bridge_rows):
        r["rank"] = fmt_num(i + 1)
    for i, r in enumerate(f.tables["cut_off"]):
        r["rank"] = fmt_num(i + 1)

    # Settlements that still connect but by a longer route, longest extra time first.
    det = []
    if sdf is not None and len(sdf) and "delay_hours_target" in sdf:
        d = sdf[sdf["status"] == "detour"].sort_values("delay_hours_target", ascending=False)
        for _, row in d.head(25).iterrows():
            _, _, coords_s, url_s = _coords(row)
            det.append(
                {
                    "name": row["name"] or "(unnamed)",
                    "buildings": fmt_num(row["buildings_500m"]),
                    "h_pre": fmt_num(row.get("hours_to_target_pre"), 1)
                    if pd.notna(row.get("hours_to_target_pre"))
                    else "n/a",
                    "h_post": fmt_num(row.get("hours_to_target_post"), 1)
                    if pd.notna(row.get("hours_to_target_post"))
                    else "n/a",
                    "delay": fmt_num(row.get("delay_hours_target"), 1),
                    "coords": coords_s,
                    "gmaps_url": url_s,
                }
            )
    f.tables["detours"] = det

    # Schools, health posts and public buildings in the hazard zone.
    fac = []
    bld = _read_gdf(job_dir / "buildings.parquet")
    if bld is not None and len(bld) and "facility_type" in bld:
        hit = bld[
            (bld["status"] == "hit") & bld["facility_type"].isin(["school", "health", "community"])
        ]
        if hit.crs is not None and hit.crs.to_epsg() != 4326:
            hit = hit.to_crs(4326)
        for _, row in hit.sort_values("p_max", ascending=False).head(25).iterrows():
            _, _, coords_s, url_s = _coords(row)
            fac.append(
                {
                    "name": str(row.get("name") or "")
                    if str(row.get("name") or "") not in ("", "None")
                    else "(unnamed)",
                    "type": str(row["facility_type"]),
                    "hazard": str(row.get("hazard") or "n/a"),
                    "confidence": str(row.get("confidence") or "n/a"),
                    "coords": coords_s,
                    "gmaps_url": url_s,
                }
            )
    f.add("facilities_hit", len(fac), source="buildings.parquet")
    f.tables["facilities_hit"] = fac

    _flowpaths(f, job_dir, prov)
    return f


EXPOSURE = {
    "direct_path": "in the path",
    "fringe": "edge of path",
    "ridge_safe": "well above river",
}


def _flowpaths(f: Facts, job_dir: Path, prov: dict) -> None:
    """Flood paths traced from a start point inside this job's area (sibling fp_* job folders)."""
    from shapely.geometry import Point, shape

    geo = prov.get("inputs", {}).get("aoi", {}).get("geojson")
    aoi = shape(geo) if geo else None
    traces = []
    for d in sorted(job_dir.parent.glob("fp_*")):
        meta_p = d / "meta.json"
        if aoi is None or not meta_p.exists() or not (d / "settlements.geojson").exists():
            continue
        meta = json.loads(meta_p.read_text())
        start = meta.get("start_snapped") or meta.get("start_clicked")
        if not start or not aoi.contains(Point(start)):
            continue
        traces.append((d, meta))
    # Newest first; one trace per start point (repeat clicks re-trace the same path); at most 3.
    traces.sort(key=lambda t: (t[0] / "meta.json").stat().st_mtime, reverse=True)
    seen: set[tuple[float, float]] = set()
    unique = []
    for d, meta in traces:
        start = meta.get("start_snapped") or meta.get("start_clicked")
        key = (round(start[0], 3), round(start[1], 3))
        if key not in seen:
            seen.add(key)
            unique.append((d, meta))
    traces = unique[:3]
    f.ids["flowpaths"] = [d.name for d, _ in traces]
    f.add("flowpath_count", len(traces), source="fp_*/meta.json")
    rows = []
    for i, (d, meta) in enumerate(traces):
        start = meta.get("start_snapped") or meta.get("start_clicked")
        rows.append(
            {
                "rank": fmt_num(i + 1),
                "start": f"{start[1]:.4f}, {start[0]:.4f}",
                "start_url": f"https://www.google.com/maps?q={start[1]:.6f},{start[0]:.6f}",
                "length_km": fmt_num(meta.get("length_km", 0), 1),
                "drop_m": fmt_num(meta.get("drop_m", 0), 0),
                "settlements": fmt_num(meta.get("settlements", 0)),
                "stopped": str(meta.get("stopped_because", "n/a")),
            }
        )
        gj = json.loads((d / "settlements.geojson").read_text())
        srows = []
        for feat in gj.get("features", []):
            p = feat.get("properties") or {}
            lon, lat = (feat.get("geometry") or {}).get("coordinates", [0, 0])[:2]
            arr = p.get("arrival_time_min")
            srows.append(
                {
                    "order": fmt_num(int(p.get("order", 0))),
                    "name": p.get("name") or "(unnamed)",
                    "km": fmt_num(p.get("distance_km", 0), 1),
                    "dz": fmt_num(p.get("delta_z_river_m", 0), 0),
                    "exposure": EXPOSURE.get(str(p.get("exposure_status")), "n/a"),
                    "arrival": fmt_num(arr, 0) if arr is not None else "n/a",
                    "gmaps_url": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}",
                }
            )
        f.tables[f"flow_{i}"] = srows
    f.tables["flowpaths"] = rows
