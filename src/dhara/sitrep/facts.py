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

from dhara.damage.change import WIDENED_M
from dhara.osm_names import shown_name

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
    """Build display facts from job outputs; confidence rasters use the job's local UTM CRS."""
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

    coords = prov.get("inputs", {}).get("aoi", {}).get("geojson", {}).get("coordinates", [[]])[0]
    if coords:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        c_lon, c_lat = sum(lons) / len(lons), sum(lats) / len(lats)
        lat_txt = f"{abs(c_lat):.2f}°{'N' if c_lat >= 0 else 'S'}"
        lon_txt = f"{abs(c_lon):.2f}°{'E' if c_lon >= 0 else 'W'}"
        coord_txt = f"{lat_txt}, {lon_txt}"
        f.add("target_lat", f"{abs(c_lat):.2f}", source=P)
        f.add("target_lon", f"{abs(c_lon):.2f}", source=P)

        top_name = ""
        settl_p = job_dir / "settlements.parquet"
        if settl_p.exists():
            try:
                sdf_all = gpd.read_parquet(settl_p)
                in_aoi = sdf_all[sdf_all["in_aoi"]] if "in_aoi" in sdf_all.columns else sdf_all
                named = [shown_name(n) for n in in_aoi["name"].tolist() if n and str(n).strip()]
                if named:
                    top_name = str(named[0]).strip()
            except Exception:
                pass
        if top_name:
            f.add("target_location", f"{top_name} ({coord_txt})", source=P)
        else:
            f.add("target_location", coord_txt, source=P)
    else:
        f.add("target_location", "Unknown AOI", source=P)
    tracks = prov["data_sources"].get("sentinel1") or []
    f.add("s1_track_count", len(tracks), source=P)
    pre_dates = sorted({str(t["pre_start"])[:10] for t in tracks if t.get("pre_start")})
    post_dates = sorted({str(t["post_start"])[:10] for t in tracks if t.get("post_start")})
    f.add("s1_pre_dates", ", ".join(pre_dates) if pre_dates else "unknown", source=P)
    f.add("s1_post_dates", ", ".join(post_dates) if post_dates else "unknown", source=P)
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
    class_path, confidence_path = job_dir / "classes.tif", job_dir / "confidence.tif"
    if class_path.exists() and confidence_path.exists():
        import rasterio

        from dhara.fusion.fuse import FLOOD, TIER_HIGH, TIER_LOW, TIER_MOD

        resolution_m = float(prov["parameters"]["grid"]["res_m"])
        with rasterio.open(class_path) as src:
            classes = src.read(1)
        with rasterio.open(confidence_path) as src:
            confidence = src.read(1)
        if classes.shape == confidence.shape:
            pixel_km2 = resolution_m**2 / 1_000_000
            for name, tier in (("low", TIER_LOW), ("moderate", TIER_MOD), ("high", TIER_HIGH)):
                area = float(((classes == FLOOD) & (confidence == tier)).sum()) * pixel_km2
                f.add(
                    f"flood_area_{name}_km2",
                    area,
                    "km²",
                    decimals=2,
                    source="classes.tif + confidence.tif",
                )
        else:
            for name in ("low", "moderate", "high"):
                f.add(
                    f"flood_area_{name}_km2",
                    None,
                    "km²",
                    source="classes.tif + confidence.tif",
                    text="not available",
                )
    else:
        for name in ("low", "moderate", "high"):
            f.add(
                f"flood_area_{name}_km2",
                None,
                "km²",
                source="classes.tif + confidence.tif",
                text="not available",
            )
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
                    "name": str(shown_name(row.get("name")) or "Unnamed health facility"),
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
                "name": shown_name(row["name"]) or "(unnamed)",
                "name_ne": str(row.get("name_ne") or shown_name(row.get("name")) or ""),
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
            + (f" {r['name']}" if shown_name(r.get("name")) else ""),
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
            "what": shown_name(r.get("name")) or r.get("highway") or "Bridge",
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
                    "name": str(shown_name(row.get("name")) or row.get("highway") or "Bridge"),
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
                    "name": shown_name(row["name"]) or "(unnamed)",
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
                    "name": str(shown_name(row.get("name")) or "")
                    if str(shown_name(row.get("name")) or "") not in ("", "None")
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
    _change(f, job_dir)
    _verdict(f)
    return f


def _verdict(f: Facts) -> None:
    """Severity of the mapped impact, from corroborated evidence.

    SEVERE and HIGH need high-confidence evidence (probability ≥ 0.8 and two or more methods
    agree, see fusion), because single-pair radar change alone raises false alarms on hillsides
    and river banks: on images taken before a flood it can still mark dozens of communities as
    cut off. Counts built on low-confidence evidence can at most make a run MODERATE. The
    thresholds are design choices for ranking runs, not calibrated against outcomes.

    SEVERE    high-confidence flood/debris ≥ 1 km², or ≥ 250 buildings, or ≥ 5 bridges lost
    HIGH      high-confidence flood/debris ≥ 0.1 km², or ≥ 25 buildings, or ≥ 2 bridges lost
    MODERATE  any community cut off or without hospital access, ≥ 50 buildings, or ≥ 0.25 km²
              mapped, on any confidence
    LOW       otherwise
    """

    def _val(k: str) -> float:
        it = f.items.get(k)
        if it is None or not isinstance(it.value, int | float | str):
            return 0.0
        try:
            return float(it.value)
        except (ValueError, TypeError):
            return 0.0

    cut_off = int(_val("settlements_cut_off"))
    hosp_cut_off = int(_val("settlements_hospital_cut_off"))
    buildings_hit = int(_val("buildings_hit"))
    hazard_km2 = _val("area_debris_km2") + _val("area_flood_km2")
    tier_high = _val("area_tier_high_km2")
    bld_high = int(_val("buildings_hit_high"))
    br_high = int(_val("bridges_lost_high"))

    def plural(n: int, one: str, many: str) -> str:
        return f"{n:,} {one if n == 1 else many}"

    if tier_high >= 1.0 or bld_high >= 250 or br_high >= 5:
        level = "SEVERE"
    elif tier_high >= 0.1 or bld_high >= 25 or br_high >= 2:
        level = "HIGH"
    elif cut_off > 0 or hosp_cut_off > 0 or buildings_hit >= 50 or hazard_km2 >= 0.25:
        level = "MODERATE"
    else:
        level = "LOW"

    if level in ("SEVERE", "HIGH"):
        parts = []
        if bld_high:
            parts.append(plural(bld_high, "building", "buildings") + " in high-confidence damage")
        if br_high:
            parts.append(plural(br_high, "bridge", "bridges") + " likely lost (high confidence)")
        if cut_off:
            parts.append(plural(cut_off, "community", "communities") + " cut off")
        headline = " · ".join(parts) or f"{tier_high:.2f} km² of high-confidence flood or debris"
        parts_ne = []
        if bld_high:
            parts_ne.append(f"{bld_high:,} भवन उच्च विश्वासका साथ क्षतिग्रस्त")
        if br_high:
            parts_ne.append(f"{br_high:,} पुल सम्भवतः बगेका (उच्च विश्वास)")
        if cut_off:
            parts_ne.append(f"{cut_off:,} बस्ती सम्पर्कविहीन")
        headline_ne = " · ".join(parts_ne) or f"{tier_high:.2f} km² उच्च विश्वासको बाढी वा लेदो"
    elif level == "MODERATE":
        headline = (
            "Mapped change is mostly low-confidence: "
            + (plural(cut_off, "community", "communities") + " cut off, " if cut_off else "")
            + plural(buildings_hit, "building", "buildings")
            + " touched. Check before acting."
        )
        headline_ne = (
            "नक्साङ्कित परिवर्तन प्रायः कम विश्वासको छ: "
            + (f"{cut_off:,} बस्ती सम्पर्कविहीन, " if cut_off else "")
            + f"{buildings_hit:,} भवन प्रभावित। कदम चाल्नुअघि पुष्टि गर्नुहोस्।"
        )
    else:
        headline = "Little or no damage detected"
        headline_ne = "क्षति थोरै वा देखिएन"

    f.add("verdict_level", level, source="evidence classification")
    f.add("verdict_headline", headline, source="evidence classification")
    f.add("verdict_headline_ne", headline_ne, source="evidence classification")


def _signed(v: float, decimals: int = 0) -> str:
    return ("+" if v >= 0 else "−") + fmt_num(abs(v), decimals)


def _change(f: Facts, job_dir: Path) -> None:
    """Measured before/after change (dhara.damage.change, via layers.json and web/change.geojson).
    Nothing is added when the job has no cloud-free optical pair."""
    lj = job_dir / "layers.json"
    c = json.loads(lj.read_text()).get("change") if lj.exists() else None
    if not c:
        return
    src = "layers.json (damage.change)"
    f.add("change_before_day", c["before_day"], source=src)
    f.add("change_after_day", c["after_day"], source=src)
    f.add("change_clear_pct", c["clear_both_pct"], "%", decimals=1, source=src)
    for k in ("water", "channel"):
        b, a = float(c[f"{k}_before_km2"]), float(c[f"{k}_after_km2"])
        f.add(f"change_{k}_before_km2", b, "km²", decimals=2, source=src)
        f.add(f"change_{k}_after_km2", a, "km²", decimals=2, source=src)
        if b > 0:
            f.add(
                f"change_{k}_pct",
                (a - b) / b * 100,
                "%",
                source=src,
                text=_signed((a - b) / b * 100) + "%",
            )
    f.add("change_stations", c["stations"], source=src)
    if c.get("width_before_median_m") is None or c.get("width_after_median_m") is None:
        return
    wb, wa = float(c["width_before_median_m"]), float(c["width_after_median_m"])
    f.add("change_width_before_m", wb, "m", source=src)
    f.add("change_width_after_m", wa, "m", source=src)
    f.add("change_width_delta_m", wa - wb, "m", source=src, text=_signed(wa - wb))
    f.add("change_widened", c["widened_stations"], source=src)
    f.add("change_widened_threshold_m", c.get("widened_threshold_m", WIDENED_M), "m", source=src)
    f.add("change_shift_median_m", c["shift_median_m"], "m", source=src)
    f.add("change_shift_max_m", c["shift_max_m"], "m", source=src)
    path = job_dir / "web" / "change.geojson"
    if not path.exists():
        return
    g = gpd.read_file(path)
    st = g[g["kind"] == "station"].sort_values("width_change_m", ascending=False).head(8)
    rows = []
    for i, (_, r) in enumerate(st.iterrows(), start=1):
        lat, lon = r.geometry.y, r.geometry.x
        rows.append(
            {
                "rank": str(i),
                "before": fmt_num(r["width_before_m"]),
                "after": fmt_num(r["width_after_m"]),
                "delta": _signed(float(r["width_change_m"])),
                "shift": fmt_num(r["shift_m"]),
                "coords": f"{lat:.5f}, {lon:.5f}",
                "gmaps_url": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}",
            }
        )
    f.tables["change_stations"] = rows


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
                    "name": shown_name(p.get("name")) or "(unnamed)",
                    "km": fmt_num(p.get("distance_km", 0), 1),
                    "dz": fmt_num(p.get("delta_z_river_m", 0), 0),
                    "exposure": EXPOSURE.get(str(p.get("exposure_status")), "n/a"),
                    "arrival": fmt_num(arr, 0) if arr is not None else "n/a",
                    "gmaps_url": f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}",
                }
            )
        f.tables[f"flow_{i}"] = srows
    f.tables["flowpaths"] = rows
