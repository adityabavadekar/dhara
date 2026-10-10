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
    f.add("bridges_total", br.get("count", 0), source=S)
    f.add("bridges_lost", br.get("status_likely_lost", 0), source=S)
    f.add("bridges_lost_high", br.get("confidence_high", 0), source=S)
    f.add("bridges_at_risk", br.get("status_at_risk", 0), source=S)

    st = summary.get("settlements", {})
    sdf = _read_gdf(job_dir / "settlements.parquet")
    cut = pd.DataFrame()
    if sdf is not None and len(sdf):
        cut = sdf[sdf["status"].isin(["cut_off", "hospital_cut_off"])].copy()
        cut = cut.sort_values(["buildings_500m", "p_cut_off"], ascending=False)
    f.add("settlements_total", int(sum(st.values())), source=S)
    f.add("settlements_cut_off", st.get("cut_off", 0), source=S)
    f.add("settlements_hospital_cut_off", st.get("hospital_cut_off", 0), source=S)
    f.add("settlements_detour", st.get("detour", 0), source=S)
    f.add("settlements_connected", st.get("connected", 0), source=S)
    f.add(
        "buildings_in_cut_off",
        int(cut["buildings_500m"].sum()) if len(cut) else 0,
        source="settlements.parquet",
    )
    f.tables["cut_off"] = [
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
        }
        for _, row in cut.head(10).iterrows()
    ]
    repairs = (
        json.loads((job_dir / "repairs.json").read_text())
        if (job_dir / "repairs.json").exists()
        else []
    )
    f.tables["repairs"] = [
        {
            "rank": fmt_num(i + 1),
            "what": ("Bridge" if r["is_bridge"] else "Road")
            + (f" {r['name']}" if r["name"] else ""),
            "highway": r["highway"],
            "settlements": fmt_num(r["settlements_reconnected"]),
            "buildings": fmt_num(r["buildings_reconnected"]),
            "osm_id": r["osm_id"],
        }
        for i, r in enumerate(repairs[:5])
    ]
    bridge_rep = [r for r in repairs if r["is_bridge"]]
    f.add("bridge_repairs_count", len(bridge_rep), source="repairs.json")
    f.tables["bridge_repairs"] = [
        {
            "rank": fmt_num(i + 1),
            "what": r["name"] or r["highway"],
            "settlements": fmt_num(r["settlements_reconnected"]),
            "buildings": fmt_num(r["buildings_reconnected"]),
            "osm_id": r["osm_id"],
        }
        for i, r in enumerate(bridge_rep[:3])
    ]
    return f
