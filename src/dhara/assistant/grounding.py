"""Extract and synthesize multimodal disaster context for LLM grounding.

Combines SAR radar change detection metrics, optical vegetation loss,
Copernicus DEM topography, OSM infrastructure network reachability,
and ranked bridge/road repair priorities.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from dhara.osm_names import shown_name


def _int(v: Any) -> int:
    """GeoJSON exports store numbers as strings; "None" and blanks count as 0."""
    try:
        return int(float(str(v)))
    except ValueError:
        return 0


def _float(v: Any) -> float:
    try:
        return float(str(v))
    except ValueError:
        return 0.0


def _approximate_aoi_center(aoi: Any) -> tuple[float, float] | None:
    """Return the bounding-box center of a GeoJSON AOI in EPSG:4326 (longitude, latitude)."""
    if not isinstance(aoi, dict):
        return None
    geometry = aoi.get("geojson", aoi)
    if not isinstance(geometry, dict):
        return None
    coordinates = geometry.get("coordinates")
    positions: list[tuple[float, float]] = []

    def collect(value: Any) -> None:
        if not isinstance(value, list):
            return
        if len(value) >= 2 and all(isinstance(part, (int, float)) for part in value[:2]):
            positions.append((float(value[0]), float(value[1])))
            return
        for child in value:
            collect(child)

    collect(coordinates)
    if not positions:
        return None
    longitudes = [position[0] for position in positions]
    latitudes = [position[1] for position in positions]
    return (
        (min(longitudes) + max(longitudes)) / 2,
        (min(latitudes) + max(latitudes)) / 2,
    )


def load_job_context(job_dir: Path) -> dict[str, Any]:
    """Load job evidence; GeoJSON geometries and AOI coordinates use EPSG:4326."""
    context: dict[str, Any] = {
        "job_id": job_dir.name,
        "kpis": {},
        "summary": {},
        "cut_off_settlements": [],
        "damaged_bridges": [],
        "health_facilities": [],
        "repair_priorities": [],
        "location": {},
    }

    facts_path = job_dir / "facts.json"
    if facts_path.exists():
        with contextlib.suppress(Exception):
            facts_data = json.loads(facts_path.read_text())
            items = facts_data.get("facts", {})
            # facts.json stores {"facts": {key: fact}}; accept a plain list as well.
            for item in items.values() if isinstance(items, dict) else items:
                context["kpis"][item["key"]] = {
                    "text": item.get("text", ""),
                    "val": item.get("val"),
                    "num": item.get("num"),
                }

    # The report's facts (cached by the API, rebuilt when an input or the facts code changes)
    # supersede the ones written when the job ran, so the assistant, dashboard and report agree.
    report_facts = job_dir / "sitrep_facts.json"
    if report_facts.exists():
        with contextlib.suppress(Exception):
            items = json.loads(report_facts.read_text())["payload"]["items"]
            for key, item in items.items():
                context["kpis"][key] = {"text": item.get("text", ""), "val": item.get("value")}

    summary_path = job_dir / "summary.json"
    if summary_path.exists():
        with contextlib.suppress(Exception):
            context["summary"] = json.loads(summary_path.read_text())

    provenance_path = job_dir / "provenance.json"
    if provenance_path.exists():
        with contextlib.suppress(Exception):
            provenance = json.loads(provenance_path.read_text())
            inputs = provenance.get("inputs", {})
            aoi = inputs.get("aoi", {})
            context["location"] = {
                "name": aoi.get("name") or inputs.get("location_name"),
                "center": _approximate_aoi_center(aoi),
            }

    settlements_path = job_dir / "web" / "settlements.geojson"
    if settlements_path.exists():
        with contextlib.suppress(Exception):
            gj = json.loads(settlements_path.read_text())
            for feat in gj.get("features", []):
                props = feat.get("properties", {})
                status = str(props.get("status", ""))
                if status in ("cut_off", "hospital_cut_off", "detour"):
                    context["cut_off_settlements"].append(
                        {
                            "name": shown_name(props.get("name")) or "(unnamed settlement)",
                            "status": status,
                            "p_cut_off": props.get("p_cut_off"),
                            "buildings": _int(props.get("buildings_500m")),
                            "km_to_hospital": props.get("km_to_hospital_before"),
                            "coordinates": feat.get("geometry", {}).get("coordinates"),
                        }
                    )
            # sort by largest exposure
            context["cut_off_settlements"].sort(key=lambda s: s.get("buildings") or 0, reverse=True)

    bridges_path = job_dir / "web" / "bridges.geojson"
    if bridges_path.exists():
        with contextlib.suppress(Exception):
            gj = json.loads(bridges_path.read_text())
            for feat in gj.get("features", []):
                props = feat.get("properties", {})
                status = str(props.get("status", ""))
                if status in ("likely_lost", "at_risk"):
                    context["damaged_bridges"].append(
                        {
                            "name": (
                                str(shown_name(props.get("name")))
                                if shown_name(props.get("name")) not in (None, "", "None")
                                else ""
                            )
                            or f"Unnamed {props.get('highway') or 'road'} bridge",
                            "status": status,
                            "p_lost": props.get("p_lost"),
                            "bright_drop_db": props.get("bright_drop_db"),
                            "radar_pre_db": props.get("radar_pre_db"),
                            "radar_post_db": props.get("radar_post_db"),
                            "coordinates": feat.get("geometry", {}).get("coordinates", [None])[0],
                        }
                    )
            context["damaged_bridges"].sort(key=lambda b: _float(b.get("p_lost")), reverse=True)

    hospitals_path = job_dir / "web" / "hospitals.geojson"
    if hospitals_path.exists():
        with contextlib.suppress(Exception):
            gj = json.loads(hospitals_path.read_text())
            for feat in gj.get("features", []):
                props = feat.get("properties", {})
                raw_isolated = props.get("is_isolated")
                is_isolated = raw_isolated is True or str(raw_isolated).lower() == "true"
                is_isolated = is_isolated or str(props.get("status", "")).lower() == "isolated"
                context["health_facilities"].append(
                    {
                        "name": shown_name(props.get("name")) or "Unnamed health facility",
                        "kind": props.get("amenity") or props.get("kind") or "health facility",
                        "is_isolated": is_isolated,
                    }
                )

    repairs_path = job_dir / "repairs.json"
    if repairs_path.exists():
        with contextlib.suppress(Exception):
            repairs_data = json.loads(repairs_path.read_text())
            context["repair_priorities"] = repairs_data[:10]

    return context


def build_system_prompt(ctx: dict[str, Any], lang: str = "en") -> str:
    """Build a system prompt grounding an AI API in verified disaster facts."""
    k = ctx.get("kpis", {})

    def get_kpi(key: str, default: str = "N/A") -> str:
        return str(k.get(key, {}).get("text", default))

    kpis_text = f"""
- Settlements cut off: {get_kpi("settlements_cut_off")} (out of {get_kpi("settlements_total")})
- Bridges likely lost (radar return lost): {get_kpi("bridges_lost")} (out of {get_kpi("bridges_total")})
- Bridges at risk (abutment hazard): {get_kpi("bridges_at_risk")}
- Buildings hit by mapped hazard: {get_kpi("buildings_hit")} (out of {get_kpi("buildings_total")})
- Road/trail network severed: {get_kpi("network_failed_km")} km (out of {get_kpi("network_total_km")} km)
- Debris/scour extent: {get_kpi("area_debris_km2")} km²
- New open floodwater: {get_kpi("area_flood_km2")} km²
- Area not observable (radar shadow/layover/cloud): {get_kpi("area_not_observable_km2")} km²
"""

    cut_settlements = ctx.get("cut_off_settlements", [])[:10]
    settlements_list = "\n".join(
        [
            f"  * {s['name']}: status={s['status']}, buildings={s['buildings']}, P(cutoff)={s.get('p_cut_off', 'N/A')}"
            for s in cut_settlements
        ]
    )

    damaged_bridges = ctx.get("damaged_bridges", [])[:10]
    bridges_list = "\n".join(
        [
            f"  * {b['name']}: status={b['status']}, heuristic loss score={b.get('p_lost', 'N/A')}, radar drop={b.get('bright_drop_db', 'N/A')} dB"
            for b in damaged_bridges
        ]
    )

    repairs = ctx.get("repair_priorities", [])[:5]
    repairs_list = "\n".join(
        [
            f"  * Priority {i + 1}: {(r.get('name') or r.get('highway') or 'unnamed way')} reconnects {r.get('settlements_reconnected', 'unknown')} settlements and {r.get('buildings_reconnected', 'unknown')} buildings"
            for i, r in enumerate(repairs)
        ]
    )

    return f"""You are Dhara AI, an expert multimodal disaster intelligence assistant for flood, debris, and infrastructure isolation analysis.
Your primary role is to assist emergency commanders, search & rescue teams, local authorities, and the public.

CRITICAL INSTRUCTIONS:
1. TRUTHFULNESS & GROUNDING: You must ONLY state numerical facts backed by the verified disaster facts provided below. NEVER invent or hallucinate numbers or casualty counts. Hazard and bridge values are heuristic model scores, not calibrated probabilities; do not call them chances or likelihood percentages.
2. OBSERVABILITY LIMITATION: Explicitly remind users that radar shadow, gorge layover, and cloud cover obscure terrain ({get_kpi("area_not_observable_km2")} km² unobservable). "Unobserved" means unknown, NOT undamaged.
3. DEBRIS NATURE: Remind users that debris flows in mountain valleys are rough and bright in SAR, unlike dark open water.
4. LANGUAGE: Answer fluently, professionally, and accurately in the requested language: '{lang}'. Use culturally appropriate phrasing and respectful tone.

VERIFIED DISASTER FACTS:
{kpis_text}

TOP ISOLATED SETTLEMENTS:
{settlements_list if settlements_list else "  (None cut off in current analysis)"}

CRITICAL DAMAGED BRIDGES:
{bridges_list if bridges_list else "  (No bridge losses detected)"}

STRATEGIC REPAIR BOTTLENECKS (Reconnects Most Settlements):
{repairs_list if repairs_list else "  (No single repair reconnects cut-off settlements)"}
"""
