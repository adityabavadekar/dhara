"""Multimodal and Agentic LLM Service for Grounded Tactical Q&A and Situation Briefings.

By default (`ASSISTANT_LLM=off`) every answer comes from deterministic templates over the job's
verified facts and nothing leaves the machine. With `ASSISTANT_LLM=groq|gemini|openai` custom
questions go to that one provider through `dhara.adapters.allowlist.guarded_request()`; the Groq
route runs a tool-calling agent over typed queries of the job's outputs. Random or out-of-scope questions strictly trigger a constant
string fallback refusing the query.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any

from dhara.adapters.allowlist import RequestTimeout, guarded_request
from dhara.assistant.grounding import build_system_prompt, load_job_context
from dhara.config import get_settings
from dhara.osm_names import shown_name

OFF_TOPIC_FALLBACK = (
    "I can help with this analysis, including its data and satellite observations, mapped hazards, "
    "assets and access, evidence, interpretation, and limitations."
)
MODEL_UNAVAILABLE_FALLBACK = "I couldn't answer that just now. Please try again."
logger = logging.getLogger(__name__)
_CHAT_CACHE_MAX = 512
# Answers are kept on disk per job (a finished analysis never changes). Bump to drop old answers
# when the assistant's rules change.
_CHAT_CACHE_VERSION = 1
_CHAT_CACHE_LOCK = threading.Lock()
_CHAT_RESPONSE_CACHE: OrderedDict[
    tuple[str, str, str, tuple[Any, ...] | None], tuple[float, str, str]
] = OrderedDict()

EXISTING_GENERATED_PROMPTS = frozenset(
    {
        "summary of isolated villages",
        "which bridges are likely lost?",
        "top repair priority",
        "explain unobservable area",
        "which settlements are completely cut off without road access?",
        "which bridges are likely lost or at risk across river crossings?",
        "which health facilities cannot be reached by road?",
        "what is the top priority road or bridge repair to reconnect communities?",
        "what are the longest hospital evacuation detours caused by road severances?",
        "how many homes and structures are exposed in the high-hazard zone?",
        "where did flood water and debris flows cause the most damage?",
        "which areas could not be observed due to cloud cover or radar shadow?",
        "where should emergency relief and helicopter evacuations be prioritized?",
        "provide an executive situation briefing for disaster commanders.",
    }
)

GIS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_kpis",
            "description": "Retrieve verified disaster KPIs, event date, AOI location, and the satellite acquisition dates and scene IDs used in this analysis.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_settlements",
            "description": "Query surveyed settlements and villages in the AOI. Can filter by status ('cut_off', 'hospital_cut_off', 'detour', 'connected', or 'all') and minimum buildings within the analysis exposure radius. The result includes full status_counts; settlements is only a limited sample when returned_count is below count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Status filter: 'cut_off', 'hospital_cut_off', 'detour', 'connected', or 'all'",
                        "enum": ["cut_off", "hospital_cut_off", "detour", "connected", "all"],
                    },
                    "min_buildings": {
                        "type": "integer",
                        "description": "Only return settlements with at least this many buildings within the analysis exposure radius",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of settlements to return (default 15)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_bridges",
            "description": "Query surveyed bridge structures and road river crossings. Returns full status_counts plus a limited sample. Scores are heuristic, not calibrated probabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Status filter: 'likely_lost', 'at_risk', 'likely_standing', or 'all'",
                        "enum": ["likely_lost", "at_risk", "likely_standing", "all"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of bridges to return (default 15)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_health_facilities",
            "description": "Query surveyed health facilities with road network accessibility status. Returns full status_counts plus a limited sample.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by accessibility status: 'accessible', 'isolated', or 'all'",
                        "enum": ["accessible", "isolated", "all"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of facilities to return (default 15)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_restoration_priorities",
            "description": "Query ranked priority road and bridge repair bottlenecks from percolation analysis. Returns which assets restore access to the highest number of cut-off villages and buildings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of priority interventions to return (default 10)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spatial_nearest_features",
            "description": "Find nearest settlements, hospitals, or damaged bridges to a specific GPS coordinate [lat, lon] in EPSG:4326.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude in WGS84 EPSG:4326"},
                    "lon": {"type": "number", "description": "Longitude in WGS84 EPSG:4326"},
                    "feature_type": {
                        "type": "string",
                        "description": "Feature type to search: 'settlement', 'bridge', 'hospital', or 'all'",
                        "enum": ["settlement", "bridge", "hospital", "all"],
                    },
                    "limit": {"type": "integer", "description": "Maximum results (default 5)"},
                },
                "required": ["lat", "lon"],
            },
        },
    },
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two WGS84 points in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)


def _execute_gis_tool(job_dir: Path, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a GIS query against the active analysis job data."""
    if tool_name == "query_kpis":
        facts_file = job_dir / "facts.json"
        kpis: dict[str, Any] = {}
        if facts_file.exists():
            with contextlib.suppress(Exception):
                data = json.loads(facts_file.read_text())
                raw_facts = data.get("facts", {})
                if isinstance(raw_facts, dict):
                    for k, item in raw_facts.items():
                        kpis[k] = item.get("value", item.get("val", item.get("text")))
                elif isinstance(raw_facts, list):
                    for item in raw_facts:
                        kpis[item["key"]] = item.get("val", item.get("value", item.get("text")))
        summary_file = job_dir / "summary.json"
        if summary_file.exists():
            with contextlib.suppress(Exception):
                kpis.update(json.loads(summary_file.read_text()))
        job_context = load_job_context(job_dir)
        event = kpis.get("event")
        location = job_context.get("location", {})
        center = location.get("center")
        provenance: dict[str, Any] = {}
        provenance_file = job_dir / "provenance.json"
        if provenance_file.exists():
            with contextlib.suppress(Exception):
                provenance = json.loads(provenance_file.read_text())
        data_sources = provenance.get("data_sources", {})
        satellite_scans: list[dict[str, Any]] = []
        sentinel1_sources = data_sources.get("sentinel1") or []
        if not isinstance(sentinel1_sources, list):
            sentinel1_sources = []
        for pass_data in sentinel1_sources:
            if not isinstance(pass_data, dict):
                continue
            for role, key in (("pre-event", "pre"), ("baseline", "pre2"), ("post-event", "post")):
                scenes = pass_data.get(key, [])
                if isinstance(scenes, (dict, str)):
                    scenes = [scenes]
                if not isinstance(scenes, list):
                    continue
                for scene in scenes:
                    if isinstance(scene, str):
                        scene = {"scene_ids": [scene]}
                    if not isinstance(scene, dict):
                        continue
                    scene_ids = scene.get("scene_ids", [])
                    if isinstance(scene_ids, str):
                        scene_ids = [scene_ids]
                    if not isinstance(scene_ids, list):
                        scene_ids = []
                    acquisition_date = str(scene.get("start") or "")[:10] or None
                    if acquisition_date is None and scene_ids:
                        match = re.search(r"_(\d{8})T", str(scene_ids[0]))
                        if match:
                            date_text = match.group(1)
                            acquisition_date = f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]}"
                    satellite_scans.append(
                        {
                            "sensor": "Sentinel-1",
                            "role": role,
                            "relative_orbit": pass_data.get("relative_orbit"),
                            "orbit_state": pass_data.get("orbit_state"),
                            "acquisition_date": acquisition_date,
                            "scene_ids": scene_ids,
                        }
                    )
        sentinel2 = data_sources.get("sentinel2", {})
        if isinstance(sentinel2, dict):
            for role, key in (("pre-event", "pre"), ("post-event", "post")):
                scene = sentinel2.get(key, {})
                if isinstance(scene, dict):
                    scene_ids = scene.get("scene_ids", [])
                    if isinstance(scene_ids, str):
                        scene_ids = [scene_ids]
                    if not isinstance(scene_ids, list):
                        scene_ids = []
                    satellite_scans.append(
                        {
                            "sensor": "Sentinel-2",
                            "role": role,
                            "acquisition_date": scene.get("day"),
                            "scene_ids": scene_ids,
                        }
                    )
        analysis_context = {
            "event_date_or_timestamp": event,
            "event_time_recorded": isinstance(event, str) and len(event) > 10,
            "analysis_area_name": location.get("name"),
            "approximate_aoi_center_epsg4326": (
                {"longitude": center[0], "latitude": center[1]} if center else None
            ),
            "analysis_workflow": (
                "Compare satellite observations from before and after the event, map possible "
                "flood and debris changes, and relate those areas to mapped buildings and "
                "transport and health networks. Results are screening outputs, not field confirmation."
            ),
            "satellite_scans": satellite_scans,
        }
        return {"status": "success", "kpis": kpis, "analysis_context": analysis_context}

    elif tool_name == "query_settlements":
        path = job_dir / "web" / "settlements.geojson"
        if not path.exists():
            return {"status": "error", "message": "settlements layer not found"}
        with contextlib.suppress(Exception):
            gj = json.loads(path.read_text())
            feats = gj.get("features", [])
            st_filter = args.get("status")
            min_b = args.get("min_buildings")
            limit = args.get("limit", 15)

            results = []
            for f in feats:
                p = f.get("properties", {})
                st = str(p.get("status", "connected"))
                if st_filter and st_filter != "all" and st != st_filter:
                    continue
                b_count = int(p.get("buildings_500m") or 0)
                if min_b and b_count < min_b:
                    continue
                coords = f.get("geometry", {}).get("coordinates")
                results.append(
                    {
                        "name": shown_name(p.get("name")) or "Unnamed Settlement",
                        "status": st,
                        "buildings_500m": b_count,
                        "p_cut_off": p.get("p_cut_off"),
                        "nearest_hospital": p.get("hospital_name"),
                        "km_to_hospital_pre": p.get("km_to_hospital_pre"),
                        "km_to_hospital_post": p.get("km_to_hospital_post"),
                        "delay_hours": p.get("delay_hours_target"),
                        "coordinates": coords,
                    }
                )
            results.sort(key=lambda s: s["buildings_500m"], reverse=True)
            status_counts: dict[str, int] = {}
            for result in results:
                status = str(result["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
            returned = results[:limit]
            return {
                "status": "success",
                "count": len(results),
                "returned_count": len(returned),
                "truncated": len(returned) < len(results),
                "status_counts": status_counts,
                "settlements": returned,
            }
        return {"status": "error", "message": "Failed to parse settlements"}

    elif tool_name == "query_bridges":
        path = job_dir / "web" / "bridges.geojson"
        if not path.exists():
            return {"status": "error", "message": "bridges layer not found"}
        with contextlib.suppress(Exception):
            gj = json.loads(path.read_text())
            feats = gj.get("features", [])
            st_filter = args.get("status")
            limit = args.get("limit", 15)

            results = []
            for f in feats:
                p = f.get("properties", {})
                st = str(p.get("status", "likely_standing"))
                if st_filter and st_filter != "all" and st != st_filter:
                    continue
                coords = f.get("geometry", {}).get("coordinates", [None])[0]
                results.append(
                    {
                        "name": shown_name(p.get("name")) or p.get("highway") or "Bridge",
                        "highway": p.get("highway"),
                        "status": st,
                        "p_lost": p.get("p_lost"),
                        "radar_drop_db": p.get("bright_drop_db"),
                        "span_length_m": p.get("length_m"),
                        "coordinates": coords,
                    }
                )
            results.sort(key=lambda b: float(b.get("p_lost") or 0), reverse=True)
            status_counts: dict[str, int] = {}
            for result in results:
                status = str(result["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
            returned = results[:limit]
            return {
                "status": "success",
                "count": len(results),
                "returned_count": len(returned),
                "truncated": len(returned) < len(results),
                "status_counts": status_counts,
                "bridges": returned,
            }
        return {"status": "error", "message": "Failed to parse bridges"}

    elif tool_name == "query_health_facilities":
        path = job_dir / "web" / "hospitals.geojson"
        if not path.exists():
            return {"status": "error", "message": "hospitals layer not found"}
        with contextlib.suppress(Exception):
            gj = json.loads(path.read_text())
            feats = gj.get("features", [])
            st_filter = args.get("status")
            limit = args.get("limit", 15)

            results = []
            for f in feats:
                p = f.get("properties", {})
                st = str(p.get("status", "accessible"))
                if st_filter and st_filter != "all" and st != st_filter:
                    continue
                coords = f.get("geometry", {}).get("coordinates")
                results.append(
                    {
                        "name": shown_name(p.get("name")) or "Emergency Health Facility",
                        "amenity": p.get("amenity") or p.get("kind") or "hospital",
                        "status": st,
                        "is_isolated": bool(p.get("is_isolated")),
                        "coordinates": coords,
                    }
                )
            status_counts: dict[str, int] = {}
            for result in results:
                status = str(result["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
            returned = results[:limit]
            return {
                "status": "success",
                "count": len(results),
                "returned_count": len(returned),
                "truncated": len(returned) < len(results),
                "status_counts": status_counts,
                "facilities": returned,
            }
        return {"status": "error", "message": "Failed to parse health facilities"}

    elif tool_name == "query_restoration_priorities":
        path = job_dir / "repairs.json"
        if not path.exists():
            return {"status": "error", "message": "repairs data not found"}
        with contextlib.suppress(Exception):
            data = json.loads(path.read_text())
            limit = args.get("limit", 10)
            items = []
            for i, r in enumerate(data[:limit]):
                items.append(
                    {
                        "rank": i + 1,
                        "name": shown_name(r.get("name")) or r.get("highway") or "Corridor",
                        "is_bridge": r.get("is_bridge", False),
                        "settlements_reconnected": r.get("settlements_reconnected", 0),
                        "buildings_reconnected": r.get("buildings_reconnected", 0),
                        "osm_id": r.get("osm_id"),
                    }
                )
            return {"status": "success", "count": len(items), "priorities": items}
        return {"status": "error", "message": "Failed to parse repairs"}

    elif tool_name == "spatial_nearest_features":
        lat, lon = float(args["lat"]), float(args["lon"])
        f_type = args.get("feature_type", "all")
        limit = args.get("limit", 5)

        candidates: list[dict[str, Any]] = []

        if f_type in ("settlement", "all"):
            s_path = job_dir / "web" / "settlements.geojson"
            if s_path.exists():
                with contextlib.suppress(Exception):
                    gj = json.loads(s_path.read_text())
                    for f in gj.get("features", []):
                        p = f.get("properties", {})
                        pt = f.get("geometry", {}).get("coordinates")
                        if pt and len(pt) >= 2:
                            d_km = _haversine_km(lat, lon, pt[1], pt[0])
                            candidates.append(
                                {
                                    "type": "settlement",
                                    "name": shown_name(p.get("name")) or "Settlement",
                                    "osm_id": p.get("osm_id"),
                                    "status": p.get("status"),
                                    "distance_km": d_km,
                                    "coordinates": pt,
                                }
                            )

        if f_type in ("hospital", "all"):
            h_path = job_dir / "web" / "hospitals.geojson"
            if h_path.exists():
                with contextlib.suppress(Exception):
                    gj = json.loads(h_path.read_text())
                    for f in gj.get("features", []):
                        p = f.get("properties", {})
                        pt = f.get("geometry", {}).get("coordinates")
                        if pt and len(pt) >= 2:
                            d_km = _haversine_km(lat, lon, pt[1], pt[0])
                            candidates.append(
                                {
                                    "type": "hospital",
                                    "name": shown_name(p.get("name")) or "Hospital",
                                    "osm_id": p.get("osm_id"),
                                    "status": p.get("status"),
                                    "distance_km": d_km,
                                    "coordinates": pt,
                                }
                            )

        if f_type in ("bridge", "all"):
            b_path = job_dir / "web" / "bridges.geojson"
            if b_path.exists():
                with contextlib.suppress(Exception):
                    gj = json.loads(b_path.read_text())
                    for f in gj.get("features", []):
                        p = f.get("properties", {})
                        coords = f.get("geometry", {}).get("coordinates")
                        pt = coords[0] if coords and isinstance(coords[0], list) else coords
                        if pt and len(pt) >= 2:
                            d_km = _haversine_km(lat, lon, pt[1], pt[0])
                            candidates.append(
                                {
                                    "type": "bridge",
                                    "name": shown_name(p.get("name"))
                                    or p.get("highway")
                                    or "Bridge",
                                    "osm_id": p.get("osm_id"),
                                    "status": p.get("status"),
                                    "distance_km": d_km,
                                    "coordinates": pt,
                                }
                            )

        candidates.sort(key=lambda x: x["distance_km"])
        return {
            "status": "success",
            "origin": {"lat": lat, "lon": lon},
            "nearest": candidates[:limit],
        }

    return {"status": "error", "message": f"Unknown tool: {tool_name}"}


def _is_obviously_off_topic(q: str) -> bool:
    """Pre-filter obvious random or out-of-scope non-GIS questions with zero latency."""
    q_clean = q.lower().strip()
    patterns = [
        r"\bwrite\b.{0,30}\b(poem|song|story|code|script|essay|joke)s?\b",
        r"\b(who\s+is\s+(batman|superman|elon|trump|biden|modi|the\s+president))\b",
        r"\b(what\s+is\s+the\s+capital\s+of)\b",
        r"\b(how\s+to\s+(bake|cook|make|code|program|hack))\b",
        r"\b(tell\s+me\s+a\s+joke)\b",
        r"\b(what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+)\b",
        r"^(hi|hello|hey|test|ping|how are you)\b",
        # Questions about the assistant itself: answered with the standard reply, so no
        # provider or model name is ever exposed.
        r"\bare\s+you\s+(an?\s+)?(ai|bot|robot|human|chatgpt|llm)\b",
        r"^what\s+are\s+you\b",
        r"\bwho\s+(made|built|created|trained)\s+you\b",
    ]
    return any(re.search(pat, q_clean) for pat in patterns)


def _call_tool_agent(
    api_key: str,
    job_dir: Path,
    question: str,
    lang: str = "en",
    model: str = "openai/gpt-oss-120b",
    endpoint: str = "https://api.groq.com/openai/v1/chat/completions",
    max_tokens: int | None = 1024,
    selected_feature_record: str | None = None,
) -> tuple[str, str]:
    """Run a tool-calling GIS assistant through an OpenAI-compatible chat endpoint."""
    if _is_obviously_off_topic(question):
        return (OFF_TOPIC_FALLBACK, "grounded_rules")

    system_prompt = (
        f"You are the Dhara Disaster Response Assistant, an expert disaster response assistant.\n"
        f"Respond in language: {lang}.\n\n"
        f"OPERATIONAL MANDATES:\n"
        f"Assume your users are disaster researchers, emergency decision-makers, and relief coordinators. Questions about data provenance, satellite observations, interpretation, evidence, uncertainty, limitations, field follow-up, mapped assets/access, and the assistant's role are in scope.\n"
        f"RESPONSE STYLE: Be concise and specific. Start with the direct answer; usually use 1–3 sentences or up to 4 bullets. Use exact names, dates, statuses, and evidence from the tools. Answer every part asked, but do not repeat the question, add a generic briefing, or include unrequested details. Expand only when the user asks for explanation, comparison, or a full briefing.\n"
        f"If asked what you can explain, name the relevant capabilities briefly; do not treat a broad but in-scope question as a reason to refuse. For a broad overview, give only the event/area if recorded, a few key results, and the main caveat. Do not list every scene ID or KPI unless asked.\n"
        f"Never use internal technical jargon like 'GIS algorithms', 'Dijkstra', 'graph mechanics', or 'backend pipeline'.\n"
        f"1. STRICT FACTUAL ACCURACY: Ground job-specific facts in query tools. You may explain documented workflow and limitations from this prompt without inventing job-specific results. Clearly distinguish measured outputs, heuristic scores, and unverified interpretation.\n"
        f"2. ALWAYS CALL QUERY TOOLS: Before answering queries regarding event date/time, analysis area, satellite acquisitions/scans, how this analysis works, disaster extent, cut-off villages, damaged bridges, health facilities, nearby or nearest features, or restoration priority, call the appropriate query tools to retrieve facts. Use query_kpis for event date/time, AOI metadata, workflow context, and scan records. For nearby/nearest questions with a selected map point, call spatial_nearest_features using the selected point's latitude and longitude.\n"
        f"2a. TOOL SCOPE: For aggregate claims use count and status_counts. Returned asset lists may be truncated; never generalize from the returned sample to all records. No tool provides ambulance routing, vehicle passability, helicopter access, or evacuation need, so explicitly say those cannot be determined. A nearby road, bridge, or hospital is not a verified route.\n"
        f"3b. UNITS: every number must say what it counts (e.g. '220 buildings'), even if the user asks for a number only.\n"
        f"3a. PEOPLE: population is not an input. Building counts are buildings, never people, citizens or households. "
        f"If asked how many people are affected, say population is not known and give the building count as buildings.\n"
        f"3. NO HALLUCINATION: Never invent, extrapolate, or estimate figures. 'Not observable' means ground hidden from the radar only; for things this system does not measure (structural damage, arrival times, people, livestock, casualties) say they are not measured.\n"
        f"4. CALIBRATION: Hazard and bridge values are heuristic model scores, not calibrated probabilities. Never describe them as chances or likelihood percentages; state the score and its evidence instead.\n"
        f"4a. OBSERVABILITY: Questions about radar shadow, cloud, or unobserved ground are in scope. State that unobserved means unknown, never no damage.\n"
        f"4b. MULTI-PART QUESTIONS: Answer each part of the user's question under a short, clear label. Do not replace an unanswered part with a generic disaster summary. For questions about how Dhara works, explain only the high-level workflow: compare pre-event and post-event satellite observations, map possible flood/debris changes, and relate these to mapped assets and access. Do not claim field confirmation.\n"
        f"4c. MODEL IDENTITY: Questions about the assistant used in this deployment are in scope. Do not disclose a backend provider or model name; say those names are not displayed in this interface.\n"
        f"4d. LOCATION: Use only the AOI name returned by query_kpis. If it is absent, say no city name is stored and use the approximate AOI center if useful; never infer a city from coordinates.\n"
        f"Do not infer a country, region, municipality, or AOI area from coordinates. Only report coordinates and area values explicitly provided by query_kpis. Do not print technical CRS codes such as EPSG labels.\n"
        f"Do not add unrequested numbers or calculations; for a broad overview, summarize evidence and main limitations without dumping every KPI.\n"
        f"4e. SCANS: For acquisition questions, report only scans in query_kpis analysis_context.satellite_scans, including sensor, event-relative role, date, and scene IDs when useful. Treat roles as mutually exclusive: 'pre-event' is the pre role, 'baseline' is the baseline role, and 'post-event' is the post role. Never duplicate a scene under multiple roles. Do not invent scans or dates.\n"
        f"Format each long satellite scene ID as inline Markdown code so the interface can display it as a copyable identifier.\n"
        f"4f. LIMITATIONS: State only limitations grounded in the analysis context or documented in the provided facts. Do not assume scene coverage gaps, asset-list sampling, or missing ground truth unless the data says so. Explain that mapped/modelled results need local verification where relevant.\n"
        f"Questions about any named place, settlement, village, road, bridge or facility in this area are IN SCOPE: "
        f"look it up with the query tools (names may be in Nepali or another script, so search broadly).\n"
        f"4g. SCOPE: Answer questions about this job and disaster-analysis context, including broad requests for a summary or explanation. For unrelated requests, briefly state that you can help with this analysis and its evidence, hazards, assets, access, and limitations. Do not use a refusal sentinel."
    )
    if selected_feature_record:
        system_prompt += (
            "\n\nSELECTED MAP FEATURE RECORD (verified job data):\n"
            + selected_feature_record
            + "\nWhen the user says this, that, or it, they mean this selected feature. Answer the "
            "question about it directly and concisely. Do not describe yourself or the assistant."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = endpoint

    # Allow up to three tool rounds and a final answer, under the browser's request timeout.
    deadline = time.monotonic() + 45.0
    for _ in range(4):
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        payload = {
            "model": model,
            "messages": messages,
            "tools": GIS_TOOLS,
            "tool_choice": "auto",
            "temperature": 0.1,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        resp = guarded_request(
            "POST",
            url,
            json=payload,
            headers=headers,
            retries=0,
            timeout_s=min(15.0, remaining_s),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API returned HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls")

        if not tool_calls:
            content = str(msg.get("content") or "").strip()
            # If the model returned the 1-token sentinel REFUSE or recognized off-topic, normalize to the verified fallback string
            if (
                content.upper() == "REFUSE"
                or content.upper().startswith("REFUSE")
                or "i cannot assist with queries outside verified" in content.lower()
                or "strictly calibrated" in content.lower()
            ):
                _write_conversation_log(
                    job_dir, question, lang, messages, OFF_TOPIC_FALLBACK, "grounded_rules"
                )
                return (OFF_TOPIC_FALLBACK, "grounded_rules")
            ans = content or "No response generated by model."
            _write_conversation_log(job_dir, question, lang, messages, ans, "grounded_agent")
            return (ans, "grounded_agent")

        # Execute tool calls
        messages.append(msg)
        for tc in tool_calls:
            t_name = tc.get("function", {}).get("name", "")
            t_raw_args = tc.get("function", {}).get("arguments", "{}")
            with contextlib.suppress(Exception):
                t_args = json.loads(t_raw_args) if isinstance(t_raw_args, str) else t_raw_args
            t_res = _execute_gis_tool(job_dir, t_name, t_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": json.dumps(t_res),
                }
            )

    final_answer = ("GIS Agent exceeded its response budget.", "grounded_agent")
    _write_conversation_log(job_dir, question, lang, messages, final_answer[0], final_answer[1])
    return final_answer


# A model that failed on every account is skipped for this long (overload usually passes).
MODEL_COOLDOWN_S = 120.0
_model_down_until: dict[str, float] = {}
GEMINI_OPENAI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def _call_groq_with_account_fallback(
    api_keys: list[str],
    job_dir: Path,
    question: str,
    lang: str,
    model: str,
    selected_feature_record: str | None = None,
    endpoint: str = GROQ_ENDPOINT,
) -> tuple[str, str]:
    """Try configured accounts in order when one is unavailable, rate-limited or failing (any
    OpenAI-compatible endpoint: Groq, or Gemini's compatibility endpoint). The model only reads
    this job's results through the GIS tools; no web search tool is ever given to it."""
    last_error: Exception | None = None
    for api_key in api_keys:
        try:
            return _call_tool_agent(
                api_key,
                job_dir,
                question,
                lang=lang,
                model=model,
                endpoint=endpoint,
                max_tokens=2048,
                selected_feature_record=selected_feature_record,
            )
        except Exception as exc:
            retryable = isinstance(exc, RequestTimeout) or (
                isinstance(exc, RuntimeError)
                and re.search(r"HTTP (?:401|403|429|500|502|503|504)\b", str(exc))
            )
            if not retryable:
                raise
            last_error = exc
            logger.warning(
                "Assistant request failed for one configured account; trying next account"
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("No Groq accounts are configured")


def _write_conversation_log(
    job_dir: Path,
    question: str,
    lang: str,
    messages: list[dict[str, Any]],
    answer: str,
    provider: str,
) -> None:
    """Log full conversation, model responses, and GIS tool calls for researcher audit."""
    with contextlib.suppress(Exception):
        from datetime import UTC, datetime

        log_path = job_dir / "assistant_log.jsonl"
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "question": question,
            "lang": lang,
            "provider": provider,
            "answer": answer,
            "dialogue_history": messages,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def _call_gemini_api(api_key: str, system_prompt: str, user_prompt: str, model: str) -> str:
    """Call Google Gemini API via guarded_request()."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }
    resp = guarded_request(
        "POST", url, json=payload, headers={"x-goog-api-key": api_key}, retries=0, timeout_s=45.0
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API returned HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return "No response generated by model."
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def _call_openai_api(api_key: str, system_prompt: str, user_prompt: str, model: str) -> str:
    """Call OpenAI API via guarded_request()."""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = guarded_request("POST", url, json=payload, headers=headers, retries=0, timeout_s=45.0)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API returned HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return "No response generated by model."
    return choices[0].get("message", {}).get("content", "")


def _generate_grounded_fallback(ctx: dict[str, Any], question: str, lang: str = "en") -> str:
    """Deterministic multilingual responder used for canned prompts or fallback."""
    k = ctx.get("kpis", {})

    def get_kpi(key: str, default: str = "unknown") -> str:
        # A missing fact is "unknown", never a made-up zero.
        return str(k.get(key, {}).get("text") or "unknown")

    settlements_cut = get_kpi("settlements_cut_off")
    bridges_lost = get_kpi("bridges_lost")
    bridges_at_risk = get_kpi("bridges_at_risk")
    bridges_total = get_kpi("bridges_total")
    buildings_hit = get_kpi("buildings_hit")
    road_km = get_kpi("network_failed_km")
    unobs_km2 = get_kpi("area_not_observable_km2")
    debris_km2 = get_kpi("area_debris_km2")

    top_settlements = [s for s in ctx.get("cut_off_settlements", []) if s["status"] == "cut_off"][
        :5
    ]
    settlement_names = (
        ", ".join([s["name"] for s in top_settlements]) if top_settlements else "None"
    )

    top_repairs = ctx.get("repair_priorities", [])[:3]
    top_repair_text = ""
    if top_repairs:
        r0 = top_repairs[0]
        kind = "Bridge" if r0.get("is_bridge") else "Road"
        what = r0.get("name") or f"unnamed {r0.get('highway') or 'way'}"
        top_repair_text = (
            f"{kind} ({what}) reconnects "
            f"{r0.get('settlements_reconnected', 'unknown')} settlements and {r0.get('buildings_reconnected', 'unknown')} buildings."
        )

    lang_lower = lang.lower()
    q_lower = question.lower()
    is_ne = lang_lower.startswith("ne") or "nepal" in lang_lower

    if any(
        w in q_lower
        for w in ("people", "citizen", "population", "resident", "household", "जनसंख्या", "मानिस")
    ):
        asks_hazard_exposure = any(
            term in q_lower
            for term in ("hazard", "flood", "debris", "overlap", "buildings", "structures")
        )
        asks_access_exposure = any(
            term in q_lower
            for term in ("cut off", "cut-off", "isolated", "road access", "hospital access")
        )
        if is_ne:
            answer = (
                "### प्रभावित जनसंख्या\n\n"
                "जनसंख्या तथ्यांक यस विश्लेषणको इनपुट होइन, त्यसैले मानिसको संख्या थाहा छैन।\n"
            )
            if asks_hazard_exposure:
                answer += f"\n- प्रकोप क्षेत्रमा **{buildings_hit} भवन** छन्; यो मानिसको संख्या होइन।\n"
            if asks_access_exposure or not asks_hazard_exposure:
                answer += (
                    f"- सडक वा अस्पताल पहुँच गुमाएका बस्तीको {get_kpi('exposure_radius_m')} मिटरभित्र "
                    f"**{get_kpi('buildings_in_cut_off')} भवन** छन्।\n"
                )
            return answer
        answer = (
            "### People at risk\n\n"
            "Population is not an input to this analysis, so it cannot give a number of people affected.\n"
        )
        if asks_hazard_exposure:
            answer += f"\n- **{buildings_hit} buildings** overlap the mapped flood or debris area; this is not a people count.\n"
        if asks_access_exposure or not asks_hazard_exposure:
            answer += (
                f"- **{get_kpi('buildings_in_cut_off')} buildings** lie within "
                f"{get_kpi('exposure_radius_m')} m of settlements that lost road or hospital access.\n"
            )
        return answer

    if any(
        w in q_lower
        for w in ("cut off", "cut-off", "isolated", "village", "settlement", "सम्पर्कविहीन", "बस्ती")
    ):
        exposure_radius = get_kpi("exposure_radius_m")
        groups = (
            ("cut_off", "No road out at all", "कुनै सडक बाहिर छैन"),
            ("hospital_cut_off", "No road to a hospital", "अस्पतालसम्म सडक छैन"),
            ("detour", "Longer route than before", "पहिलेभन्दा लामो बाटो"),
        )
        out = ["### सम्पर्कविहीन बस्तीहरू" if is_ne else "### Settlements that lost road access", ""]
        for key, en, ne_label in groups:
            rows = [s for s in ctx.get("cut_off_settlements", []) if s["status"] == key]
            if not rows:
                continue
            out.append(f"**{ne_label if is_ne else en} ({len(rows)})**")
            for s in rows[:8]:
                out.append(
                    f"- {s['name']}: {s['buildings']} {'भवन' if is_ne else f'buildings within {exposure_radius} m'}"
                )
            if len(rows) > 8:
                out.append(
                    f"- … {'र' if is_ne else 'and'} {len(rows) - 8} {'थप' if is_ne else 'more'}"
                )
            out.append("")
        if len(out) == 2:
            out.append(
                "कुनै बस्तीले सडक पहुँच गुमाएको छैन।" if is_ne else "No settlement lost road access."
            )
        return "\n".join(out)

    if any(w in q_lower for w in ("hospital", "clinic", "health", "detour", "अस्पताल")):
        hosp_cut = get_kpi("hospitals_isolated")
        hosp_total = get_kpi("hospitals_total")
        try:
            no_isolated_facilities = float(hosp_cut.replace(",", "")) == 0
        except ValueError:
            no_isolated_facilities = False
        if no_isolated_facilities:
            if is_ne:
                return (
                    f"नक्साङ्कित {hosp_total} स्वास्थ्य संस्थामध्ये कुनैको पनि सडक पहुँच छुटेको छैन (0)। "
                    "यसले हालको सडक चल्ने अवस्था पुष्टि गर्दैन।"
                )
            return (
                f"No mapped health facility has lost road access (0 of {hosp_total}). "
                "This does not verify current road passability."
            )
        isolated_names = [
            str(facility["name"])
            for facility in ctx.get("health_facilities", [])
            if facility.get("is_isolated")
        ]
        if is_ne:
            names = ", ".join(isolated_names) if isolated_names else "नाम उपलब्ध छैन"
            return (
                f"सडक पहुँच नभएका स्वास्थ्य संस्था: {hosp_cut} / {hosp_total}। "
                f"{names}। हालको सडक चल्ने अवस्था पुष्टि गरिएको छैन।"
            )
        names = (
            ", ".join(isolated_names) if isolated_names else "Names are unavailable in this result."
        )
        return (
            f"{hosp_cut} of {hosp_total} mapped health facilities lack road access: {names}. "
            "This does not verify current road passability."
        )

    if any(w in q_lower for w in ("rescue", "helicopter", "relief", "उद्धार")):
        if is_ne:
            return (
                f"### आपतकालीन उद्धार तथा प्राथमिक क्षेत्र\n\n"
                f"- **सर्वोच्च प्राथमिकता बस्तीहरू:** {settlement_names}\n"
                f"- **सम्पर्कविहीन संरचनाहरू:** {buildings_hit} घर/भवनहरू\n"
                "- **सीमा:** यो विश्लेषणले उद्धारको आवश्यकता वा हवाई पहुँच निर्धारण गर्दैन।\n"
            )
        return (
            f"### Emergency Relief & Evacuation Priorities\n\n"
            f"- **Top Isolated Settlements:** {settlement_names}\n"
            f"- **Buildings in mapped hazard areas:** {buildings_hit}\n"
            "- **Limit:** This analysis does not determine rescue need, aircraft access, or evacuation priority.\n"
        )

    if any(w in q_lower for w in ("debris", "extent", "flood water", "गेग्रान", "बाढी")):
        water_km2 = get_kpi("area_flood_km2")
        if is_ne:
            return (
                f"### बाढी तथा गेग्रान प्रभाव क्षेत्रफल\n\n"
                f"- **गेग्रान तथा पहिरो क्षेत्रफल:** {debris_km2} वर्ग कि.मी.\n"
                f"- **जलमग्न क्षेत्रफल:** {water_km2} वर्ग कि.मी.\n"
                f"- **क्षतिग्रस्त सडक सञ्जाल:** {road_km} कि.मी.\n"
            )
        return (
            f"### Flood Water & Debris Flow Impact\n\n"
            f"- **Debris & Landslide Scour:** {debris_km2} km²\n"
            f"- **Inundated Surface Water:** {water_km2} km²\n"
            f"- **Severed Road Network:** {road_km} km\n"
            f"- **Impacted Buildings:** {buildings_hit} structures within active hazard footprint\n"
        )

    if any(w in q_lower for w in ("bridge", "पुल")) and not any(
        w in q_lower for w in ("repair", "मर्मत")
    ):
        rows = ctx.get("damaged_bridges", [])
        out = [
            "### बगेका वा जोखिममा रहेका पुल" if is_ne else "### Bridges likely lost or at risk",
            "",
            (
                f"{bridges_lost} हराएको सम्भावित, {bridges_at_risk} जोखिममा (जम्मा {bridges_total} पुलमध्ये)."
                if is_ne
                else f"{bridges_lost} likely lost; {bridges_at_risk} at risk out of {bridges_total} mapped bridges."
            ),
            "मोडेल स्कोर अनुमानमा आधारित हुन्, क्यालिब्रेट गरिएका सम्भाव्यता होइनन्."
            if is_ne
            else "Model scores are heuristic, not calibrated probabilities.",
            "",
        ]
        for i, b in enumerate(rows[:10]):
            pl = b.get("p_lost")
            try:
                score = f"{float(pl):.2f}"
            except (TypeError, ValueError):
                score = "unknown"
            label = {"likely_lost": "likely lost", "at_risk": "at risk"}.get(
                b["status"], b["status"]
            )
            priority = ("पहिले निरीक्षण गर्नुहोस् — " if is_ne else "Inspect first — ") if i == 0 else ""
            out.append(f"- **{priority}{b['name']}**: {label}, heuristic loss score {score}")
            try:
                radar_drop = float(b.get("bright_drop_db"))
                if not math.isfinite(radar_drop):
                    radar_drop = None
            except (TypeError, ValueError):
                radar_drop = None
            if radar_drop is not None:
                out.append(
                    f"  - {'राडार फर्कावट घटेको' if is_ne else 'Observed radar backscatter drop'}: "
                    f"{radar_drop:.2f} dB."
                )
            elif not is_ne:
                out.append("  - No bridge-level radar-change value is available in this result.")
        if len(rows) > 10:
            out.append(
                f"- … {'र' if is_ne else 'and'} {len(rows) - 10} {'थप' if is_ne else 'more'}"
            )
        if not rows:
            out.append(
                "कुनै पुल बगेको वा जोखिममा छैन।" if is_ne else "No bridge is likely lost or at risk."
            )
        return "\n".join(out)

    if any(w in q_lower for w in ("bridge", "repair", "पुल", "मर्मत")):
        if is_ne:
            return (
                f"### पुल क्षति तथा मर्मत प्राथमिकता विश्लेषण\n\n"
                f"- **अनुमानित नष्ट पुलहरू:** {bridges_lost} (कुल {bridges_total} मध्ये)\n"
                f"- **शीर्ष मर्मत प्राथमिकता:** {top_repair_text or 'कुनै एकल मर्मतले सबै सम्पर्क पुनर्स्थापित गर्दैन'}\n"
            )
        return (
            f"### Bridge Damage & Restoration Assessment\n\n"
            f"- **Bridges Likely Lost:** {bridges_lost} of {bridges_total}\n"
            f"- **Strategic Bottleneck:** {top_repair_text or 'No single repair reconnected all isolated nodes'}\n"
        )

    if any(w in q_lower for w in ("unobs", "shadow", "cloud", "अदृश्य")):
        if is_ne:
            return (
                f"### नदेखिएको क्षेत्र विश्लेषण (Not Observable Area)\n\n"
                f"- **नदेखिएको क्षेत्रफल:** {unobs_km2} वर्ग कि.मी.\n"
                f"- **कारण:** राडार छायाँ वा बाक्लो बादल।\n"
            )
        return (
            f"### Unobservable Area Assessment\n\n"
            f"- **Unobservable Extent:** {unobs_km2} km²\n"
            f"- **Cause:** Radar layover/shadow or thick cloud cover.\n"
        )

    if is_ne:
        return (
            f"### विपद् प्रभाव स्थिति प्रतिवेदन\n\n"
            f"- **सम्पर्कविहीन बस्तीहरू:** {settlements_cut} ({settlement_names})\n"
            f"- **अनुमानित नष्ट पुलहरू:** {bridges_lost} (कुल {bridges_total} मध्ये)\n"
            f"- **प्रभावित संरचनाहरू:** {buildings_hit}\n"
            f"- **अवरुद्ध सडक सञ्जाल:** {road_km} कि.मी.\n"
            f"- **गेग्रान बहाव:** {debris_km2} वर्ग कि.मी.\n"
            f"- **शीर्ष मर्मत प्राथमिकता:** {top_repair_text or 'कुनै छैन'}\n"
        )

    return (
        f"### Disaster Assessment Summary\n\n"
        f"- **Cut-off Settlements:** {settlements_cut} ({settlement_names})\n"
        f"- **Bridges Likely Lost:** {bridges_lost} of {bridges_total}\n"
        f"- **Buildings Affected:** {buildings_hit}\n"
        f"- **Severed Network:** {road_km} km\n"
        f"- **Debris Extent:** {debris_km2} km²\n"
        f"- **Top Restoration Priority:** {top_repair_text or 'None'}\n"
    )


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


def _numbers(text: str) -> set[str]:
    return {m.group(0).rstrip(",").replace(",", "") for m in _NUM.finditer(text.translate(_ASCII))}


def _unbacked_numbers(answer: str, backing: str) -> list[str]:
    """Return numeric values in an answer that do not occur in its evidence."""
    ok = _numbers(backing)
    # Markdown ordered-list markers are structure, not factual quantities.
    answer = re.sub(r"(?m)^\s*\d+[.)]\s+", "", answer)
    out = []
    for n in _numbers(answer):
        if n in ok:
            continue
        # 12.30 vs 12.3, 7 vs 7.0
        with contextlib.suppress(ValueError):
            precision = len(n.partition(".")[2])
            if any(
                math.isclose(float(n), float(o), rel_tol=1e-9, abs_tol=1e-9)
                or (precision > 0 and round(float(o), precision) == float(n))
                for o in ok
                if o.replace(".", "", 1).isdigit()
            ):
                continue
        out.append(n)
    return sorted(out)


def _backing_text(job_dir: Path, ctx: dict[str, Any]) -> str:
    """Evidence corpus for numeric checks; user-supplied values are never evidence."""
    parts = [json.dumps(ctx, default=str)]
    for name in (
        "web/settlements.geojson",
        "web/bridges.geojson",
        "web/hospitals.geojson",
        "repairs.json",
        "provenance.json",
    ):
        p = job_dir / name
        if p.exists():
            parts.append(p.read_text())
    return "\n".join(parts)


# What each headline fact counts, for labelling a model reply that is only a number.
FACT_LABELS = {
    "buildings_hit": ("buildings in the mapped flood or debris zone", "प्रकोप क्षेत्रमा भवन"),
    "buildings_in_cut_off": (
        "buildings near settlements that lost road or hospital access",
        "पहुँच गुमाएका बस्ती नजिकका भवन",
    ),
    "settlements_cut_off": ("settlements with no road out", "बाहिर जाने सडक नभएका बस्ती"),
    "settlements_hospital_cut_off": (
        "settlements without road access to a hospital",
        "अस्पताल पहुँच नभएका बस्ती",
    ),
    "settlements_detour": ("settlements that need a longer route", "लामो बाटो चाहिने बस्ती"),
    "bridges_lost": ("bridges likely lost", "बगेको सम्भावित पुल"),
    "bridges_at_risk": ("bridges at risk", "जोखिममा रहेका पुल"),
    "network_failed_km": ("km of road and trail impassable", "कि.मी. सडक अवरुद्ध"),
    "area_debris_km2": ("km² of debris and scour", "वर्ग कि.मी. लेदो"),
    "area_flood_km2": ("km² of new water", "वर्ग कि.मी. नयाँ पानी"),
}


def _label_bare_number(answer: str, ctx: dict[str, Any], lang: str) -> str:
    """A reply that is only a number gets the label of the fact it equals, so it is never read
    as something else (for example buildings read as people). Unknown bare numbers are left for
    the unbacked-number check to reject."""
    bare = answer.strip().strip("*_ ").rstrip(".")
    if not _NUM.fullmatch(bare.translate(_ASCII)):
        return answer
    kpis = ctx.get("kpis", {})
    ne = lang.lower().startswith("ne")
    for key, (en, ne_label) in FACT_LABELS.items():
        if str(kpis.get(key, {}).get("text")) == bare:
            return f"{bare} {ne_label if ne else en}"
    return answer


def _not_measured(question: str, ctx: dict[str, Any], lang: str) -> str | None:
    """Fixed answers for things this system does not measure, so they are never guessed at
    and never mislabelled "not observable" (which only means ground hidden from the radar)."""
    q = question.lower()
    ne = lang.lower().startswith("ne")
    k = ctx.get("kpis", {})

    def t(key: str) -> str:
        return str(k.get(key, {}).get("text") or "unknown")

    asks_unscoped_damage_confirmation = (
        "confirm" in q
        and "damag" in q
        and not any(term in q for term in ("bridge", "road", "facility", "settlement"))
    )
    if (
        any(w in q for w in ("structural", "structurally", "collapse", "damage grade"))
        or asks_unscoped_damage_confirmation
    ):
        if ne:
            return (
                "### संरचनात्मक क्षति\n\nयो प्रणालीले भवनको संरचनात्मक क्षति मापन गर्दैन। "
                f"प्रकोप क्षेत्रमा **{t('buildings_hit')} भवन** छन्; ती प्रकोपसँग खप्टिएका भवन हुन्, "
                "पुष्टि भएको क्षति होइन। पुष्टिका लागि स्थलगत वा ड्रोन सर्वेक्षण चाहिन्छ।"
            )
        return (
            "### Structural damage\n\nNot measured by this system. On a 10 m grid a house is one or two pixels, "
            f"so the **{t('buildings_hit')} buildings** in the hazard zone are buildings whose footprint overlaps "
            "mapped flood or debris, not confirmed damage. Confirming damage needs a field or drone survey, or "
            "very-high-resolution imagery."
        )
    if any(
        w in q
        for w in ("arrive", "arrival", "when will", "eta", "forecast", "how long until", "predict")
    ) and not _is_multipart_question(question):
        if ne:
            return (
                "### बाढी आइपुग्ने समय\n\nयो प्रणालीले पूर्वानुमान गर्दैन। बाढी मार्गका समय अनुमानित नियमबाट निकालिएका हुन्, "
                "कुनै अवलोकित बाढीसँग मिलाइएका छैनन्; चेतावनीको क्रम मिलाउन मात्र प्रयोग गर्नुहोस्।"
            )
        return (
            "### Flood arrival time\n\nNot predicted by this system. Flood-path times come from an assumed "
            "slope-to-speed rule that has not been calibrated against any observed flood, so they only rank which "
            "settlements a surge would reach first. Sentinel satellites revisit every few days, so the system cannot "
            "give a live warning or an actual arrival time."
        )
    if any(
        w in q
        for w in (
            "livestock",
            "animal",
            "cattle",
            "goat",
            "buffalo",
            "sheep",
            "crop",
            "harvest",
            "पशु",
            "बाली",
        )
    ):
        if ne:
            return "### पशुधन तथा बाली\n\nपशुधन वा बाली यस विश्लेषणको इनपुटमा छैनन्; उपग्रह तस्बिरले जनावर गन्न सक्दैन। अनुमान गर्न सकिँदैन।"
        return (
            "### Livestock and crops\n\nNot in the inputs, and 10 m satellite images cannot count animals. "
            "No estimate can be given."
        )
    if any(
        w in q
        for w in (
            "death",
            "dead",
            "died",
            "casualt",
            "injur",
            "missing",
            "fatalit",
            "killed",
            "मृत्यु",
        )
    ):
        if ne:
            return "### मानवीय क्षति\n\nयो प्रणालीले मृत्यु, घाइते वा हराइरहेकाको संख्या मापन गर्दैन। आधिकारिक संख्या सरकारी निकायबाट लिनुहोस्।"
        return (
            "### Casualties\n\nNot measured by this system: satellite images and maps cannot count deaths, injuries "
            "or missing people. Use official figures from the national disaster authority."
        )
    return None


_SITREP_WORDS = (
    "situation report",
    "sitrep",
    "sit rep",
    "situation summary",
    "स्थिति प्रतिवेदन",
    "प्रतिवेदन",
)


_CAPABILITY_WORDS = (
    "what can you do",
    "what can you answer",
    "what can i ask",
    "what do you do",
    "how can you help",
    "your capabilities",
    "help me",
    "तपाईं के गर्न सक्नुहुन्छ",
    "के सोध्न सकिन्छ",
    "के गर्न सक्छौ",
)


def _capabilities(question: str, lang: str) -> str | None:
    """What the assistant can answer, stated exactly (fixed text, never generated)."""
    q = question.lower().strip(" ?!.")
    if not (q in ("help", "सहयोग") or any(w in q for w in _CAPABILITY_WORDS)):
        return None
    if lang.lower().startswith("ne"):
        return (
            "म यही विश्लेषणको नतिजाबाट मात्र जवाफ दिन्छु:\n"
            "- **बाढी कहाँ पुग्यो:** नक्साङ्कित बाढी र लेदोको क्षेत्रफल, राडारले नदेखेको क्षेत्र।\n"
            "- **के क्षति भयो:** बाढी/लेदो क्षेत्रमा पर्ने भवन, सडक र पुल (घटनाअघिको OpenStreetMap अनुसार)।\n"
            "- **को सम्पर्कविहीन छ:** सडक सम्पर्क वा अस्पतालसम्मको सडक पहुँच गुमाएका बस्ती, लामो बाटो, र कुन मर्मतले सबैभन्दा धेरै पहुँच फर्काउँछ।\n"
            "- नाम दिइएको बस्ती, पुल वा स्वास्थ्य संस्था, वा कुनै निर्देशाङ्कको नजिकको संरचना।\n"
            "- अङ्ग्रेजी वा नेपालीमा छोटो स्थिति प्रतिवेदन।\n"
            "- यो विश्लेषणले के भन्न सक्दैन (मानिसको सङ्ख्या, बाढी आइपुग्ने ठ्याक्कै समय, एउटा घरको क्षति)।\n"
            "हरेक सङ्ख्या यही विश्लेषणबाट आउँछ; बाहिरी जानकारी प्रयोग गरिँदैन।"
        )
    return (
        "I answer questions about this analysis only, from its own results:\n"
        "- **Where did the flood hit?** Mapped flood and debris area, and where the radar could not see.\n"
        "- **What was damaged?** Buildings, roads and bridges in the mapped footprint (from pre-event OpenStreetMap).\n"
        "- **Who is cut off?** Settlements with no road out or no road to a hospital, longer routes, and which repair would restore the most access.\n"
        "- A named settlement, bridge or health facility, or the nearest one to a coordinate.\n"
        "- A short situation report in English or Nepali.\n"
        "- What this analysis cannot tell you (people, exact flood arrival times, damage to a single house).\n"
        "Every number comes from this analysis; I do not use outside information."
    )


def _rivers(question: str, lang: str) -> str | None:
    q = question.lower()
    if not (
        any(w in q for w in ("river", "stream", "नदी", "खोला"))
        and any(w in q for w in ("how many", "number of", "count", "list", "कति", "सूची"))
    ):
        return None
    if lang.lower().startswith("ne"):
        return (
            "यो विश्लेषणले नदी गन्दैन। उचाइ मोडेलबाट निकालिएको जलनिकास सञ्जाल पानी कुन तल्लो जमिनसम्म पुग्न सक्छ "
            "र बाढी/लेदो मुख्य नदीसँग जोडिएको छ कि छैन भनेर जाँच्न मात्र प्रयोग हुन्छ। नदीका च्यानल नक्साको उपग्रह तस्बिरमा देखिन्छन्।"
        )
    return (
        "Not counted by this analysis. It uses the drainage network derived from the Copernicus "
        "elevation model only to judge which low ground water can reach and whether mapped flood or "
        "debris connects to the main river; it does not list or count rivers. The river channels are "
        "visible in the map's satellite imagery."
    )


_LEVEL_NE = {"SEVERE": "गम्भीर", "HIGH": "उच्च", "MODERATE": "मध्यम", "LOW": "न्यून"}


def _sitrep_brief(question: str, ctx: dict[str, Any], lang: str) -> str | None:
    """A short situation report from the job's facts only (template, no language model): the
    three rescue questions, severity and limits. Every number is a fact's own text."""
    ql = question.lower()
    if not any(w in ql for w in _SITREP_WORDS):
        return None
    k = ctx.get("kpis", {})

    def t(key: str) -> str:
        return str(k.get(key, {}).get("text") or "—")

    ne = lang.lower().startswith("ne")
    where = t("target_location")
    if ne:
        lines = [
            f"**स्थिति प्रतिवेदन — {where}** (उपग्रह तथ्याङ्कबाट अनुमान; स्थलगत पुष्टि भएको छैन)",
            f"- **गम्भीरता:** {_LEVEL_NE.get(t('verdict_level'), t('verdict_level'))}। {t('verdict_headline_ne') if 'verdict_headline_ne' in k else t('verdict_headline')}",
            f"- **बाढी कहाँ पुग्यो:** {t('aoi_area_km2')} वर्ग कि.मि. मध्ये करिब {t('area_debris_km2')} वर्ग कि.मि. लेदो तथा कटान र {t('area_flood_km2')} वर्ग कि.मि. नयाँ पानी। राडारले नदेखेको {t('area_not_observable_km2')} वर्ग कि.मि. सुरक्षित भन्न सकिँदैन।",
            f"- **के क्षति भयो:** {t('buildings_hit')} भवन प्रभावित हुन सक्ने; {t('network_failed_km')} कि.मि. सडक अवरुद्ध हुन सक्ने; {t('bridges_lost')} पुल सम्भवतः बगेका।",
            f"- **को सम्पर्कविहीन छ:** {t('settlements_cut_off')} बस्तीको सडक सम्पर्क टुटेको हुन सक्ने; थप {t('settlements_hospital_cut_off')} बस्तीबाट अस्पताल पुग्न नसकिने।",
            "- **सीमा:** तस्बिर केही दिनको फरकमा लिइन्छन्; राडारले नदेखेको क्षेत्र अज्ञात हो; कदम चाल्नु अघि स्थलगत पुष्टि गर्नुहोस्।",
        ]
    else:
        lines = [
            f"**Situation report — {where}** (estimates from satellite data; not field-verified)",
            f"- **Severity:** {t('verdict_level')}. {t('verdict_headline')}",
            f"- **Where did the flood hit?** About {t('area_debris_km2')} km² of debris and scour and {t('area_flood_km2')} km² of new water in {t('aoi_area_km2')} km². {t('area_not_observable_km2')} km² was hidden from the radar: unknown, not safe.",
            f"- **What was damaged?** {t('buildings_hit')} mapped buildings likely hit; {t('network_failed_km')} km of road and trail likely impassable; {t('bridges_lost')} bridges likely lost.",
            f"- **Who is cut off?** {t('settlements_cut_off')} communities likely have no road out; {t('settlements_hospital_cut_off')} more likely cannot reach a hospital by road.",
            "- **Limits:** images are days apart; ground the radar could not see is unknown; confirm on the ground before acting.",
        ]
    return "\n".join(lines)


def _operational_limit(question: str, ctx: dict[str, Any], lang: str) -> str | None:
    """Keep rescue and vehicle-routing recommendations out of a road-access analysis."""
    q = question.lower()
    terms = (
        "ambulance",
        "helicopter",
        "aircraft",
        "evacuat",
        "rescue",
        "emergency vehicle",
        "safe route",
        "which road should",
    )
    if not any(term in q for term in terms):
        return None
    k = ctx.get("kpis", {})
    cut = str(k.get("settlements_cut_off", {}).get("text") or "unknown")
    hospital = str(k.get("settlements_hospital_cut_off", {}).get("text") or "unknown")
    if lang.lower().startswith("ne"):
        return (
            "### उद्धार तथा सवारी मार्ग\n\n"
            f"यस विश्लेषणमा सडक बाहिर जाने बाटो नभएका बस्ती: **{cut}**; अस्पतालसम्म सडक पहुँच नभएका बस्ती: **{hospital}**। "
            "यसले एम्बुलेन्सको लागि सडक सुरक्षित वा चल्न मिल्ने छ कि छैन, हेलिकप्टर पहुँच, वा उद्धार आवश्यक छ कि छैन जाँच्दैन। "
            "यसबाट सुरक्षित एम्बुलेन्स मार्ग वा हेलिकप्टर प्राथमिकता निर्धारण गर्न सकिँदैन; स्थानीय टोलीसँग पुष्टि गर्नुहोस्।"
        )
    return (
        "### Rescue and vehicle routing\n\n"
        f"This analysis records **{cut}** settlements with no road out and **{hospital}** without mapped road access to a hospital. "
        "It does not verify road passability for ambulances, aircraft access, or whether evacuation is needed. "
        "It cannot identify a safe ambulance route or helicopter priority; confirm those with local responders."
    )


def _observability_answer(question: str, ctx: dict[str, Any], lang: str) -> str | None:
    q = question.lower().replace("’", "'")
    if not any(
        w in q
        for w in (
            "shadow",
            "not observable",
            "unobservable",
            "unobserved",
            "hidden by radar",
            "cloud cover",
            "couldn't be observed",
            "could not be observed",
            "not observed",
        )
    ):
        return None
    value = str(ctx.get("kpis", {}).get("area_not_observable_km2", {}).get("text") or "unknown")
    if lang.lower().startswith("ne"):
        return (
            f"यस विश्लेषणमा **{value} वर्ग कि.मी.** क्षेत्रमा कुनै मान्य उपग्रह अवलोकन उपलब्ध थिएन। "
            "सम्भावित कारण Sentinel‑1 राडारको छायाँ/लेआभर वा Sentinel‑2 मा बादल, बादलको छायाँ वा डेटा नहुनु हो। "
            "यो नदेखिएको क्षेत्रको संयुक्त नक्सा हो; कारणअनुसार छुट्टिएको क्षेत्रफल वा स्थानहरूको सूची उपलब्ध छैन। "
            "नक्साको ‘Not observable’ तहमा क्षेत्र हेर्नुहोस्। त्यहाँको क्षति अज्ञात छ, त्यसैले क्षतिरहित नमान्नुहोस्।"
        )
    return (
        f"**{value} km²** had no valid satellite observation in this analysis. Likely causes are "
        "Sentinel‑1 radar layover/shadow or cloud, cloud shadow, or missing data in Sentinel‑2. "
        "The result combines unobserved areas; it does not report area or locations by cause. "
        "Use the map’s **Not observable** layer to see where. Damage there is unknown, not absent: "
        "we cannot say whether anything there was hit."
    )


def _answer_cards(question: str, ctx: dict[str, Any], lang: str) -> list[dict[str, str]]:
    """Build display cards from the verified facts ledger, never from model-generated content."""
    q = question.lower()
    is_ne = lang.lower().startswith("ne")

    def has(*terms: str) -> bool:
        return any(term in q for term in terms)

    summary = has("summary", "briefing", "situation", "overview", "स्थिति", "सारांश")
    groups: list[tuple[str, str, str, str, str]] = []
    if summary:
        groups.extend(
            [
                (
                    "area_flood_km2",
                    "Mapped flood area",
                    "नक्साङ्कित बाढी क्षेत्र",
                    "km²",
                    "Mapped from available satellite observations",
                ),
                (
                    "area_debris_km2",
                    "Mapped debris area",
                    "नक्साङ्कित गेग्रान क्षेत्र",
                    "km²",
                    "Mapped from available satellite observations",
                ),
                (
                    "buildings_hit",
                    "Buildings in mapped hazard area",
                    "जोखिम क्षेत्रमा रहेका भवन",
                    "buildings",
                    "Building footprints overlapping the mapped hazard",
                ),
                (
                    "settlements_cut_off",
                    "Settlements with no road out",
                    "बाहिर जाने सडक नभएका बस्ती",
                    "settlements",
                    "No mapped road out",
                ),
                (
                    "bridges_at_risk",
                    "Bridges at risk",
                    "जोखिममा रहेका पुल",
                    "bridges",
                    "Heuristic assessment; not a confirmed inspection",
                ),
                (
                    "hospitals_isolated",
                    "Facilities without mapped road access",
                    "सडक पहुँच नभएका स्वास्थ्य संस्था",
                    "facilities",
                    "No mapped road access in this analysis",
                ),
            ]
        )
    if not summary and has("settlement", "village", "बस्ती", "गाउँ"):
        groups.extend(
            [
                (
                    "settlements_cut_off",
                    "Settlements with no road out",
                    "बाहिर जाने सडक नभएका बस्ती",
                    "settlements",
                    "No mapped road out",
                ),
                (
                    "settlements_hospital_cut_off",
                    "Settlements without hospital road access",
                    "अस्पताल सडक पहुँच नभएका बस्ती",
                    "settlements",
                    "No mapped road route to a hospital",
                ),
                (
                    "settlements_detour",
                    "Settlements with longer routes",
                    "लामो बाटो चाहिने बस्ती",
                    "settlements",
                    "Modeled route is longer than before",
                ),
                (
                    "settlements_total",
                    "Mapped settlements",
                    "नक्साङ्कित बस्ती",
                    "settlements",
                    "Settlements included in this analysis",
                ),
            ]
        )
    if not summary and has("bridge", "crossing", "पुल"):
        groups.extend(
            [
                (
                    "bridges_lost",
                    "Bridges likely lost",
                    "हराएको सम्भावित पुल",
                    "bridges",
                    "Heuristic assessment; not a confirmed inspection",
                ),
                (
                    "bridges_at_risk",
                    "Bridges at risk",
                    "जोखिममा रहेका पुल",
                    "bridges",
                    "Heuristic assessment; not a confirmed inspection",
                ),
                (
                    "bridges_total",
                    "Mapped bridges",
                    "नक्साङ्कित पुल",
                    "bridges",
                    "Bridges included in this analysis",
                ),
            ]
        )
    if not summary and has("hospital", "health facilit", "clinic", "medical", "अस्पताल", "स्वास्थ्य"):
        groups.extend(
            [
                (
                    "hospitals_isolated",
                    "Facilities without mapped road access",
                    "सडक पहुँच नभएका स्वास्थ्य संस्था",
                    "facilities",
                    "No mapped road access in this analysis",
                ),
                (
                    "hospitals_accessible",
                    "Facilities with mapped road access",
                    "सडक पहुँच भएका स्वास्थ्य संस्था",
                    "facilities",
                    "Mapped network access; passability is not verified",
                ),
                (
                    "hospitals_total",
                    "Mapped health facilities",
                    "नक्साङ्कित स्वास्थ्य संस्था",
                    "facilities",
                    "Facilities included in this analysis",
                ),
            ]
        )
    if not summary and has("building", "structure", "home", "house", "exposure", "बिल्डिङ", "घर"):
        groups.append(
            (
                "buildings_hit",
                "Buildings in mapped hazard area",
                "जोखिम क्षेत्रमा रहेका भवन",
                "buildings",
                "Building footprints overlapping the mapped hazard",
            )
        )
    if not summary and has("flood", "debris", "hazard", "inundat", "बाढी", "गेग्रान"):
        groups.extend(
            [
                (
                    "area_flood_km2",
                    "Mapped flood area",
                    "नक्साङ्कित बाढी क्षेत्र",
                    "km²",
                    "Mapped from available satellite observations",
                ),
                (
                    "area_debris_km2",
                    "Mapped debris area",
                    "नक्साङ्कित गेग्रान क्षेत्र",
                    "km²",
                    "Mapped from available satellite observations",
                ),
            ]
        )
    if not summary and has("shadow", "observable", "unobserved", "cloud", "दृश्य", "नदेखिएको"):
        groups.extend(
            [
                (
                    "area_not_observable_km2",
                    "Area not observable",
                    "अवलोकन गर्न नसकिएको क्षेत्र",
                    "km²",
                    "Damage status in this area is unknown",
                ),
                (
                    "observable_pct",
                    "Area observable",
                    "अवलोकन गर्न सकिएको क्षेत्र",
                    "%",
                    "Share of the area with usable observations",
                ),
            ]
        )
    if not summary and has("road", "route", "ambulance", "evacuat", "rescue", "सडक", "उद्धार"):
        groups.append(
            (
                "network_failed_km",
                "Road network affected",
                "प्रभावित सडक सञ्जाल",
                "km",
                "Mapped network impact; vehicle passability is not verified",
            )
        )

    kpis = ctx.get("kpis", {})
    cards: list[dict[str, str]] = []
    seen: set[str] = set()
    for key, title, title_ne, unit, detail in groups:
        if key in seen:
            continue
        seen.add(key)
        raw = kpis.get(key, {}).get("text")
        if raw is None or not str(raw).strip() or str(raw).strip().lower() == "unknown":
            continue
        cards.append(
            {
                "key": key,
                "title": title_ne if is_ne else title,
                "value": str(raw),
                "unit": unit,
                "detail": detail,
                "source": "Analysis results" if not is_ne else "विश्लेषण नतिजा",
                "kind": "metric",
            }
        )

    if summary:
        unobserved = kpis.get("area_not_observable_km2", {}).get("text")
        try:
            has_unobserved_area = float(str(unobserved).replace(",", "")) > 0
        except (TypeError, ValueError):
            has_unobserved_area = False
        if has_unobserved_area:
            cards.append(
                {
                    "key": "area_not_observable_km2",
                    "title": "Area not observable" if not is_ne else "अवलोकन गर्न नसकिएको क्षेत्र",
                    "value": str(unobserved),
                    "unit": "km²",
                    "detail": "Damage status in this area is unknown",
                    "source": "Analysis results" if not is_ne else "विश्लेषण नतिजा",
                    "kind": "metric",
                }
            )

    if has(
        "ambulance",
        "helicopter",
        "aircraft",
        "evacuat",
        "rescue",
        "safe route",
        "which road should",
    ):
        cards.append(
            {
                "key": "operational_limit",
                "title": "Operational limit" if not is_ne else "विश्लेषणको सीमा",
                "value": "Not assessed" if not is_ne else "मापन गरिएको छैन",
                "unit": "",
                "detail": (
                    "This analysis does not verify vehicle passability, safe routes, aircraft access, or evacuation need."
                    if not is_ne
                    else "यस विश्लेषणले सवारीको बाटो चल्ने अवस्था, सुरक्षित मार्ग, हवाई पहुँच वा उद्धार आवश्यकता जाँच्दैन।"
                ),
                "source": "Scope of this analysis" if not is_ne else "विश्लेषणको दायरा",
                "kind": "limit",
            }
        )
    return cards


# The assistant panel's suggested questions, owned by the server: id, English and Nepali wording,
# and the template prompt that answers it from the job's facts (never a language model).
SUGGESTIONS: tuple[tuple[str, str, str, str], ...] = (
    ("briefing", "Could you give me an executive situation briefing for disaster commanders?",
     "विपद् व्यवस्थापन कमाण्डरका लागि स्थिति ब्रिफिङ दिनुहोस्।",
     "provide an executive situation briefing for disaster commanders."),
    ("cut_off", "Which settlements are completely cut off from road access?",
     "कुन बस्तीहरूको सडक सम्पर्क पूर्ण रूपमा टुटेको छ?",
     "which settlements are completely cut off without road access?"),
    ("bridges", "Which bridges look likely lost or at risk at river crossings?",
     "नदी क्रसिङका कुन पुल बगेका वा जोखिममा छन्?",
     "which bridges are likely lost or at risk across river crossings?"),
    ("health", "Which health facilities can no longer be reached by road?",
     "कुन स्वास्थ्य संस्थामा अब सडकबाट पुग्न सकिँदैन?",
     "which health facilities cannot be reached by road?"),
    ("repair", "What's the top road or bridge repair priority to reconnect communities?",
     "बस्तीहरू जोड्न सबैभन्दा पहिले कुन सडक वा पुल मर्मत गर्ने?",
     "what is the top priority road or bridge repair to reconnect communities?"),
    ("detours", "Which hospital evacuation detours are longest because of road damage?",
     "सडक क्षतिका कारण अस्पताल पुग्ने कुन बाटो सबैभन्दा लामो भएको छ?",
     "what are the longest hospital evacuation detours caused by road severances?"),
    ("exposure", "How many homes and buildings are exposed in the high-hazard zone?",
     "उच्च जोखिम क्षेत्रमा कति घर र भवन पर्छन्?",
     "how many homes and structures are exposed in the high-hazard zone?"),
    ("unobserved", "Which areas couldn’t be observed because of cloud cover or radar shadow?",
     "बादल वा राडारको छायाँका कारण कुन क्षेत्र देखिएन?",
     "which areas could not be observed due to cloud cover or radar shadow?"),
)  # fmt: skip


def _norm(q: str) -> str:
    q = q.lower().replace("’", "'").replace("couldn't", "could not").replace("what's", "what is")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", q)).strip()


_CANNED_BY_NORM: dict[str, str] = {_norm(c): c for c in EXISTING_GENERATED_PROMPTS} | {
    _norm(t): c for _, en, ne, c in SUGGESTIONS for t in (en, ne)
}


def _closest_canned(question: str) -> str | None:
    """The template prompt for a suggested question or a near-identical wording of one (case,
    punctuation and small edits ignored); other questions go through the general assistant."""
    import difflib

    q = _norm(question)
    if q in _CANNED_BY_NORM:
        return _CANNED_BY_NORM[q]
    close = difflib.get_close_matches(q, list(_CANNED_BY_NORM), n=1, cutoff=0.88)
    return _CANNED_BY_NORM[close[0]] if close else None


def _special_answer(sid: str, job_dir: Path, ctx: dict[str, Any], lang: str) -> str | None:
    if sid == "detours":
        return _detours_answer(job_dir, ctx, lang)
    if sid == "exposure":
        return _exposure_answer(ctx, lang)
    return None


def _report_tables(job_dir: Path) -> dict[str, list[dict]]:
    with contextlib.suppress(Exception):
        return json.loads((job_dir / "sitrep_facts.json").read_text())["payload"]["tables"]
    return {}


def _detours_answer(job_dir: Path, ctx: dict[str, Any], lang: str) -> str:
    k = ctx.get("kpis", {})

    def t(key: str) -> str:
        return str(k.get(key, {}).get("text") or "—")

    rows = sorted(
        _report_tables(job_dir).get("detours", []),
        key=lambda r: float(str(r.get("delay", "0")).replace(",", "") or 0),
        reverse=True,
    )[:5]
    if lang.lower().startswith("ne"):
        head = (
            f"### लामो बाटो\n\n{t('settlements_detour')} बस्ती अब लामो बाटोबाट मात्र बाहिर जान सक्छन्; "
            f"सबैभन्दा बढी थप समय {t('max_detour_hours')} घण्टा, औसत {t('avg_detour_hours')} घण्टा।"
        )
        body = [
            f"- {r.get('name') or 'नाम नभएको'}: {r.get('h_pre')} → {r.get('h_post')} घण्टा (+{r.get('delay')}), {r.get('buildings')} भवन"
            for r in rows
        ]
    else:
        head = (
            f"### Longest detours\n\n{t('settlements_detour')} settlements can now only get out by a longer "
            f"route; the longest adds {t('max_detour_hours')} h (average {t('avg_detour_hours')} h)."
        )
        body = [
            f"- {r.get('name') or 'unnamed'}: {r.get('h_pre')} → {r.get('h_post')} h (+{r.get('delay')} h), {r.get('buildings')} buildings"
            for r in rows
        ]
    tail = (
        "\n\nबाटोको समय सडक वर्गको अनुमानित गतिबाट निकालिएको हो।"
        if lang.lower().startswith("ne")
        else "\n\nTravel times use assumed speeds per road class; they rank detours, not exact times."
    )
    return head + ("\n" + "\n".join(body) if body else "") + tail


def _exposure_answer(ctx: dict[str, Any], lang: str) -> str:
    k = ctx.get("kpis", {})

    def t(key: str) -> str:
        return str(k.get(key, {}).get("text") or "—")

    if lang.lower().startswith("ne"):
        return (
            f"### जोखिममा परेका भवन\n\n- नक्साङ्कित बाढी/लेदो क्षेत्रमा: **{t('buildings_hit')}** "
            f"(जम्मा {t('buildings_total')} मध्ये)\n- उच्च विश्वास: {t('buildings_hit_high')}, मध्यम: "
            f"{t('buildings_hit_moderate')}, कम: {t('buildings_hit_low')}\n- सम्पर्कविहीन बस्तीको "
            f"{t('exposure_radius_m')} मि. भित्र: {t('buildings_in_cut_off')}\n- राडारले नदेखेको जमिनमा: "
            f"{t('buildings_unobserved')}\n\nभवन गनिन्छन्, मानिस होइन: जनसङ्ख्या तथ्याङ्क प्रयोग गरिँदैन।"
        )
    return (
        f"### Buildings exposed\n\n- In the mapped flood and debris footprint: **{t('buildings_hit')}** "
        f"of {t('buildings_total')} mapped\n- By confidence: high {t('buildings_hit_high')}, moderate "
        f"{t('buildings_hit_moderate')}, low {t('buildings_hit_low')}\n- Within "
        f"{t('exposure_radius_m')} m of cut-off communities: {t('buildings_in_cut_off')}\n- On ground "
        f"the radar could not see: {t('buildings_unobserved')}\n\nBuildings are counted, not people: "
        "no population data is used, and a building in the footprint is not confirmed damage."
    )


def suggested_answers(job_dir: Path, lang: str = "en") -> list[dict[str, Any]]:
    """Every suggested question with its answer, from the job's facts (template, no model), so the
    panel can show them instantly."""
    ctx = load_job_context(job_dir)
    ne = lang.lower().startswith("ne")
    out = []
    for sid, en, ne_text, canned in SUGGESTIONS:
        out.append(
            {
                "id": sid,
                "question": ne_text if ne else en,
                "answer": _special_answer(sid, job_dir, ctx, lang)
                or _generate_grounded_fallback(ctx, canned, lang=lang),
                "cards": _answer_cards(canned, ctx, lang),
                "provider": "dhara_verified_ledger",
            }
        )
    return out


def _is_multipart_question(question: str) -> bool:
    """Avoid returning a one-topic limitation when the user asks multiple questions at once."""
    # Count questions, not sentences: "When will it arrive? Return a time only." is one question
    # plus an instruction about the answer's form.
    sentences = [x.strip() for x in re.split(r"(?<=[?.;!])\s+|\n+", question) if x.strip()]
    asks = [
        x
        for x in sentences
        if x.endswith("?")
        or re.match(r"(what|who|where|when|why|how|which|is|are|did|does|can|will)\b", x, re.I)
    ]
    return len(asks) > 1 or bool(
        re.search(
            r"\b(?:and|also)\s+(?:what|who|where|when|why|how|is|did|does|can)\b",
            question,
            re.I,
        )
    )


def _chat_cache_key(
    job_id: str,
    question: str,
    lang: str,
    selected_entity: dict[str, Any] | None,
) -> tuple[str, str, str, tuple[Any, ...] | None]:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()
    entity_key = None
    if selected_entity:
        coords = selected_entity.get("coordinates")
        rounded = ()
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            try:
                rounded = (round(float(coords[0]), 5), round(float(coords[1]), 5))
            except (TypeError, ValueError):
                rounded = ()
        entity_key = (
            str(selected_entity.get("kind") or ""),
            str(selected_entity.get("id") or ""),
            *rounded,
        )
    return job_id, normalized, lang.lower(), entity_key


def _disk_cache_path(job_dir: Path) -> Path:
    return job_dir / "cache" / "assistant_answers.json"


def _disk_key(key: tuple[str, str, str, tuple[Any, ...] | None]) -> str:
    return json.dumps([_CHAT_CACHE_VERSION, *key[1:]], ensure_ascii=False, default=str)


def _get_cached_chat_answer(
    key: tuple[str, str, str, tuple[Any, ...] | None], job_dir: Path
) -> tuple[str, str] | None:
    with _CHAT_CACHE_LOCK:
        cached = _CHAT_RESPONSE_CACHE.get(key)
        if cached is not None:
            _CHAT_RESPONSE_CACHE.move_to_end(key)
            return cached[1], cached[2]
        with contextlib.suppress(Exception):
            hit = json.loads(_disk_cache_path(job_dir).read_text(encoding="utf-8")).get(
                _disk_key(key)
            )
            if hit:
                _CHAT_RESPONSE_CACHE[key] = (time.monotonic(), hit["answer"], hit["provider"])
                return hit["answer"], hit["provider"]
    return None


def _cache_chat_answer(
    key: tuple[str, str, str, tuple[Any, ...] | None], answer: str, provider: str, job_dir: Path
) -> None:
    """Keep a verified answer (memory and the job's disk cache). Failures are never kept, so the
    next ask tries again."""
    if provider == "assistant_unavailable":
        return
    with _CHAT_CACHE_LOCK:
        _CHAT_RESPONSE_CACHE[key] = (time.monotonic(), answer, provider)
        _CHAT_RESPONSE_CACHE.move_to_end(key)
        while len(_CHAT_RESPONSE_CACHE) > _CHAT_CACHE_MAX:
            _CHAT_RESPONSE_CACHE.popitem(last=False)
        with contextlib.suppress(Exception):
            path = _disk_cache_path(job_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            data[_disk_key(key)] = {"answer": answer, "provider": provider}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)


def ask_assistant(
    job_dir: Path,
    question: str,
    lang: str = "en",
    selected_entity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer a question about one job, in English or Nepali, using only that job's results.

    Order: off-topic refusal; a named settlement, bridge or health facility (answered from its
    own attributes); a suggested question (template over the facts ledger); the language model
    when one is configured, kept only if every number in it is in the results; else templates.
    """
    from dhara.assistant import entities

    cache_key = _chat_cache_key(job_dir.name, question, lang, selected_entity)
    ctx = load_job_context(job_dir)
    kpis = {k: v.get("text") for k, v in ctx.get("kpis", {}).items()}

    def done(answer: str, provider: str, keep: bool = True) -> dict[str, Any]:
        if keep:
            _cache_chat_answer(cache_key, answer, provider, job_dir)
        _write_conversation_log(
            job_dir, question, lang, [{"role": "user", "content": question}], answer, provider
        )
        return {
            "job_id": job_dir.name,
            "question": question,
            "lang": lang,
            "provider": provider,
            "answer": answer,
            "kpis": kpis,
            "cards": (
                _answer_cards(question, ctx, lang)
                if provider == "grounded_agent" and answer != OFF_TOPIC_FALLBACK
                else []
            ),
        }

    cached = _get_cached_chat_answer(cache_key, job_dir)
    if cached:
        return done(*cached)

    if _is_obviously_off_topic(question):
        return done(OFF_TOPIC_FALLBACK, "grounded_rules")

    caps = _capabilities(question, lang) or _rivers(question, lang)
    if caps:
        return done(caps, "dhara_verified_ledger")
    brief = _sitrep_brief(question, ctx, lang)
    if brief:
        return done(brief, "dhara_verified_ledger")
    fixed = _not_measured(question, ctx, lang)
    if fixed:
        return done(fixed, "dhara_verified_ledger")
    observability = _observability_answer(question, ctx, lang)
    if observability:
        return done(observability, "dhara_verified_ledger")
    operational_limit = _operational_limit(question, ctx, lang)
    if operational_limit:
        return done(operational_limit, "dhara_verified_ledger")

    ql = question.lower()
    if any(w in ql for w in ("how many", "count", "number of", "कति")) and any(
        w in ql for w in ("settlement", "village", "बस्ती")
    ):
        k = ctx.get("kpis", {})

        def t(key: str) -> str:
            return str(k.get(key, {}).get("text") or "unknown")

        ne = lang.lower().startswith("ne")
        rows = (
            (t("settlements_cut_off"), "बाहिर जाने सडक नभएका बस्ती", "settlements with no road out"),
            (
                t("settlements_hospital_cut_off"),
                "अस्पतालसम्म सडक पहुँच नभएका बस्ती",
                "settlements without road access to a hospital",
            ),
            (
                t("settlements_detour"),
                "लामो बाटो चाहिने बस्ती",
                "settlements that need a longer route",
            ),
        )
        body = "\n".join(f"- **{n}** {ne_l if ne else en}" for n, ne_l, en in rows)
        return done(
            f"{body}\n\n{'जम्मा' if ne else 'Out of'} {t('settlements_total')} {'बस्ती' if ne else 'mapped settlements'}.",
            "dhara_verified_ledger",
        )

    if any(
        w in question.lower() for w in ("people", "citizen", "population", "resident", "household")
    ):
        return done(_generate_grounded_fallback(ctx, question, lang=lang), "dhara_verified_ledger")

    feature = entities.match(job_dir, question, selected_entity)
    if not feature:
        ambiguous = entities.ambiguous_exact_matches(job_dir, question)
        if ambiguous:
            options = []
            for item in ambiguous:
                geometry = item.props.get("_geometry") or {}
                coordinates = geometry.get("coordinates") or []
                location = (
                    f"{float(coordinates[1]):.5f}°N, {float(coordinates[0]):.5f}°E"
                    if geometry.get("type") == "Point" and len(coordinates) >= 2
                    else "location unavailable"
                )
                osm_id = str(item.props.get("osm_id") or "")
                options.append(
                    f"- {item.name} — {location}" + (f" (OSM {osm_id})" if osm_id else "")
                )
            return done(
                "I found multiple mapped features with that name. Select the intended feature on the map "
                "or include its coordinates:\n" + "\n".join(options),
                "dhara_verified_ledger",
            )
    canned = _closest_canned(question)
    if canned and not feature:
        sid = next((i for i, _, _, c in SUGGESTIONS if c == canned), "")
        answer = _special_answer(sid, job_dir, ctx, lang) or _generate_grounded_fallback(
            ctx, canned, lang=lang
        )
        return done(answer, "dhara_verified_ledger")

    cfg = get_settings()
    provider = cfg.llm_provider
    answer = ""
    system_prompt = (
        build_system_prompt(ctx, lang=lang)
        + "\n\nDo not disclose backend AI provider or model names. If asked, explain that this "
        "interface does not display those names."
    )
    feature_record = entities.answer(job_dir, feature, "en") if feature else None
    selected_coordinates = (selected_entity or {}).get("coordinates")
    if (
        feature_record
        and isinstance(selected_coordinates, (list, tuple))
        and len(selected_coordinates) == 2
    ):
        try:
            lon, lat = float(selected_coordinates[0]), float(selected_coordinates[1])
            feature_record += (
                f"\nSelected map location (EPSG:4326): latitude {lat:.6f}, longitude {lon:.6f}. "
                "Use these coordinates as the origin for spatial_nearest_features when the user asks "
                "about nearby or nearest mapped features. Distances from that tool are straight-line."
            )
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid selected map coordinates in assistant context")
    if feature_record:
        system_prompt += (
            "\n\nVERIFIED RECORD FOR THE SELECTED OR NAMED MAP FEATURE:\n"
            + feature_record
            + "\nInterpret references to 'this', 'that', or 'it' as referring to this feature. "
            "Answer the user's actual question from this record; do not answer as the assistant."
        )
    model_error: Exception | None = None
    try:
        groq_keys = [key.get_secret_value() for key in cfg.groq_api_keys]
        gemini_keys = [key.get_secret_value() for key in cfg.gemini_api_keys]
        if provider == "gemini" and gemini_keys:
            models = [cfg.gemini_model] + [
                m.strip() for m in cfg.gemini_fallback_models.split(",") if m.strip()
            ]
            gemini_error: Exception | None = None
            now = time.monotonic()
            live = [m for m in models if _model_down_until.get(m, 0.0) <= now] or models[-1:]
            for gm in live:  # every account on the preferred model, then the next model
                try:
                    answer, _ = _call_groq_with_account_fallback(
                        gemini_keys,
                        job_dir,
                        question,
                        lang,
                        gm,
                        feature_record,
                        endpoint=GEMINI_OPENAI_ENDPOINT,
                    )
                    gemini_error = None
                    break
                except Exception as exc:
                    gemini_error = exc
                    _model_down_until[gm] = time.monotonic() + MODEL_COOLDOWN_S
                    logger.warning("Gemini model %s unavailable on all accounts", gm)
            if gemini_error is not None:
                if not groq_keys:
                    raise gemini_error
                logger.warning("All Gemini accounts failed; trying Groq")
                answer, _ = _call_groq_with_account_fallback(
                    groq_keys, job_dir, question, lang, cfg.groq_model, feature_record
                )
        elif provider == "groq" and groq_keys:
            answer, _ = _call_groq_with_account_fallback(
                groq_keys, job_dir, question, lang, cfg.groq_model, feature_record
            )
        elif provider == "openrouter" and cfg.openrouter_api_key:
            answer, _ = _call_tool_agent(
                cfg.openrouter_api_key.get_secret_value(),
                job_dir,
                question,
                lang=lang,
                model=cfg.openrouter_model,
                endpoint="https://openrouter.ai/api/v1/chat/completions",
                max_tokens=None,
                selected_feature_record=feature_record,
            )
        elif provider == "gemini" and cfg.gemini_api_key:
            answer = _call_gemini_api(
                cfg.gemini_api_key.get_secret_value(), system_prompt, question, cfg.gemini_model
            )
        elif provider == "openai" and cfg.openai_api_key:
            answer = _call_openai_api(
                cfg.openai_api_key.get_secret_value(), system_prompt, question, cfg.openai_model
            )
    except Exception as exc:
        model_error = exc
        logger.exception("Assistant model request failed")
    if answer.strip() in (
        "No response generated by model.",
        "GIS Agent exceeded maximum reasoning iterations.",
        "GIS Agent exceeded its response budget.",
    ):
        answer = ""  # an empty or failed model turn falls back to the verified answer below
    if answer and answer != OFF_TOPIC_FALLBACK:
        answer = _label_bare_number(answer, ctx, lang)
        bad = _unbacked_numbers(answer, _backing_text(job_dir, ctx))
        if not bad:
            return done(answer, "grounded_agent")
        logger.warning("Discarded assistant answer with unsupported numeric claims: %s", bad)
        answer = ""  # a number the results do not contain: fall back to the verified answer
    elif answer == OFF_TOPIC_FALLBACK and not feature:
        return done(answer, "grounded_rules")

    if feature:
        return done(entities.answer(job_dir, feature, lang), "dhara_verified_ledger")
    if model_error is not None:
        return done(MODEL_UNAVAILABLE_FALLBACK, "assistant_unavailable")
    fallback = (
        "उपलब्ध नतिजाबाट यस प्रश्नको प्रमाणमा आधारित उत्तर दिन सकिनँ। कृपया एक पटकमा एउटा प्रश्न सोध्नुहोस्।"
        if lang.lower().startswith("ne")
        else "I couldn't verify every detail needed for a safe answer. Try rephrasing, or ask for a specific result, its evidence, or a limitation."
    )
    return done(fallback, "grounded_rules", keep=False)


def explain_asset_impact(
    job_dir: Path, asset_type: str, asset_name: str, lang: str = "en"
) -> dict[str, Any]:
    """Generate an explainable AI briefing detailing why a bridge or road segment failed."""
    ctx = load_job_context(job_dir)

    target_bridge = None
    if asset_type == "bridge":
        for b in ctx.get("damaged_bridges", []):
            if asset_name.lower() in str(b["name"]).lower():
                target_bridge = b
                break

    target_settlement = None
    if asset_type == "settlement":
        for s in ctx.get("cut_off_settlements", []):
            if asset_name.lower() in str(s["name"]).lower():
                target_settlement = s
                break

    prompt = f"Explain the physical and network reasons why {asset_type} '{asset_name}' is classified with damage or cut-off status in {lang}."
    if target_bridge:
        prompt += (
            f" Details: status={target_bridge['status']}, heuristic loss score={target_bridge.get('p_lost')}, "
            f"radar drop={target_bridge.get('bright_drop_db')} dB."
        )
    elif target_settlement:
        prompt += (
            f" Details: status={target_settlement['status']}, P(cut)={target_settlement.get('p_cut_off')}, "
            f"buildings={target_settlement.get('buildings')}, hospital distance before={target_settlement.get('km_to_hospital')} km."
        )

    res = ask_assistant(job_dir, prompt, lang=lang)
    return {
        "asset_type": asset_type,
        "asset_name": asset_name,
        "explanation": res["answer"],
        "provider": res["provider"],
    }
