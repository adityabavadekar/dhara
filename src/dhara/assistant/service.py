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
import math
import re
from pathlib import Path
from typing import Any

from dhara.adapters.allowlist import guarded_request
from dhara.assistant.grounding import build_system_prompt, load_job_context
from dhara.config import get_settings

OFF_TOPIC_FALLBACK = "I can only answer questions about flood and debris damage, severed roads, cut-off villages, and hospital access for this disaster."

EXISTING_GENERATED_PROMPTS = frozenset(
    {
        "summary of isolated villages",
        "which bridges are likely lost?",
        "top repair priority",
        "explain unobservable area",
        "which settlements are completely cut off without road access?",
        "which bridges are destroyed or impassable across river crossings?",
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
            "description": "Retrieve verified summary disaster KPIs (cut-off villages, damaged bridges, buildings hit, debris and water km², not observable area, max detour hours).",
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
            "description": "Query surveyed settlements and villages in the AOI. Can filter by status ('cut_off', 'hospital_cut_off', 'detour', 'connected', or 'all') and minimum buildings in 500m.",
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
                        "description": "Only return settlements with at least this many buildings in 500m exposure radius",
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
            "description": "Query surveyed bridge structures and road river crossings. Returns status, radar backscatter drop (dB), failure probability, span length, and coordinates.",
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
            "description": "Query surveyed emergency hospitals, clinics, and health centers in the AOI with road network accessibility status.",
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
        return {"status": "success", "kpis": kpis}

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
                        "name": p.get("name") or "Unnamed Settlement",
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
            return {"status": "success", "count": len(results), "settlements": results[:limit]}
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
                        "name": p.get("name") or p.get("highway") or "Bridge",
                        "highway": p.get("highway"),
                        "status": st,
                        "p_lost": p.get("p_lost"),
                        "radar_drop_db": p.get("bright_drop_db"),
                        "span_length_m": p.get("length_m"),
                        "coordinates": coords,
                    }
                )
            results.sort(key=lambda b: float(b.get("p_lost") or 0), reverse=True)
            return {"status": "success", "count": len(results), "bridges": results[:limit]}
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
                        "name": p.get("name") or "Emergency Health Facility",
                        "amenity": p.get("amenity") or p.get("kind") or "hospital",
                        "status": st,
                        "is_isolated": bool(p.get("is_isolated")),
                        "coordinates": coords,
                    }
                )
            return {"status": "success", "count": len(results), "facilities": results[:limit]}
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
                        "name": r.get("name") or r.get("highway") or "Corridor",
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
                                    "name": p.get("name") or "Settlement",
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
                                    "name": p.get("name") or "Hospital",
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
                                    "name": p.get("name") or p.get("highway") or "Bridge",
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
        r"^(hi|hello|hey|test|ping|how are you|who are you)\b",
        r"\b(are\s+you\s+(an?\s+)?(ai|robot|bot|human|llm|chatgpt|model|real))\b",
        r"\b(what\s+are\s+you|who\s+(made|created|built)\s+you)\b",
    ]
    return any(re.search(pat, q_clean) for pat in patterns)


def _call_groq_agent(
    api_key: str,
    job_dir: Path,
    question: str,
    lang: str = "en",
    model: str = "openai/gpt-oss-120b",
) -> tuple[str, str]:
    """Execute autonomous GIS Agent on Groq with typed tool calling."""
    if _is_obviously_off_topic(question):
        return (OFF_TOPIC_FALLBACK, "grounded_agent")

    system_prompt = (
        f"You are the Dhara Disaster Response Assistant, an expert disaster response assistant.\n"
        f"Respond in language: {lang}.\n\n"
        f"OPERATIONAL MANDATES:\n"
        f"Assume your users are emergency decision-makers and relief coordinators who need direct, actionable answers.\n"
        f"Never use internal technical jargon like 'GIS algorithms', 'Dijkstra', 'graph mechanics', or 'backend pipeline'.\n"
        f"1. STRICT FACTUAL ACCURACY: Every statement you make must be verified by the query tools for this disaster area.\n"
        f"2. ALWAYS CALL QUERY TOOLS: Before answering queries regarding disaster extent, cut-off villages, damaged bridges, health facilities, or restoration priority, call the appropriate query tools (query_kpis, query_settlements, query_bridges, query_health_facilities, query_restoration_priorities, spatial_nearest_features) to retrieve facts.\n"
        f"3b. UNITS: every number must say what it counts (e.g. '220 buildings'), even if the user asks for a number only.\n"
        f"3a. PEOPLE: population is not an input. Building counts are buildings, never people, citizens or households. "
        f"If asked how many people are affected, say population is not known and give the building count as buildings.\n"
        f"3. NO HALLUCINATION: Never invent, extrapolate, or estimate figures. 'Not observable' means ground hidden from the radar only; for things this system does not measure (structural damage, arrival times, people, livestock, casualties) say they are not measured.\n"
        f"Questions about any named place, settlement, village, road, bridge or facility in this area are IN SCOPE: "
        f"look it up with the query tools (names may be in Nepali or another script, so search broadly).\n"
        f"4. DOMAIN BOUNDARY & 1-TOKEN MANDATORY REFUSAL SENTINEL:\n"
        f"If the user asks ANY question not directly related to this disaster event, flood/debris hazards, bridges, roads, settlements, or health facilities (such as general knowledge, coding, poetry, entertainment, politics, general world geography, jokes, chit-chat, or identity questions), you MUST refuse by outputting ONLY this single token:\n"
        f"REFUSE\n"
        f"Do NOT output any greeting, apology, or extra words. Output ONLY the single word REFUSE."
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = "https://api.groq.com/openai/v1/chat/completions"

    # Agent tool-calling loop (up to 5 iterations)
    for _ in range(5):
        payload = {
            "model": model,
            "messages": messages,
            "tools": GIS_TOOLS,
            "tool_choice": "auto",
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        resp = guarded_request("POST", url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API returned HTTP {resp.status_code}: {resp.text}")

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
                    job_dir, question, lang, messages, OFF_TOPIC_FALLBACK, "grounded_agent"
                )
                return (OFF_TOPIC_FALLBACK, "grounded_agent")
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

    final_answer = ("GIS Agent exceeded maximum reasoning iterations.", "grounded_agent")
    _write_conversation_log(job_dir, question, lang, messages, final_answer[0], final_answer[1])
    return final_answer


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
    resp = guarded_request("POST", url, json=payload, headers={"x-goog-api-key": api_key})
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
    resp = guarded_request("POST", url, json=payload, headers=headers)
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
        top_repair_text = (
            f"Asset '{r0.get('name') or r0.get('highway') or 'unnamed way'}' reconnects "
            f"{r0.get('settlements_reconnected', 'unknown')} settlements and {r0.get('buildings_reconnected', 'unknown')} buildings."
        )

    lang_lower = lang.lower()
    q_lower = question.lower()
    is_ne = lang_lower.startswith("ne") or "nepal" in lang_lower

    if any(
        w in q_lower
        for w in ("people", "citizen", "population", "resident", "household", "जनसंख्या", "मानिस")
    ):
        bic = get_kpi("buildings_in_cut_off")
        radius = get_kpi("exposure_radius_m")
        if is_ne:
            return (
                "### प्रभावित जनसंख्या\n\n"
                "जनसंख्या तथ्यांक यस विश्लेषणको इनपुट होइन, त्यसैले मानिसको संख्या थाहा छैन।\n\n"
                f"- सडक वा अस्पताल पहुँच गुमाएका बस्तीको {radius} मिटरभित्र **{bic} भवन** छन्।\n"
                f"- प्रकोप क्षेत्रमा **{buildings_hit} भवन** छन्।\n"
            )
        return (
            "### People at risk\n\n"
            "Population is not an input to this analysis, so the number of people is not known.\n\n"
            f"- **{bic} buildings** lie within {radius} m of settlements that lost road or hospital access.\n"
            f"- **{buildings_hit} buildings** lie in the mapped flood or debris zone.\n\n"
            "Combine these building counts with local census or household data to estimate people.\n"
        )

    if any(
        w in q_lower
        for w in ("cut off", "cut-off", "isolated", "village", "settlement", "सम्पर्कविहीन", "बस्ती")
    ):
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
                    f"- {s['name']}: {s['buildings']} {'भवन' if is_ne else 'buildings within 500 m'}"
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
        max_detour = get_kpi("max_detour_hours")
        hosp_cut = get_kpi("hospitals_isolated")
        if is_ne:
            return (
                f"### स्वास्थ्य चौकी तथा अस्पताल सम्पर्क विश्लेषण\n\n"
                f"- **सम्पर्कविहीन स्वास्थ्य संस्थाहरू:** {hosp_cut}\n"
                f"- **अधिकतम वैकल्पिक घुम्ती ढिलाइ:** +{max_detour} घण्टा\n"
                f"- **पहुँच अवस्था:** सडक अवरुद्ध भएकाले हेलिकप्टर वा पैदल मार्ग आवश्यक।\n"
            )
        return (
            f"### Healthcare Facility & Detour Assessment\n\n"
            f"- **Isolated Facilities:** {hosp_cut} medical posts/clinics severed from direct road network\n"
            f"- **Max Evacuation Detour:** +{max_detour} hours due to road and bridge severances\n"
            f"- **Transport Status:** Ground ambulance routes cut; secondary foot trail or rotary-wing transit required\n"
        )

    if any(w in q_lower for w in ("rescue", "helicopter", "relief", "उद्धार")):
        if is_ne:
            return (
                f"### आपतकालीन उद्धार तथा प्राथमिक क्षेत्र\n\n"
                f"- **सर्वोच्च प्राथमिकता बस्तीहरू:** {settlement_names}\n"
                f"- **सम्पर्कविहीन संरचनाहरू:** {buildings_hit} घर/भवनहरू\n"
                f"- **सिफारिस:** तत्काल हवाई राहत तथा स्थलगत टोली परिचालन।\n"
            )
        return (
            f"### Emergency Relief & Evacuation Priorities\n\n"
            f"- **Top Isolated Settlements:** {settlement_names}\n"
            f"- **Exposed Structures:** {buildings_hit} buildings isolated across severed corridors\n"
            f"- **Action Required:** Prioritize immediate rotary-wing supplies and emergency evacuation for highest-isolation hamlets\n"
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
            f"{bridges_lost} {'पुल बगेको सम्भावना, जम्मा' if is_ne else 'likely lost of'} {bridges_total}.",
            "",
        ]
        for b in rows[:10]:
            pl = b.get("p_lost")
            try:
                chance = f"{round(float(pl) * 100)}%"
            except (TypeError, ValueError):
                chance = "unknown"
            label = {"likely_lost": "likely lost", "at_risk": "at risk"}.get(
                b["status"], b["status"]
            )
            out.append(
                f"- **{b['name']}**: {label}, {'बगेको सम्भावना' if is_ne else 'chance lost'} {chance}"
            )
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
    """Numbers in a model answer that appear nowhere in the job's results or the question.

    Integers up to 20 are allowed for list numbering and counts of items the model enumerates.
    """
    ok = _numbers(backing)
    out = []
    for n in _numbers(answer):
        if n in ok or (n.isdigit() and int(n) <= 20):
            continue
        # 12.30 vs 12.3, 7 vs 7.0
        with contextlib.suppress(ValueError):
            if any(float(n) == float(o) for o in ok if o.replace(".", "", 1).isdigit()):
                continue
        out.append(n)
    return sorted(out)


def _backing_text(job_dir: Path, ctx: dict[str, Any], question: str) -> str:
    parts = [question, json.dumps(ctx, default=str)]
    for name in (
        "web/settlements.geojson",
        "web/bridges.geojson",
        "web/hospitals.geojson",
        "repairs.json",
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

    if any(w in q for w in ("structural", "structurally", "collapse", "damage grade")) or (
        "confirm" in q and "damag" in q
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
    ):
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


def _closest_canned(question: str) -> str | None:
    """The suggested question this one is a light rewording of, if any (difflib ratio >= 0.8)."""
    import difflib

    q = question.strip().lower()
    if q in EXISTING_GENERATED_PROMPTS:
        return q
    best = difflib.get_close_matches(q, list(EXISTING_GENERATED_PROMPTS), n=1, cutoff=0.8)
    return best[0] if best else None


def ask_assistant(job_dir: Path, question: str, lang: str = "en") -> dict[str, Any]:
    """Answer a question about one job, in English or Nepali, using only that job's results.

    Order: off-topic refusal; a named settlement, bridge or health facility (answered from its
    own attributes); a suggested question (template over the facts ledger); the language model
    when one is configured, kept only if every number in it is in the results; else templates.
    """
    from dhara.assistant import entities

    ctx = load_job_context(job_dir)
    kpis = {k: v.get("text") for k, v in ctx.get("kpis", {}).items()}

    def done(answer: str, provider: str) -> dict[str, Any]:
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
        }

    if _is_obviously_off_topic(question):
        return done(OFF_TOPIC_FALLBACK, "grounded_rules")

    fixed = _not_measured(question, ctx, lang)
    if fixed:
        return done(fixed, "dhara_verified_ledger")

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

    feature = entities.match(job_dir, question)
    canned = _closest_canned(question)
    if canned and not feature:
        return done(_generate_grounded_fallback(ctx, canned, lang=lang), "dhara_verified_ledger")

    cfg = get_settings()
    provider = cfg.llm_provider
    answer = ""
    system_prompt = build_system_prompt(ctx, lang=lang)
    if feature:
        system_prompt += "\n\nVERIFIED RECORD FOR THE FEATURE ASKED ABOUT:\n" + entities.answer(
            job_dir, feature, "en"
        )
    with contextlib.suppress(Exception):
        if provider == "groq" and cfg.groq_api_key:
            answer, _ = _call_groq_agent(
                cfg.groq_api_key.get_secret_value(),
                job_dir,
                question,
                lang=lang,
                model=cfg.groq_model,
            )
        elif provider == "gemini" and cfg.gemini_api_key:
            answer = _call_gemini_api(
                cfg.gemini_api_key.get_secret_value(), system_prompt, question, cfg.gemini_model
            )
        elif provider == "openai" and cfg.openai_api_key:
            answer = _call_openai_api(
                cfg.openai_api_key.get_secret_value(), system_prompt, question, cfg.openai_model
            )
    if answer.strip() in (
        "No response generated by model.",
        "GIS Agent exceeded maximum reasoning iterations.",
    ):
        answer = ""  # an empty or failed model turn falls back to the verified answer below
    if answer and answer != OFF_TOPIC_FALLBACK:
        answer = _label_bare_number(answer, ctx, lang)
        bad = _unbacked_numbers(answer, _backing_text(job_dir, ctx, question))
        if not bad:
            return done(answer, "grounded_agent")
        answer = ""  # a number the results do not contain: fall back to the verified answer
    elif answer == OFF_TOPIC_FALLBACK and not feature:
        return done(answer, "grounded_agent")

    if feature:
        return done(entities.answer(job_dir, feature, lang), "dhara_verified_ledger")
    return done(_generate_grounded_fallback(ctx, question, lang=lang), "grounded_rules")


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
            f" Details: status={target_bridge['status']}, P(lost)={target_bridge.get('p_lost')}, "
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
