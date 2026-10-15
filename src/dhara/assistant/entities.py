"""Answer about a selected or explicitly named map feature using its job-layer record.

Feature resolution uses stable OSM identifiers, selected coordinates, or exact names. Coordinates
and exported API geometries are EPSG:4326.
"""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dhara.osm_names import shown_name

STATUS_EN = {
    "cut_off": "cut off: no road or trail out to a town, health facility or the edge of the area",
    "hospital_cut_off": "without road access to a hospital, though it can still reach a town or the edge of the area",
    "detour": "still connected, but only by a longer route than before",
    "connected": "connected: its pre-event road route still works",
    "no_baseline_route": "not reachable by road even before the event in the mapped network",
    "not_on_network": "not on the mapped road network, so access cannot be judged",
}
STATUS_NE = {
    "cut_off": "सम्पर्कविहीन: सहर, स्वास्थ्य संस्था वा क्षेत्रबाहिर जाने कुनै सडक बाँकी छैन",
    "hospital_cut_off": "अस्पतालसम्म सडक पहुँच छैन",
    "detour": "जोडिएको, तर पहिलेभन्दा लामो बाटोबाट",
    "connected": "जोडिएको: घटनाअघिको सडक मार्ग अझै चल्छ",
    "no_baseline_route": "घटनाअघि पनि नक्सामा सडक मार्ग थिएन",
    "not_on_network": "नक्साको सडक सञ्जालमा छैन, पहुँच आकलन गर्न सकिँदैन",
}
BRIDGE_EN = {
    "likely_lost": "likely lost",
    "at_risk": "at risk (flood or debris at both ends, but no radar sign of loss)",
    "likely_standing": "likely standing",
    "not_observable": "not observable by the radar",
}


_CONS: dict[str, str] = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "ङ": "ng",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ञ": "ny",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "ण": "n",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "w",
    "श": "sh",
    "ष": "sh",
    "स": "s",
    "ह": "h",
}
_VOW: dict[str, str] = {
    "अ": "a",
    "आ": "a",
    "इ": "i",
    "ई": "i",
    "उ": "u",
    "ऊ": "u",
    "ए": "e",
    "ऐ": "ai",
    "ओ": "o",
    "औ": "au",
    "ऋ": "ri",
}
_SIGN: dict[str, str] = {
    "ा": "a",
    "ि": "i",
    "ी": "i",
    "ु": "u",
    "ू": "u",
    "े": "e",
    "ै": "ai",
    "ो": "o",
    "ौ": "au",
    "ृ": "ri",
}
_NASAL: dict[str, str] = {"ँ": "n", "ं": "n", "ः": "h"}


def latin(text: str) -> str:
    """Rough Devanagari to Latin transliteration for matching."""
    c = list(text)
    out: list[str] = []
    for i, ch in enumerate(c):
        nxt = c[i + 1] if i + 1 < len(c) else ""
        if ch in _CONS:
            out.append(_CONS[ch] + ("a" if nxt in _CONS or nxt in _NASAL else ""))
        elif ch in _VOW:
            out.append(_VOW[ch])
        elif ch in _SIGN:
            out.append(_SIGN[ch])
        elif ch in _NASAL:
            out.append(_NASAL[ch])
        elif ch != "्":
            out.append(ch.lower())
    return "".join(out)


@dataclass
class Feature:
    kind: str  # settlement | bridge | hospital
    name: str
    props: dict[str, Any]


def _dur(hours: str) -> str:
    """'0.02' hours -> '1 min'; '1.4' -> '1.4 h'."""
    try:
        h = float(hours)
    except ValueError:
        return hours
    return f"{max(1, round(h * 60))} min" if h < 1 else f"{hours} h"


def _s(v: Any) -> str:
    return "" if v is None or str(v) in ("None", "nan", "") else str(v)


def _features(job_dir: Path) -> list[Feature]:
    out: list[Feature] = []
    for kind, fname in (
        ("settlement", "settlements"),
        ("bridge", "bridges"),
        ("hospital", "hospitals"),
    ):
        p = job_dir / "web" / f"{fname}.geojson"
        if not p.exists():
            continue
        with contextlib.suppress(Exception):
            for f in json.loads(p.read_text()).get("features", []):
                props = f.get("properties") or {}
                name = _s(shown_name(props.get("name")))
                osm_id = _s(props.get("osm_id"))
                if name.lower() in ("hospital", "bridge"):
                    continue
                if name or osm_id:
                    out.append(Feature(kind, name, props | {"_geometry": f.get("geometry")}))
    return out


def match(
    job_dir: Path,
    question: str,
    selected_entity: dict[str, Any] | None = None,
) -> Feature | None:
    """Match a named feature, preferring selected OSM ID/coordinates to resolve duplicate names."""
    feats = _features(job_dir)
    q = question.lower()
    selected = selected_entity or {}
    selected_id = _s(selected.get("id"))
    selected_kind = _s(selected.get("kind"))
    kind_aliases = {"flow-settlement": "settlement"}
    if selected_id:
        by_id = [
            f
            for f in feats
            if _s(f.props.get("osm_id")) == selected_id
            and (not selected_kind or f.kind == kind_aliases.get(selected_kind, selected_kind))
        ]
        if len(by_id) == 1:
            return by_id[0]
    id_matches = [f for f in feats if _s(f.props.get("osm_id")).lower() in q]
    if len(id_matches) == 1:
        return id_matches[0]
    exact = [f for f in feats if f.name.lower() in q]
    exact = [f for f in exact if f.name]
    if exact:
        longest = max(len(f.name) for f in exact)
        exact = [f for f in exact if len(f.name) == longest]
        if len(exact) == 1:
            return exact[0]
        if selected_id:
            by_id = [f for f in exact if _s(f.props.get("osm_id")) == selected_id]
            if len(by_id) == 1:
                return by_id[0]
        coordinates = selected.get("coordinates")
        if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
            try:
                lon, lat = float(coordinates[0]), float(coordinates[1])

                # GeoJSON feature coordinates are EPSG:4326. Scale longitude by latitude so
                # nearest-feature selection behaves sensibly away from the equator.
                def distance(feature: Feature) -> float:
                    geometry = feature.props.get("_geometry") or {}
                    coords = geometry.get("coordinates") or []
                    if geometry.get("type") == "Point" and len(coords) >= 2:
                        lon_scale = max(0.01, abs(math.cos(math.radians(lat))))
                        return ((float(coords[0]) - lon) * lon_scale) ** 2 + (
                            float(coords[1]) - lat
                        ) ** 2
                    return float("inf")

                nearest = sorted(exact, key=distance)
                if len(nearest) == 1 or distance(nearest[0]) < distance(nearest[1]):
                    return nearest[0]
            except (TypeError, ValueError):
                pass
        # Never resolve same-name features by arbitrary file order.
        return None
    return None


def ambiguous_exact_matches(job_dir: Path, question: str) -> list[Feature]:
    """Return same-name exact matches when the question cannot identify one uniquely."""
    q = question.lower()
    exact = [f for f in _features(job_dir) if f.name and f.name.lower() in q]
    if not exact:
        return []
    longest = max(len(f.name) for f in exact)
    exact = [f for f in exact if len(f.name) == longest]
    return exact if len(exact) > 1 else []


def _repairs(job_dir: Path) -> list[dict]:
    p = job_dir / "repairs.json"
    with contextlib.suppress(Exception):
        return json.loads(p.read_text())
    return []


def _settlement(f: Feature, repairs: list[dict], ne: bool) -> str:
    p = f.props
    st = _s(p.get("status")) or "not_on_network"
    place = _s(p.get("place"))
    label = f"mapped {place}" if place else "mapped settlement"
    label = f"unnamed {label}" if not f.name else f"{label} **{f.name}**"
    osm_id = _s(p.get("osm_id"))
    identity = f"यो नक्सामा देखाइएको {label} हो।" if ne else f"This is the {label}."
    if osm_id:
        identity += f" (OSM {osm_id})"
    heading = f.name or ("नाम नभएको बस्ती" if ne else "Unnamed settlement")
    lines = [f"### {heading}", "", identity, ""]  # same layout as bridge and hospital answers
    lines.append(
        f"- **{'पहुँच' if ne else 'Road access'}:** {(STATUS_NE if ne else STATUS_EN).get(st, st)}."
    )
    b = _s(p.get("buildings_500m"))
    if b:
        lines.append(f"- {'५০০ मिटरभित्रका भवन' if ne else 'Buildings within 500 m'}: {b}")
    hosp = _s(shown_name(p.get("hospital_name")))
    kb, ka = _s(p.get("km_to_hospital_pre")), _s(p.get("km_to_hospital_post"))
    hb, ha = _s(p.get("hours_to_hospital_pre")), _s(p.get("hours_to_hospital_post"))
    if hosp and kb:
        if st in ("cut_off", "hospital_cut_off"):
            lines.append(
                f"- {'अस्पतालसम्मको सडक मार्ग' if ne else 'Hospital road route'}: "
                f"{kb} km before; {'घटनापछि मार्ग छैन' if ne else 'no mapped route after the event'}."
            )
        elif ka:
            if kb == ka and hb == ha:
                route = f"{kb} km ({_dur(hb)}) before and after"
            else:
                route = f"{kb} km ({_dur(hb)}) before; {ka} km ({_dur(ha)}) after"
            lines.append(f"- {'अस्पतालसम्मको सडक मार्ग' if ne else 'Hospital road route'}: {route}.")
    pc = _s(p.get("p_cut_off"))
    if pc:
        with contextlib.suppress(ValueError):
            if float(pc) > 0:
                lines.append(
                    f"- {'सिमुलेसनमा कट-अफ भएको' if ne else 'Cut off in network simulations'}: "
                    f"{round(float(pc) * 100)}%"
                )
    fixes = [r for r in repairs if f.name in (r.get("settlement_names") or [])]
    if st in ("connected", "detour", "no_baseline_route", "not_on_network"):
        if st == "connected":
            lines.append(
                "हाल नक्सामा मार्ग जोडिएको छ; पुनःजोड्ने मर्मत पहिचान भएको छैन।"
                if ne
                else "No reconnection repair is identified because the mapped route remains connected."
            )
        elif st == "detour":
            lines.append(
                "क्षतिग्रस्त मुख्य मार्ग मर्मत गर्दा छोटो बाटो फर्किन सक्छ।"
                if ne
                else "Repairing the disrupted usual route could restore the shorter trip."
            )
        else:
            lines.append(
                "नक्सामा सडक नभएकाले पहुँच परीक्षण गर्न सकिँदैन।"
                if ne
                else "Road access cannot be assessed because no mapped road route is available."
            )
    elif fixes:
        head = "यसलाई जोड्ने मर्मत:" if ne else "Repairs that would reconnect it:"
        lines.append(head)
        for r in fixes[:3]:
            kind = "Bridge" if r.get("is_bridge") else "Road"
            nm = shown_name(r.get("name")) or r.get("highway") or r.get("osm_id")
            lines.append(
                f"- {kind} {nm}: reconnects {r.get('settlements_reconnected')} settlement(s), "
                f"{r.get('buildings_reconnected')} buildings"
            )
    else:
        lines.append(
            "यसलाई जोड्न धेरै क्षतिग्रस्त खण्ड मर्मत गर्नुपर्ने देखिन्छ।"
            if ne
            else "Reconnecting it appears to require repairs to multiple damaged links."
        )
    return "\n".join(lines)


def _bridge(f: Feature, repairs: list[dict], ne: bool) -> str:
    p = f.props
    st = _s(p.get("status"))
    lines = [f"### {f.name}", "", f"**Status:** {BRIDGE_EN.get(st, st)}."]
    pl, ph, dd = _s(p.get("p_lost")), _s(p.get("p_hazard")), _s(p.get("bright_drop_db"))
    with contextlib.suppress(ValueError):
        if pl:
            lines.append(f"- Heuristic loss score (not a calibrated probability): {float(pl):.2f}")
        if ph:
            lines.append(f"- Heuristic flood or debris score at both ends: {float(ph):.2f}")
        if dd:
            v = float(dd)
            lines.append(
                f"- Bright radar return {'fell' if v >= 0 else 'rose'} by {abs(v):.1f} dB after the event"
            )
    lines.append(
        "\nA standing bridge reflects radar strongly; a fall of about 3 dB or more is the main sign of loss."
    )
    oid = _s(p.get("osm_id"))
    for i, r in enumerate(repairs):
        if r.get("osm_id") == oid:
            lines.append(
                f"\nRepair priority {i + 1}: reconnects {r.get('settlements_reconnected')} settlement(s) and "
                f"{r.get('buildings_reconnected')} buildings."
            )
    return "\n".join(lines)


def _hospital(f: Feature, job_dir: Path, ne: bool) -> str:
    served, lost = [], []
    with contextlib.suppress(Exception):
        gj = json.loads((job_dir / "web" / "settlements.geojson").read_text())
        for s in gj.get("features", []):
            p = s.get("properties") or {}
            if _s(p.get("hospital_name")) == f.name:
                served.append(_s(shown_name(p.get("name"))) or "unnamed")
                if _s(p.get("status")) in ("cut_off", "hospital_cut_off"):
                    lost.append(_s(shown_name(p.get("name"))) or "unnamed")
    lines = [f"### {f.name}", ""]
    lines.append(f"- Nearest hospital for {len(served)} mapped settlements")
    lines.append(f"- Of those, {len(lost)} lost road access to it")
    if lost:
        lines.append(f"- {', '.join(lost[:8])}")
    return "\n".join(lines)


def answer(job_dir: Path, f: Feature, lang: str = "en") -> str:
    ne = lang.lower().startswith("ne")
    rep = _repairs(job_dir)
    if f.kind == "settlement":
        return _settlement(f, rep, ne)
    if f.kind == "bridge":
        return _bridge(f, rep, ne)
    return _hospital(f, job_dir, ne)
