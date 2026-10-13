"""Answers about one named settlement, bridge or health facility, read straight from a job's layers.

The question is matched to a feature name (exact substring first, then fuzzy with difflib), and
the answer is assembled from that feature's own attributes. Every number printed is a value from
the job's GeoJSON or repairs.json, so the answer needs no language model and cannot invent one.
Coordinates are EPSG:4326.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
                name = _s(props.get("name"))
                if name and name.lower() not in ("hospital", "bridge"):
                    out.append(Feature(kind, name, props | {"_geometry": f.get("geometry")}))
    return out


_CONS = {
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
_VOW = {
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
_SIGN = {
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
_NASAL = {"ँ": "n", "ं": "n", "ः": "h"}


def latin(text: str) -> str:
    """Rough Devanagari to Latin transliteration, only for matching typed names (गोसाइँकुण्ड -> gosainkunda)."""
    out: list[str] = []
    chars = list(text)
    for i, c in enumerate(chars):
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if c in _CONS:
            out.append(_CONS[c])
            if nxt in _CONS or nxt in _NASAL:
                out.append("a")  # inherent vowel; dropped at the end of a word, as in speech
        elif c in _VOW:
            out.append(_VOW[c])
        elif c in _SIGN:
            out.append(_SIGN[c])
        elif c in _NASAL:
            out.append(_NASAL[c])
        elif c == "्":
            continue
        else:
            out.append(c.lower())
    return "".join(out)


def match(job_dir: Path, question: str) -> Feature | None:
    """The feature the question names, or None. Longest exact name wins; else a close fuzzy match."""
    feats = _features(job_dir)
    q = question.lower()
    exact = [f for f in feats if f.name.lower() in q]
    if exact:
        return max(exact, key=lambda f: len(f.name))
    words = re.findall(r"[\wऀ-ॿ'-]+", q)
    grams = {" ".join(words[i : i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
    best, score = None, 0.0
    for f in feats:
        names = {f.name.lower(), latin(f.name)}
        for n in names:
            if len(n) < 4:
                continue
            for g in grams:
                r = difflib.SequenceMatcher(None, n, g).ratio()
                if r > score:
                    best, score = f, r
    return best if score >= 0.8 else None


def _repairs(job_dir: Path) -> list[dict]:
    p = job_dir / "repairs.json"
    with contextlib.suppress(Exception):
        return json.loads(p.read_text())
    return []


def _settlement(f: Feature, repairs: list[dict], ne: bool) -> str:
    p = f.props
    st = _s(p.get("status")) or "not_on_network"
    lines = [f"### {f.name}", ""]
    lines.append(
        f"**{'अवस्था' if ne else 'Status'}:** {(STATUS_NE if ne else STATUS_EN).get(st, st)}."
    )
    b = _s(p.get("buildings_500m"))
    if b:
        lines.append(f"- {'५०० मिटरभित्र भवन' if ne else 'Buildings within 500 m'}: {b}")
    hosp = _s(p.get("hospital_name"))
    kb, ka = _s(p.get("km_to_hospital_pre")), _s(p.get("km_to_hospital_post"))
    hb, ha = _s(p.get("hours_to_hospital_pre")), _s(p.get("hours_to_hospital_post"))
    if hosp and kb:
        lines.append(
            f"- {'घटनाअघि' if ne else 'Before the event'}: {kb} km ({_dur(hb)}) {'सम्म' if ne else 'to'} {hosp}"
        )
        if st in ("cut_off", "hospital_cut_off"):
            lines.append(
                f"- {'घटनापछि' if ne else 'After the event'}: {'सडक मार्ग छैन' if ne else 'no road route'}"
            )
        elif ka:
            lines.append(f"- {'घटनापछि' if ne else 'After the event'}: {ka} km ({_dur(ha)})")
    delay = _s(p.get("delay_hours_target"))
    if delay and delay not in ("0", "0.0"):
        lines.append(f"- {'थप यात्रा समय' if ne else 'Extra travel time'}: {_dur(delay)}")
    pc = _s(p.get("p_cut_off"))
    if pc:
        with contextlib.suppress(ValueError):
            lines.append(
                f"- {'पूर्ण सम्पर्कविहीन हुने सम्भावना' if ne else 'Chance of losing all road access'}: "
                f"{round(float(pc) * 100)}%"
            )
    fixes = [r for r in repairs if f.name in (r.get("settlement_names") or [])]
    lines.append("")
    if st in ("connected", "detour", "no_baseline_route", "not_on_network"):
        if st == "connected":
            lines.append(
                "कुनै मर्मत आवश्यक छैन।"
                if ne
                else "Nothing needs reconnecting: the road route still works."
            )
        elif st == "detour":
            lines.append(
                "मुख्य मार्गका क्षतिग्रस्त खण्ड मर्मत गर्दा छोटो बाटो फर्किन्छ।"
                if ne
                else "Repairing the broken links on its usual route would restore the shorter trip."
            )
        else:
            lines.append(
                "पहुँच परीक्षण गर्न नक्सामा सडक चाहिन्छ।"
                if ne
                else "Access cannot be tested without a mapped road."
            )
    elif fixes:
        head = "यसलाई जोड्ने मर्मत:" if ne else "Repairs that would reconnect it:"
        lines.append(head)
        for r in fixes[:3]:
            kind = "Bridge" if r.get("is_bridge") else "Road"
            nm = r.get("name") or r.get("highway") or r.get("osm_id")
            lines.append(
                f"- {kind} {nm}: reconnects {r.get('settlements_reconnected')} settlement(s), "
                f"{r.get('buildings_reconnected')} buildings"
            )
    else:
        lines.append(
            "कुनै एउटा मर्मतले मात्र यसलाई जोड्दैन; धेरै खण्ड मर्मत गर्नुपर्छ।"
            if ne
            else "No single road or bridge repair reconnects it; several breaks must be fixed together."
        )
    return "\n".join(lines)


def _bridge(f: Feature, repairs: list[dict], ne: bool) -> str:
    p = f.props
    st = _s(p.get("status"))
    lines = [f"### {f.name}", "", f"**Status:** {BRIDGE_EN.get(st, st)}."]
    pl, ph, dd = _s(p.get("p_lost")), _s(p.get("p_hazard")), _s(p.get("bright_drop_db"))
    with contextlib.suppress(ValueError):
        if pl:
            lines.append(f"- Chance lost: {round(float(pl) * 100)}%")
        if ph:
            lines.append(f"- Flood or debris at both ends: {round(float(ph) * 100)}%")
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
                served.append(_s(p.get("name")) or "unnamed")
                if _s(p.get("status")) in ("cut_off", "hospital_cut_off"):
                    lost.append(_s(p.get("name")) or "unnamed")
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
