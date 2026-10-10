"""Render the one-page situation report (English and Nepali) from a job's facts table.

The prose is fixed template text, so no language model writes numbers. Every numeral is a fact
display string, and `verify.verify` checks the rendered HTML before it is written. The Nepali
labels are a first draft and need review by a native speaker (stated in the report's limitations).
"""

import base64
import importlib
import io
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rasterio.io import MemoryFile

from dhara.fusion import fuse as F
from dhara.sitrep.facts import DEVANAGARI, Facts, build
from dhara.sitrep.verify import verify

TEMPLATES = Path(__file__).parent / "templates"

ATTR_S = "Contains modified Copernicus Sentinel data 2026."
ATTR_DEM = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space "
    "GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
)
ATTR_OSM = "© OpenStreetMap contributors (ODbL), via the ohsome API."

LABELS = {
    "en": {
        "title": "Flood & debris situation report",
        "event": "Event",
        "area": "Area",
        "generated": "Generated",
        "job": "Job",
        "prototype": "Educational prototype · not operational",
        "k_cut": "settlements cut off",
        "k_bridges": "bridges likely lost",
        "k_buildings": "buildings hit",
        "k_roads": "road/trail network cut",
        "mapped": "mapped",
        "debris": "Debris / scour",
        "flood": "New water",
        "water": "Water before",
        "unobs": "Not observable",
        "cut_pts": "Cut-off settlement",
        "where": "Where did the flood hit?",
        "debris_area": "Debris / scour",
        "flood_area": "new open water",
        "tiers": "By confidence",
        "high": "high",
        "moderate": "moderate",
        "low": "low",
        "unobs_area": "Not observable (radar shadow/layover, cloud, no data)",
        "observable": "observable",
        "unobs_note": "Unobserved ground is unknown, not undamaged.",
        "what": "What was damaged?",
        "bld_tiers": "Buildings hit, by confidence",
        "bld_unobs": "not observable",
        "roads_hit": "Road/trail ways touched by hazard",
        "net": "Analysed network",
        "net_failed": "failed",
        "net_unobs": "unobserved",
        "bridges": "Bridges likely lost (radar return lost)",
        "at_risk": "at risk (hazard at both ends, no radar evidence of loss)",
        "bridge_first": "Bridge repairs, settlements / buildings reconnected",
        "of": "of",
        "who": "Who is cut off?",
        "cut": "Cut off from all hospitals, towns and region exits",
        "hosp_cut": "lost hospital access only",
        "detour": "longer route",
        "connected": "connected",
        "exposure_a": "OSM buildings within",
        "exposure_b": "of cut-off settlements (exposure proxy, not population)",
        "t_cut": "Cut-off settlements (largest first)",
        "c_name": "Settlement",
        "c_status": "Status",
        "c_bld": "Bldgs",
        "c_p": "P(cut)",
        "c_km": "km to hospital before",
        "none_cut": "No settlement lost all access in this analysis.",
        "t_repair": "Repair priorities (reconnects most)",
        "r_what": "Asset",
        "r_set": "Settl.",
        "none_repair": "No single repair reconnects a cut-off settlement.",
        "limits": "Limitations",
        "limits_text": (
            "Radar satellites revisit each track only every few days, so this is a snapshot after "
            "the event, not a warning. At this grid size, houses are 'hit by mapped hazard', not "
            "graded damage. Narrow "
            "gorges hide ground in radar shadow and layover. Debris looks unlike water in radar, so "
            "the debris class rests on change detection. Isolation inherits gaps in pre-event "
            "OpenStreetMap (missing trails or bridges). Nepali labels need native-speaker review."
        ),
        "data": "Data",
        "osm": "OSM snapshot",
        "res": "grid",
        "members": "Evidence members",
        "method": "Generated automatically by Dhara from the job's results store; every number on this page is checked against the facts table.",
        "verified": "",
    },
    "ne": {
        "title": "बाढी तथा लेदो स्थिति प्रतिवेदन",
        "event": "घटना",
        "area": "क्षेत्र",
        "generated": "तयार",
        "job": "कार्य",
        "prototype": "शैक्षिक नमूना · सञ्चालनका लागि होइन",
        "k_cut": "सम्पर्कविहीन बस्ती",
        "k_bridges": "बगेको सम्भावित पुल",
        "k_buildings": "प्रभावित भवन",
        "k_roads": "अवरुद्ध सडक/पैदल मार्ग",
        "mapped": "नक्सामा",
        "debris": "लेदो / कटान",
        "flood": "नयाँ पानी",
        "water": "पहिलेको पानी",
        "unobs": "देख्न नसकिएको",
        "cut_pts": "सम्पर्कविहीन बस्ती",
        "where": "बाढी कहाँ पुग्यो?",
        "debris_area": "लेदो / कटान",
        "flood_area": "नयाँ खुला पानी",
        "tiers": "विश्वसनीयता अनुसार",
        "high": "उच्च",
        "moderate": "मध्यम",
        "low": "न्यून",
        "unobs_area": "देख्न नसकिएको (राडार छाया/लेओभर, बादल)",
        "observable": "देखिएको",
        "unobs_note": "नदेखिएको भूभाग अज्ञात हो, क्षतिरहित होइन।",
        "what": "के क्षति भयो?",
        "bld_tiers": "प्रभावित भवन, विश्वसनीयता अनुसार",
        "bld_unobs": "देख्न नसकिएको",
        "roads_hit": "प्रकोप छोएका सडक/मार्ग",
        "net": "विश्लेषित सञ्जाल",
        "net_failed": "अवरुद्ध",
        "net_unobs": "नदेखिएको",
        "bridges": "बगेको सम्भावित पुल",
        "at_risk": "जोखिममा",
        "bridge_first": "पुल मर्मत, जोडिने बस्ती / भवन",
        "of": "मध्ये",
        "who": "को सम्पर्कविहीन छ?",
        "cut": "सबै अस्पताल, सहर र क्षेत्र निकासबाट सम्पर्कविहीन",
        "hosp_cut": "अस्पताल पहुँच मात्र गुमेको",
        "detour": "लामो बाटो",
        "connected": "जोडिएको",
        "exposure_a": "सम्पर्कविहीन बस्तीको",
        "exposure_b": "भित्र OSM भवन (जनसंख्या होइन)",
        "t_cut": "सम्पर्कविहीन बस्ती",
        "c_name": "बस्ती",
        "c_status": "अवस्था",
        "c_bld": "भवन",
        "c_p": "सम्भावना",
        "c_km": "अस्पताल दूरी (कि.मी.)",
        "none_cut": "यस विश्लेषणमा कुनै बस्तीले सबै पहुँच गुमाएको छैन।",
        "t_repair": "मर्मत प्राथमिकता",
        "r_what": "संरचना",
        "r_set": "बस्ती",
        "none_repair": "एउटै मर्मतले कुनै बस्ती जोडिँदैन।",
        "limits": "सीमाहरू",
        "limits_text": (
            "राडार उपग्रह हरेक ट्र्याकमा केही दिनमा मात्र फर्कन्छ, त्यसैले यो चेतावनी होइन। "
            "भवनको क्षति श्रेणीकरण गरिएको छैन। साँघुरा खोंचमा राडार छायाँ हुन्छ। "
            "पूर्व-घटना OpenStreetMap का त्रुटिहरू विश्लेषणमा आउँछन्। नेपाली पाठ मातृभाषीबाट जाँच आवश्यक।"
        ),
        "data": "तथ्यांक",
        "osm": "OSM मिति",
        "res": "ग्रिड",
        "members": "प्रमाण स्रोत",
        "method": "Dhara ले स्वचालित रूपमा तयार गरेको; यस पानाका सबै अंक तथ्य तालिकासँग जाँचिएका छन्।",
        "verified": "",
    },
}


def _map_png(job_dir: Path, max_px: int = 900) -> str:
    with rasterio.open(job_dir / "classes.tif") as ds:
        step = max(1, int(np.ceil(max(ds.width, ds.height) / max_px)))
        cls = ds.read(1)[::step, ::step]
        tf = ds.transform
    rgb = np.full((3, *cls.shape), 246, np.uint8)
    pal = {
        F.NO_CHANGE: (241, 245, 249),
        F.NOT_OBSERVABLE: (120, 113, 108),
        F.PERMANENT_WATER: (125, 211, 252),
        F.FLOOD: (29, 78, 216),
        F.DEBRIS: (194, 120, 3),
        F.OUTSIDE: (255, 255, 255),
    }
    for k, c in pal.items():
        m = cls == k
        for i in range(3):
            rgb[i][m] = c[i]
    sp = job_dir / "settlements.parquet"
    if sp.exists():
        s = gpd.read_parquet(sp)
        s = s[s["status"].isin(["cut_off", "hospital_cut_off"])]
        for p in s.geometry:
            c, r = (p.x - tf.c) / tf.a, (p.y - tf.f) / tf.e
            r, c = int(r // step), int(c // step)
            r0, r1, c0, c1 = (
                max(r - 4, 0),
                min(r + 5, cls.shape[0]),
                max(c - 4, 0),
                min(c + 5, cls.shape[1]),
            )
            if r0 < r1 and c0 < c1:
                rgb[:, r0:r1, c0:c1] = np.array([185, 28, 28], np.uint8)[:, None, None]
    with MemoryFile() as mf:
        with mf.open(
            driver="PNG", width=rgb.shape[2], height=rgb.shape[1], count=3, dtype="uint8"
        ) as d:
            d.write(rgb)
        return base64.b64encode(mf.read()).decode()


def render(job_dir: Path, lang: str = "en", facts: Facts | None = None) -> tuple[str, dict]:
    f = facts or build(job_dir)
    f.add("attr_s", ATTR_S, source="required attribution")
    f.add("attr_dem", ATTR_DEM, source="required attribution")
    f.add("attr_osm", ATTR_OSM, source="required attribution")
    f.add("sensor_s1", "Sentinel-1", source="sensor name")
    f.add("sensor_s2", "Sentinel-2", source="sensor name")
    texts = {k: v.text for k, v in f.items.items()}
    tables = f.tables
    if lang == "ne":
        texts = {k: v.translate(DEVANAGARI) for k, v in texts.items()}
        tables = {
            name: [{k: str(v).translate(DEVANAGARI) for k, v in row.items()} for row in rows]
            for name, rows in tables.items()
        }
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    page = env.get_template("sitrep.html.j2").render(
        lang=lang, L=LABELS[lang], F=texts, T=tables, map_png=_map_png(job_dir)
    )
    report = verify(page, f)
    return page, report


def write(job_dir: Path) -> dict:
    """Render, verify and write sitrep_<lang>.html (+ .pdf when WeasyPrint is available)."""
    out: dict = {}
    f = build(job_dir)
    (job_dir / "facts.json").write_text(json.dumps(f.to_json(), indent=1, default=str))
    for lang in ("en", "ne"):
        page, rep = render(job_dir, lang, f)
        (job_dir / f"sitrep_{lang}.html").write_text(page)
        out[lang] = {"html": f"sitrep_{lang}.html", "verification": rep}
        try:
            weasyprint = importlib.import_module("weasyprint")  # optional; needs pango

            buf = io.BytesIO()
            weasyprint.HTML(string=page).write_pdf(buf)
            (job_dir / f"sitrep_{lang}.pdf").write_bytes(buf.getvalue())
            out[lang]["pdf"] = f"sitrep_{lang}.pdf"
        except Exception as e:  # PDF is a convenience; the HTML prints to one page as well
            out[lang]["pdf_error"] = f"{type(e).__name__}: {e}"[:200]
    return out
