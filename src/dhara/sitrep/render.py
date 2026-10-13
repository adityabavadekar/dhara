"""Render the one-page situation report (English and Nepali) from a job's facts table.

The prose is fixed template text, so no language model writes numbers. Every numeral is a fact
display string, and `verify.verify` checks the rendered HTML before it is written. The Nepali
labels are a first draft and need review by a native speaker (stated in the report's limitations).
"""

import contextlib
import importlib
import io
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from dhara.sitrep.facts import DEVANAGARI, Facts, build
from dhara.sitrep.verify import verify

TEMPLATES = Path(__file__).parent / "templates"

ATTR_S = "Contains modified Copernicus Sentinel data 2026."
ATTR_DEM = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space "
    "GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
)
ATTR_OSM = "© OpenStreetMap contributors (ODbL), via the ohsome API."  # pre-event snapshot

LABELS = {
    "en": {
        "brand": "Dhara situation report",
        "title": "Flood and debris situation report",
        "event": "Event",
        "area": "Area",
        "generated": "Generated",
        "job": "Analysis",
        "prototype": "Educational prototype · not operational",
        "bottom_line": "Bottom line",
        "bl_cut_a": "settlements have no road out at all, out of",
        "bl_cut_b": "mapped.",
        "bl_hosp": "more lost road access to a hospital, and",
        "bl_detour": "can only get out by a longer route.",
        "bl_bld": "buildings lie within",
        "bl_bld_b": "of the settlements with no road out.",
        "bl_br_a": "of",
        "bl_br_b": "bridges are likely lost and",
        "bl_br_c": "more are at risk;",
        "bl_road": "of mapped road and trail is impassable.",
        "bl_hz_a": "Mapped debris and scour covers",
        "bl_hz_b": "and new water",
        "bl_hz_c": "buildings of",
        "bl_hz_d": "lie in the hazard zone.",
        "bl_unobs_a": "of ground was hidden from the radar and is",
        "bl_unobs_b": "not known to be safe.",
        "k_hazard": "Flood and debris",
        "k_hazard_sub": "debris · new water",
        "k_cut": "No road out",
        "k_cut_sub": "settlements of",
        "k_bridges": "Bridges likely lost",
        "k_bridges_sub": "bridges, more at risk:",
        "k_bld": "Buildings in hazard zone",
        "k_bld_sub": "of mapped buildings",
        "map_overview": "Overview",
        "map_overview_note": "Numbered circles are the settlements with no road out or no hospital access, in the order of the tables. Dashed blue: analysed area.",
        "lg_debris": "Debris and scour",
        "lg_flood": "New water",
        "lg_water": "Water before",
        "lg_unobs": "Not observable",
        "lg_failed": "Impassable road",
        "lg_cut": "No road out",
        "lg_hosp_cut": "No hospital access",
        "lg_detour": "Longer route",
        "lg_connected": "Connected",
        "lg_hospital": "Hospital or clinic",
        "lg_repair": "Repair priority",
        "lg_bridge_lost": "Bridge likely lost",
        "lg_bridge_risk": "Bridge at risk",
        "lg_bld": "Building hit",
        "act_first": "Act on first",
        "act_none": "No settlement lost all road access in this analysis.",
        "top_repairs": "Repairs that reconnect the most",
        "reconnects": "reconnects",
        "settl": "settlements",
        "bldgs": "buildings",
        "trust": "How far to trust this",
        "trust_text": "A building hit means its footprint overlaps mapped flood or debris, not a damage grade. Missing trails in the pre-event map cannot be tested.",
        "observable": "of the area was visible to the radar.",
        "p2_title": "Where did the flood hit, and what was damaged?",
        "zoom_title": "Most affected window",
        "zoom_note": "Full-resolution close-up of the window with the most mapped flood and debris. Squares are bridges, numbered as in the bridge table.",
        "t_hazard": "Mapped area",
        "c_class": "Class",
        "c_km2": "km²",
        "h_debris": "Debris and scour",
        "h_flood": "New water",
        "h_water": "Water before the event",
        "h_unobs": "Not observable",
        "t_conf": "By confidence",
        "high": "High",
        "moderate": "Moderate",
        "low": "Low",
        "conf_note": "High: strong change that two or more methods agree on. Low is a lead to check, not a finding.",
        "t_damage": "Damage",
        "d_bld": "Buildings in hazard zone",
        "d_bld_unobs": "Buildings where the radar could not see",
        "d_roads": "Road and trail ways touched",
        "d_net": "Analysed road network",
        "d_failed": "Impassable",
        "d_net_unobs": "Network not observed",
        "t_bridges": "Bridges likely lost or at risk",
        "b_name": "Bridge",
        "b_status": "Status",
        "b_p": "Chance lost",
        "b_drop": "Radar drop, dB",
        "b_len": "Span, m",
        "b_none": "No bridge is likely lost or at risk.",
        "bridge_note": "A standing bridge reflects radar strongly; a drop of about three decibels or more is the main sign of loss.",
        "t_fac": "Schools and health facilities in the hazard zone",
        "f_name": "Facility",
        "f_type": "Type",
        "f_hazard": "Hazard",
        "f_conf": "Confidence",
        "f_none": "No tagged school or health facility lies in the hazard zone.",
        "p3_title": "Who is cut off?",
        "access_note": "Red lines are road links crossed by flood or debris. Numbered green squares are the repairs in the order of the repair table.",
        "t_cut": "Settlements with no road out or no hospital access",
        "c_name": "Settlement",
        "c_status": "Status",
        "c_bld": "Buildings",
        "c_p": "Chance of no road out",
        "c_km": "km to hospital before",
        "c_hosp": "Nearest hospital",
        "c_where": "Location",
        "st_cut_off": "no road out",
        "st_hospital_cut_off": "no hospital",
        "t_detour": "Longer routes",
        "d_hpre": "Hours before",
        "d_hpost": "Hours after",
        "d_delay": "Extra hours",
        "detour_none": "No settlement needs a longer route.",
        "t_repair": "Repair priorities",
        "r_asset": "Road or bridge",
        "r_settl": "Settlements reconnected",
        "r_bld": "Buildings reconnected",
        "repair_none": "No single repair reconnects a cut-off settlement; several breaks must be fixed together.",
        "t_hosp": "Hospitals and clinics",
        "h_name": "Facility",
        "h_type": "Type",
        "h_access": "Road access",
        "cut_note": "Cut off means no road or trail in the pre-event map still links the settlement to a town, a health facility or the edge of the area. Travel times use typical speeds per road type.",
        "p4_title": "Flood paths traced in this area",
        "fp_note": "Steepest-descent paths on the Copernicus DEM from points picked upstream. Arrival times use an assumed slope-to-speed rule that has not been calibrated: use them to order warnings, not as forecasts.",
        "fp_trace": "Trace",
        "fp_start": "Start",
        "fp_len": "Length, km",
        "fp_drop": "Drop, m",
        "fp_settl": "Settlements",
        "fp_stop": "Stopped at",
        "fp_order": "Order",
        "fp_km": "km down",
        "fp_dz": "m above river",
        "fp_exp": "Exposure",
        "fp_arr": "Reached in, min (rough)",
        "p5_title": "Data, method and limitations",
        "t_data": "Data used",
        "ds_s1": "radar pairs (same track)",
        "ds_s2": "optical pair",
        "ds_dem": "Copernicus DEM GLO-30",
        "ds_osm": "OpenStreetMap snapshot",
        "ds_grid": "Analysis grid",
        "ds_members": "Evidence methods",
        "method": "Method",
        "method_text": "Radar change detection maps new water and, separately, debris and scour; a segmentation model trained on Kuro Siwo adds a third opinion. Terrain from the Copernicus DEM rules out water on steep slopes and high ground. The results are pooled into probabilities and a confidence tier. Pre-event OpenStreetMap buildings, roads and bridges are overlaid; bridges are also tested for loss of their strong radar return. The road network is cut where flood or debris crosses it, and each settlement is tested for a route out, before and after.",
        "limits": "Limitations",
        "limits_list": [
            "Sentinel satellites revisit a track only every few days, so this is a snapshot after the event, not a warning.",
            "Steep valleys hide ground from the radar. Hidden ground is unknown, not undamaged.",
            "At this grid size a house is one or two pixels; building hits are exposure, not graded damage.",
            "Wet sediment and open water look alike to radar on a valley floor.",
            "Roads, trails and bridges missing from the pre-event map cannot be tested.",
            "Population is not an allowed input, so impact is counted in buildings.",
            "The Nepali text is a first draft and needs review by a native speaker.",
        ],
        "links": "Links",
        "l_prov": "Provenance: scenes, parameters, timings",
        "l_zip": "All map layers and this report (zip)",
        "l_other": "Other language",
        "credits": "Data credits",
        "footer": "Generated automatically by Dhara from this analysis. Every number in this report is read from the results and checked before the report is written.",
    },
    "ne": {
        "brand": "Dhara स्थिति प्रतिवेदन",
        "title": "बाढी तथा लेदो स्थिति प्रतिवेदन",
        "event": "घटना",
        "area": "क्षेत्र",
        "generated": "तयार",
        "job": "विश्लेषण",
        "prototype": "शैक्षिक नमूना · सञ्चालनका लागि होइन",
        "bottom_line": "मुख्य कुरा",
        "bl_cut_a": "बस्तीबाट बाहिर जाने कुनै सडक छैन, जम्मा",
        "bl_cut_b": "बस्तीमध्ये।",
        "bl_hosp": "थप बस्तीले अस्पतालसम्मको सडक पहुँच गुमाए, र",
        "bl_detour": "बस्ती लामो बाटोबाट मात्र बाहिर निस्कन सक्छन्।",
        "bl_bld": "भवन",
        "bl_bld_b": "भित्र छन्, ती सडकविहीन बस्तीहरूको।",
        "bl_br_a": "मध्ये",
        "bl_br_b": "पुल बगेको सम्भावना छ र",
        "bl_br_c": "थप जोखिममा छन्;",
        "bl_road": "सडक तथा पैदल मार्ग अवरुद्ध छ।",
        "bl_hz_a": "लेदो तथा कटान",
        "bl_hz_b": "र नयाँ पानी",
        "bl_hz_c": "भवन, जम्मा",
        "bl_hz_d": "मध्ये, प्रकोप क्षेत्रमा छन्।",
        "bl_unobs_a": "भूभाग राडारले देख्न सकेन र",
        "bl_unobs_b": "सुरक्षित भनिन सकिँदैन।",
        "k_hazard": "बाढी तथा लेदो",
        "k_hazard_sub": "लेदो · नयाँ पानी",
        "k_cut": "बाहिर जाने सडक छैन",
        "k_cut_sub": "बस्ती, जम्मा",
        "k_bridges": "बगेको सम्भावित पुल",
        "k_bridges_sub": "पुलमध्ये, जोखिममा:",
        "k_bld": "प्रकोप क्षेत्रमा भवन",
        "k_bld_sub": "नक्सामा भएका भवनमध्ये",
        "map_overview": "समग्र नक्सा",
        "map_overview_note": "अंकित गोला सडक वा अस्पताल पहुँच गुमाएका बस्ती हुन्, तालिकाको क्रममा। नीलो धर्को: विश्लेषित क्षेत्र।",
        "lg_debris": "लेदो तथा कटान",
        "lg_flood": "नयाँ पानी",
        "lg_water": "पहिलेको पानी",
        "lg_unobs": "देख्न नसकिएको",
        "lg_failed": "अवरुद्ध सडक",
        "lg_cut": "बाहिर जाने सडक छैन",
        "lg_hosp_cut": "अस्पताल पहुँच छैन",
        "lg_detour": "लामो बाटो",
        "lg_connected": "जोडिएको",
        "lg_hospital": "अस्पताल वा क्लिनिक",
        "lg_repair": "मर्मत प्राथमिकता",
        "lg_bridge_lost": "पुल बगेको सम्भावना",
        "lg_bridge_risk": "पुल जोखिममा",
        "lg_bld": "प्रभावित भवन",
        "act_first": "पहिले के गर्ने",
        "act_none": "यस विश्लेषणमा कुनै बस्तीले सबै सडक पहुँच गुमाएको छैन।",
        "top_repairs": "सबैभन्दा धेरै जोड्ने मर्मत",
        "reconnects": "जोड्छ",
        "settl": "बस्ती",
        "bldgs": "भवन",
        "trust": "कति भरोसा गर्ने",
        "trust_text": "घटनाअघि र पछिका राडार तस्बिरबीचको परिवर्तन। प्रभावित भवन भनेको नक्सामा बाढी वा लेदोसँग खप्टिएको भवन हो, क्षतिको श्रेणी होइन। सडक र पुल घटनाअघिको OpenStreetMap बाट लिइएका हुन्।",
        "observable": "क्षेत्र राडारले देख्यो।",
        "p2_title": "बाढी कहाँ पुग्यो, के क्षति भयो?",
        "zoom_title": "सबैभन्दा प्रभावित भाग",
        "zoom_note": "सबैभन्दा धेरै बाढी र लेदो भएको भागको नजिकको दृश्य। वर्गहरू पुल हुन्, पुल तालिकाको क्रममा।",
        "t_hazard": "नक्सामा देखिएको क्षेत्र",
        "c_class": "वर्ग",
        "c_km2": "वर्ग कि.मी.",
        "h_debris": "लेदो तथा कटान",
        "h_flood": "नयाँ पानी",
        "h_water": "घटनाअघिको पानी",
        "h_unobs": "देख्न नसकिएको",
        "t_conf": "विश्वसनीयता अनुसार",
        "high": "उच्च",
        "moderate": "मध्यम",
        "low": "न्यून",
        "conf_note": "न्यून विश्वसनीयता जाँच गर्नुपर्ने संकेत मात्र हो।",
        "t_damage": "क्षति",
        "d_bld": "प्रकोप क्षेत्रमा भवन",
        "d_bld_unobs": "राडारले नदेखेको ठाउँका भवन",
        "d_roads": "प्रकोप छोएका सडक/मार्ग",
        "d_net": "विश्लेषित सडक सञ्जाल",
        "d_failed": "अवरुद्ध",
        "d_net_unobs": "नदेखिएको सञ्जाल",
        "t_bridges": "बगेका वा जोखिममा रहेका पुल",
        "b_name": "पुल",
        "b_status": "अवस्था",
        "b_p": "बगेको सम्भावना",
        "b_drop": "राडार ह्रास, dB",
        "b_len": "लम्बाइ, मि.",
        "b_none": "कुनै पुल बगेको वा जोखिममा छैन।",
        "bridge_note": "उभिएको पुलले राडार बलियोसँग फर्काउँछ; त्यो घट्नु क्षतिको मुख्य संकेत हो।",
        "t_fac": "प्रकोप क्षेत्रमा विद्यालय तथा स्वास्थ्य संस्था",
        "f_name": "संस्था",
        "f_type": "प्रकार",
        "f_hazard": "प्रकोप",
        "f_conf": "विश्वसनीयता",
        "f_none": "प्रकोप क्षेत्रमा कुनै विद्यालय वा स्वास्थ्य संस्था छैन।",
        "p3_title": "को सम्पर्कविहीन छ?",
        "access_note": "रातो धर्का बाढी वा लेदोले काटेका सडक हुन्। हरिया वर्ग मर्मत तालिकाको क्रममा छन्।",
        "t_cut": "सडक वा अस्पताल पहुँच गुमाएका बस्ती",
        "c_name": "बस्ती",
        "c_status": "अवस्था",
        "c_bld": "भवन",
        "c_p": "पूर्ण सम्पर्कविहीन सम्भावना",
        "c_km": "अस्पताल दूरी पहिले (कि.मी.)",
        "c_hosp": "नजिकको अस्पताल",
        "c_where": "स्थान",
        "st_cut_off": "बाहिर जाने सडक छैन",
        "st_hospital_cut_off": "अस्पताल छैन",
        "t_detour": "लामो बाटो",
        "d_hpre": "पहिले घण्टा",
        "d_hpost": "पछि घण्टा",
        "d_delay": "थप घण्टा",
        "detour_none": "कुनै बस्तीलाई लामो बाटो चाहिँदैन।",
        "t_repair": "मर्मत प्राथमिकता",
        "r_asset": "सडक वा पुल",
        "r_settl": "जोडिने बस्ती",
        "r_bld": "जोडिने भवन",
        "repair_none": "एउटै मर्मतले कुनै बस्ती जोडिँदैन।",
        "t_hosp": "अस्पताल तथा क्लिनिक",
        "h_name": "संस्था",
        "h_type": "प्रकार",
        "h_access": "सडक पहुँच",
        "cut_note": "सम्पर्कविहीन भनेको घटनाअघिको नक्साका कुनै सडक वा मार्गले बस्तीलाई सहर, स्वास्थ्य संस्था वा क्षेत्रको किनारासँग नजोड्नु हो।",
        "p4_title": "यस क्षेत्रमा कोरिएका बाढी मार्ग",
        "fp_note": "माथिल्लो बिन्दुबाट Copernicus DEM मा सबैभन्दा भिरालो बाटो। पुग्ने समय अनुमानित मात्र हो, पूर्वानुमान होइन।",
        "fp_trace": "मार्ग",
        "fp_start": "सुरु",
        "fp_len": "लम्बाइ, कि.मी.",
        "fp_drop": "झरान, मि.",
        "fp_settl": "बस्ती",
        "fp_stop": "रोकिएको कारण",
        "fp_order": "क्रम",
        "fp_km": "कि.मी. तल",
        "fp_dz": "नदीभन्दा माथि, मि.",
        "fp_exp": "जोखिम",
        "fp_arr": "पुग्ने समय, मिनेट (अनुमानित)",
        "p5_title": "तथ्यांक, विधि र सीमाहरू",
        "t_data": "प्रयोग गरिएको तथ्यांक",
        "ds_s1": "राडार जोडी (एउटै ट्र्याक)",
        "ds_s2": "अप्टिकल जोडी",
        "ds_dem": "Copernicus DEM GLO-30",
        "ds_osm": "OpenStreetMap मिति",
        "ds_grid": "विश्लेषण ग्रिड",
        "ds_members": "प्रमाण विधि",
        "method": "विधि",
        "method_text": "राडार परिवर्तन पहिचानले नयाँ पानी र लेदो छुट्टाछुट्टै नक्सामा देखाउँछ; Kuro Siwo मा तालिम दिइएको मोडेलले थप राय दिन्छ। Copernicus DEM ले भिरालो र अग्लो ठाउँमा पानी हुन नसक्ने कुरा जाँच्छ। घटनाअघिका भवन, सडक र पुल खप्टाइन्छन्, र हरेक बस्तीको बाहिर जाने बाटो घटनाअघि र पछि जाँचिन्छ।",
        "limits": "सीमाहरू",
        "limits_list": [
            "Sentinel उपग्रह हरेक ट्र्याकमा केही दिनमा मात्र फर्कन्छ, त्यसैले यो चेतावनी होइन।",
            "साँघुरा खोंचमा राडार छायाँ हुन्छ। नदेखिएको भूभाग अज्ञात हो, क्षतिरहित होइन।",
            "भवनको क्षति श्रेणीकरण गरिएको छैन।",
            "नक्सामा नभएका सडक, पैदल मार्ग र पुल जाँच्न सकिँदैन।",
            "जनसंख्या प्रयोग गर्न नपाइने भएकाले असर भवनमा गनिएको छ।",
            "नेपाली पाठ मातृभाषीबाट जाँच आवश्यक छ।",
        ],
        "links": "लिङ्कहरू",
        "l_prov": "स्रोत विवरण: दृश्य, प्यारामिटर, समय",
        "l_zip": "सबै नक्सा तह र यो प्रतिवेदन (zip)",
        "l_other": "अर्को भाषा",
        "credits": "तथ्यांक स्रोत",
        "footer": "Dhara ले यस विश्लेषणबाट स्वचालित रूपमा तयार गरेको। यस प्रतिवेदनका सबै अंक नतिजाबाट लिइएका र जाँचिएका छन्।",
    },
}


def _pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 1) if whole > 0 else 0.0


def _maps(job_dir: Path, f: Facts) -> dict[str, str | list[str] | None]:
    """All report maps as base64 PNGs; any map that cannot be drawn is left out, not faked."""
    from dhara.sitrep import maps

    out: dict[str, str | list[str] | None] = {
        "overview": None,
        "zoom": None,
        "access": None,
        "flow": [],
    }
    try:
        L = maps.load(job_dir)
    except Exception:
        return out
    cut, rep, br = f.ids.get("cut_off", []), f.ids.get("repairs", []), f.ids.get("bridges", [])
    with contextlib.suppress(Exception):
        out["overview"] = maps.overview(L, cut[:10])
    with contextlib.suppress(Exception):
        w = maps.hazard_window(L)
        if w is not None:
            out["zoom"] = maps.hazard_zoom(L, w, br)
    with contextlib.suppress(Exception):
        out["access"] = maps.access(L, cut, rep)
    flows: list[str] = []
    for name in f.ids.get("flowpaths", []):
        try:
            flows.append(maps.flowpath(L, job_dir.parent / name))
        except Exception:
            flows.append("")
    out["flow"] = flows
    return out


def render(
    job_dir: Path, lang: str = "en", facts: Facts | None = None, maps: dict | None = None
) -> tuple[str, dict]:
    f = facts or build(job_dir)
    f.add("attr_s", ATTR_S, source="required attribution")
    f.add("attr_dem", ATTR_DEM, source="required attribution")
    f.add("attr_osm", ATTR_OSM, source="required attribution")
    f.add("sensor_s1", "Sentinel-1", source="sensor name")
    f.add("sensor_s2", "Sentinel-2", source="sensor name")
    f.add("dem_name", "Copernicus DEM GLO-30", source="dataset name")
    f.add(
        "attr_train",
        "Training data: Kuro Siwo (Bountos et al., NeurIPS 2024).",
        source="required citation",
    )
    raw = {k: v.value for k, v in f.items.items()}

    def num(k: str) -> float:
        try:
            return float(str(raw.get(k) or 0))
        except (TypeError, ValueError):
            return 0.0

    hz_total = num("area_debris_km2") + num("area_flood_km2") + num("area_not_observable_km2")
    bars = {
        "debris": _pct(num("area_debris_km2"), hz_total),
        "flood": _pct(num("area_flood_km2"), hz_total),
        "unobs": _pct(num("area_not_observable_km2"), hz_total),
        "bld": _pct(num("buildings_hit"), num("buildings_total")),
        "cut": _pct(num("settlements_cut_off"), num("settlements_total")),
        "hosp": _pct(num("settlements_hospital_cut_off"), num("settlements_total")),
        "detour": _pct(num("settlements_detour"), num("settlements_total")),
        "br_lost": _pct(num("bridges_lost"), num("bridges_total")),
        "br_risk": _pct(num("bridges_at_risk"), num("bridges_total")),
    }
    flags = {
        "unobs": num("area_not_observable_km2") >= 0.01,
        "cut": num("settlements_cut_off") > 0,
        "any_access": num("settlements_cut_off")
        + num("settlements_hospital_cut_off")
        + num("settlements_detour")
        > 0,
    }
    texts = {k: v.text for k, v in f.items.items()}
    tables = f.tables
    if lang == "ne":
        texts = {k: v.translate(DEVANAGARI) for k, v in texts.items()}
        tables = {
            name: [
                {
                    k: str(v)
                    if k.endswith("_url") or k.startswith("url")
                    else str(v).translate(DEVANAGARI)
                    for k, v in row.items()
                }
                for row in rows
            ]
            for name, rows in tables.items()
        }
    members = {
        "s1_water_logratio": ("Radar change: new water", "राडार परिवर्तन: नयाँ पानी"),
        "s1_change_bidirectional": (
            "Radar change: debris and scour",
            "राडार परिवर्तन: लेदो तथा कटान",
        ),
        "ml_unet_kurosiwo": ("Segmentation model (Kuro Siwo)", "खण्डीकरण मोडेल (Kuro Siwo)"),
        "s2_indices": ("Optical indices", "अप्टिकल सूचक"),
    }
    member_names = [
        members.get(m.strip(), (m.strip(), m.strip()))[1 if lang == "ne" else 0]
        for m in str(raw.get("members") or "").split(",")
        if m.strip()
    ]
    flow = [
        {"meta": tables["flowpaths"][i], "rows": tables.get(f"flow_{i}", [])}
        for i in range(len(tables.get("flowpaths", [])))
    ]
    m = maps if maps is not None else _maps(job_dir, f)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    page = env.get_template("sitrep.html.j2").render(
        lang=lang,
        other="ne" if lang == "en" else "en",
        L=LABELS[lang],
        F=texts,
        T=tables,
        bars=bars,
        flags=flags,
        flow=flow,
        maps=m,
        member_names=member_names,
    )
    report = verify(page, f)
    return page, report


def write(job_dir: Path) -> dict:
    """Render, verify and write sitrep_<lang>.html (+ .pdf when WeasyPrint is available)."""
    out: dict = {}
    f = build(job_dir)
    (job_dir / "facts.json").write_text(json.dumps(f.to_json(), indent=1, default=str))
    m = _maps(job_dir, f)
    for lang in ("en", "ne"):
        page, rep = render(job_dir, lang, f, m)
        (job_dir / f"sitrep_{lang}.html").write_text(page)
        out[lang] = {"html": f"sitrep_{lang}.html", "verification": rep}
        try:
            weasyprint = importlib.import_module("weasyprint")
            buf = io.BytesIO()
            weasyprint.HTML(string=page).write_pdf(buf)
            (job_dir / f"sitrep_{lang}.pdf").write_bytes(buf.getvalue())
            out[lang]["pdf"] = f"sitrep_{lang}.pdf"
        except Exception:
            out[lang]["pdf"] = None
    return out
