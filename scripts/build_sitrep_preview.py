"""Generates an interactive HTML review workbench for Dhara Situation Reports.

Features:
- SEVERE event (6f18034fa3c4) vs HIGH event (263b9b12ede9)
- Variation 1 (Dense Analytical Dashboard) vs Variation 2 (Operational Command Grid)
- Side-by-side comparison mode
- English and Nepali language switcher
- Section ID overlay toggles (#1 to #9) for precise user feedback
- True A4 sheet scale vs Fluid responsive web layout
"""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from dhara.sitrep.facts import DEVANAGARI, build
from dhara.sitrep.render import ATTR_DEM, ATTR_OSM, ATTR_S, LABELS, TEMPLATES, _maps, _pct

JOBS = {
    "severe": {
        "id": "6f18034fa3c4",
        "label": "🔴 SEVERE Event (6f18034fa3c4 · Aug 28)",
        "subtitle": "2 communities isolated · 29 hospital routes cut · 2 bridges lost · 254 buildings hit",
        "tag": "SEVERE",
    },
    "high": {
        "id": "263b9b12ede9",
        "label": "🟠 HIGH Event (263b9b12ede9 · Aug 02)",
        "subtitle": "29 hospital routes cut · 0 cut off · 18 buildings hit",
        "tag": "HIGH",
    },
}


def render_variant(job_dir: Path, template_name: str, lang: str = "en") -> str:
    f = build(job_dir)
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
    m = _maps(job_dir, f)
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    return env.get_template(template_name).render(
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


def extract_body(full_html: str) -> tuple[str, str]:
    """Extracts styles and section.sheet content so it can be isolated inside a component or iframe."""
    import re

    # Extract style block
    style_m = re.search(r"<style>(.*?)</style>", full_html, re.DOTALL)
    style_css = style_m.group(1) if style_m else ""
    # Extract sheet section
    sheet_m = re.search(r"(<section class=\"sheet\".*?</section>)", full_html, re.DOTALL)
    sheet_html = sheet_m.group(1) if sheet_m else ""
    return style_css, sheet_html


def build_workbench():
    print("Rendering variants...")
    variants = {}
    for job_key, meta in JOBS.items():
        job_dir = Path("data/jobs") / meta["id"]
        variants[job_key] = {}
        for var_key, tpl in [("var1", "sitrep.html.j2"), ("var2", "sitrep_v2.html.j2")]:
            variants[job_key][var_key] = {}
            for lang in ["en", "ne"]:
                print(f"  Rendering {job_key} · {var_key} · {lang}...")
                full_html = render_variant(job_dir, tpl, lang)
                variants[job_key][var_key][lang] = full_html

    # Build the master HTML workbench
    workbench_template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dhara Sitrep Review Workbench · Variations & Severity Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600;700;800&family=IBM+Plex+Sans+Devanagari:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-app: #090d16;
    --bg-bar: #0f172a;
    --border-bar: #1e293b;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #0284c7;
    --accent-hover: #0369a1;
    --red: #ef4444;
    --orange: #f97316;
    --green: #22c55e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-app);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* Workbench Header & Toolbar */
  header.wb-header {
    background: var(--bg-bar);
    border-bottom: 1px solid var(--border-bar);
    padding: 10px 16px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }
  .wb-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  .wb-title {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .wb-logo {
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: #fff;
    font-size: 11px;
    font-weight: 900;
    padding: 3px 8px;
    border-radius: 4px;
    letter-spacing: .08em;
  }
  .wb-title h1 {
    font-size: 14px;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.01em;
  }
  .wb-title span {
    font-size: 11px;
    color: var(--text-muted);
  }
  .wb-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* Control Groups */
  .wb-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
  }
  .ctrl-group {
    display: flex;
    align-items: center;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 2px;
  }
  .ctrl-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: #64748b;
    padding: 0 8px;
  }
  .btn-toggle {
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
    padding: 5px 11px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }
  .btn-toggle:hover {
    color: #f8fafc;
    background: rgba(255,255,255,0.05);
  }
  .btn-toggle.active {
    background: #0284c7;
    color: #fff;
    font-weight: 700;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }
  .btn-toggle.active.severe {
    background: #dc2626;
  }
  .btn-toggle.active.high {
    background: #ea580c;
  }
  .btn-action {
    background: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
    font-size: 11px;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 5px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    transition: all 0.15s ease;
  }
  .btn-action:hover {
    background: #334155;
    color: #fff;
  }
  .btn-action.active-flag {
    background: #0284c7;
    border-color: #38bdf8;
    color: #fff;
  }

  /* Info & Feedback Prompt Drawer */
  .feedback-banner {
    background: rgba(2, 132, 199, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 6px;
    padding: 8px 14px;
    margin: 10px 16px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: #bae6fd;
  }
  .feedback-banner b { color: #fff; }
  .section-tags-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
  }
  .sec-tag {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #e0f2fe;
    font-size: 9.5px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 3px;
  }

  /* Canvas Workspace */
  main.wb-canvas {
    flex: 1;
    padding: 16px;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    overflow: auto;
    background: #0b1120;
  }
  .view-container {
    display: flex;
    gap: 20px;
    justify-content: center;
    width: 100%;
    transition: all 0.2s ease;
  }
  .view-pane {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .pane-header {
    width: 100%;
    max-width: 210mm;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 11px;
    color: #94a3b8;
  }
  .pane-header b {
    color: #f8fafc;
    font-size: 12px;
  }
  .pane-badge {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 10px;
    color: #38bdf8;
    font-weight: 700;
  }

  /* Frame Styling */
  .frame-wrapper {
    background: #fff;
    border-radius: 4px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    overflow: hidden;
    position: relative;
    transition: all 0.2s ease;
  }
  .mode-a4 .frame-wrapper {
    width: 210mm;
    height: 297mm;
    min-height: 297mm;
    max-height: 297mm;
  }
  .mode-fluid .frame-wrapper {
    width: 100%;
    max-width: 1050px;
    height: auto;
    min-height: 800px;
  }
  .side-by-side .frame-wrapper {
    width: 185mm;
    height: 261mm;
    zoom: 0.88;
  }

  iframe.sitrep-frame {
    width: 100%;
    height: 100%;
    border: none;
    display: block;
    background: #fff;
  }

  /* Overlay Section Marker inside iframe */
  .inspect-active .inspect-tag {
    display: inline-block !important;
  }
</style>
</head>
<body>

<header class="wb-header">
  <div class="wb-top">
    <div class="wb-title">
      <span class="wb-logo">DHARA</span>
      <div>
        <h1>Situation Report Review Workbench</h1>
        <span>Interactive comparison of report variations across <b>Severe</b> and <b>High</b> events</span>
      </div>
    </div>
    <div class="wb-actions">
      <button class="btn-action" id="btnInspect" onclick="toggleInspect()" title="Show numbered badges on each section for easy reference">
        🏷️ Section Markers: <span id="inspectStatus">OFF</span>
      </button>
      <button class="btn-action" id="btnPageMode" onclick="togglePageMode()" title="Toggle between exact A4 page dimensions and fluid screen width">
        📄 Mode: <span id="pageModeStatus">A4 Sheet</span>
      </button>
      <button class="btn-action" onclick="printActiveFrame()">
        🖨️ Print View
      </button>
    </div>
  </div>

  <div class="wb-controls">
    <!-- Event Selector -->
    <div class="ctrl-group">
      <span class="ctrl-label">Event / Severity</span>
      <button class="btn-toggle active severe" id="btnJobSevere" onclick="setJob('severe')">
        🔴 SEVERE (Aug 28 · 2 communities cut off · 2 bridges lost)
      </button>
      <button class="btn-toggle" id="btnJobHigh" onclick="setJob('high')">
        🟠 HIGH (Aug 02 · 29 hospital routes cut · 0 cut off)
      </button>
    </div>

    <!-- Variation Selector -->
    <div class="ctrl-group">
      <span class="ctrl-label">Layout Variation</span>
      <button class="btn-toggle active" id="btnVar1" onclick="setVariant('var1')">
        Variation 1: Analytical Dashboard
      </button>
      <button class="btn-toggle" id="btnVar2" onclick="setVariant('var2')">
        Variation 2: Operational Command Grid
      </button>
      <button class="btn-toggle" id="btnVarCompare" onclick="setVariant('compare')">
        ⚖️ Side-by-Side Compare
      </button>
    </div>

    <!-- Language Selector -->
    <div class="ctrl-group">
      <span class="ctrl-label">Language</span>
      <button class="btn-toggle active" id="btnLangEn" onclick="setLang('en')">English</button>
      <button class="btn-toggle" id="btnLangNe" onclick="setLang('ne')">नेपाली</button>
    </div>
  </div>
</header>

<!-- Review Cheat Sheet Guide -->
<div class="feedback-banner">
  <div>
    <b>How to give feedback:</b> Turn on <b>Section Markers</b> to display section tags on each card. You can tell me changes like: <i>"In #2-KPIS, change metric X"</i>, <i>"In #4-MAP, adjust legend"</i>, or <i>"In #6-CUTOFF, change table columns"</i>.
    <div class="section-tags-list">
      <span class="sec-tag">#1-HEADER</span>
      <span class="sec-tag">#2-KPIS</span>
      <span class="sec-tag">#3-BOTTOM-LINE</span>
      <span class="sec-tag">#4-MAP-IMAGERY</span>
      <span class="sec-tag">#5-HAZARD-EVIDENCE</span>
      <span class="sec-tag">#6-ISOLATION-ACCESS</span>
      <span class="sec-tag">#7-BRIDGES</span>
      <span class="sec-tag">#8-PRIORITY-REPAIRS</span>
      <span class="sec-tag">#9-PROVENANCE-FOOTER</span>
    </div>
  </div>
</div>

<main class="wb-canvas mode-a4" id="canvasRoot">
  <div class="view-container" id="viewContainer">
    <!-- Left / Single Pane -->
    <div class="view-pane" id="pane1">
      <div class="pane-header">
        <span id="pane1Title"><b>Variation 1: Analytical Dashboard</b></span>
        <span class="pane-badge" id="pane1Badge">NASA Earthdata & Dense KPIs</span>
      </div>
      <div class="frame-wrapper">
        <iframe class="sitrep-frame" id="frame1"></iframe>
      </div>
    </div>

    <!-- Right Pane (Active during side-by-side comparison) -->
    <div class="view-pane" id="pane2" style="display:none;">
      <div class="pane-header">
        <span id="pane2Title"><b>Variation 2: Operational Command Grid</b></span>
        <span class="pane-badge">Sector Divided & High-Contrast</span>
      </div>
      <div class="frame-wrapper">
        <iframe class="sitrep-frame" id="frame2"></iframe>
      </div>
    </div>
  </div>
</main>

<script>
  // Pre-compiled HTML data for instantaneous zero-latency switching
  const DATA = __DATA_PLACEHOLDER__;

  let currentJob = 'severe';
  let currentVar = 'var1';
  let currentLang = 'en';
  let inspectMode = false;
  let pageMode = 'a4'; // 'a4' or 'fluid'

  function updateView() {
    const pane1 = document.getElementById('pane1');
    const pane2 = document.getElementById('pane2');
    const pane1Title = document.getElementById('pane1Title');
    const pane1Badge = document.getElementById('pane1Badge');
    const frame1 = document.getElementById('frame1');
    const frame2 = document.getElementById('frame2');
    const container = document.getElementById('viewContainer');

    if (currentVar === 'compare') {
      pane1.style.display = 'flex';
      pane2.style.display = 'flex';
      container.classList.add('side-by-side');

      pane1Title.innerHTML = '<b>Variation 1: Analytical Dashboard</b> (' + (currentJob === 'severe' ? 'SEVERE' : 'HIGH') + ')';
      pane1Badge.textContent = 'Dense KPIs & Questions';

      loadFrame(frame1, DATA[currentJob]['var1'][currentLang]);
      loadFrame(frame2, DATA[currentJob]['var2'][currentLang]);
    } else {
      pane1.style.display = 'flex';
      pane2.style.display = 'none';
      container.classList.remove('side-by-side');

      if (currentVar === 'var1') {
        pane1Title.innerHTML = '<b>Variation 1: Analytical Dashboard</b> (' + (currentJob === 'severe' ? 'SEVERE' : 'HIGH') + ')';
        pane1Badge.textContent = 'Dense KPIs & Standard Header';
        loadFrame(frame1, DATA[currentJob]['var1'][currentLang]);
      } else {
        pane1Title.innerHTML = '<b>Variation 2: Operational Command Grid</b> (' + (currentJob === 'severe' ? 'SEVERE' : 'HIGH') + ')';
        pane1Badge.textContent = 'Command Ribbon & Sector Cards';
        loadFrame(frame1, DATA[currentJob]['var2'][currentLang]);
      }
    }
  }

  function loadFrame(frame, htmlContent) {
    const doc = frame.contentWindow.document;
    doc.open();
    doc.write(htmlContent);
    doc.close();

    // Apply inspect mode if active
    if (inspectMode) {
      setTimeout(() => applyInspectToDoc(doc, true), 30);
    }
  }

  function setJob(job) {
    currentJob = job;
    document.getElementById('btnJobSevere').classList.toggle('active', job === 'severe');
    document.getElementById('btnJobHigh').classList.toggle('active', job === 'high');
    updateView();
  }

  function setVariant(v) {
    currentVar = v;
    document.getElementById('btnVar1').classList.toggle('active', v === 'var1');
    document.getElementById('btnVar2').classList.toggle('active', v === 'var2');
    document.getElementById('btnVarCompare').classList.toggle('active', v === 'compare');
    updateView();
  }

  function setLang(lang) {
    currentLang = lang;
    document.getElementById('btnLangEn').classList.toggle('active', lang === 'en');
    document.getElementById('btnLangNe').classList.toggle('active', lang === 'ne');
    updateView();
  }

  function toggleInspect() {
    inspectMode = !inspectMode;
    document.getElementById('inspectStatus').textContent = inspectMode ? 'ON (Marked)' : 'OFF';
    document.getElementById('btnInspect').classList.toggle('active-flag', inspectMode);

    const f1 = document.getElementById('frame1');
    const f2 = document.getElementById('frame2');
    if (f1 && f1.contentWindow) applyInspectToDoc(f1.contentWindow.document, inspectMode);
    if (f2 && f2.contentWindow) applyInspectToDoc(f2.contentWindow.document, inspectMode);
  }

  function applyInspectToDoc(doc, enable) {
    if (!doc || !doc.body) return;
    if (enable) {
      doc.body.classList.add('inspect-mode');
    } else {
      doc.body.classList.remove('inspect-mode');
    }
  }

  function togglePageMode() {
    const canvas = document.getElementById('canvasRoot');
    if (pageMode === 'a4') {
      pageMode = 'fluid';
      canvas.className = 'wb-canvas mode-fluid';
      document.getElementById('pageModeStatus').textContent = 'Fluid Web';
    } else {
      pageMode = 'a4';
      canvas.className = 'wb-canvas mode-a4';
      document.getElementById('pageModeStatus').textContent = 'A4 Sheet';
    }
  }

  function printActiveFrame() {
    const f1 = document.getElementById('frame1');
    if (f1 && f1.contentWindow) {
      f1.contentWindow.focus();
      f1.contentWindow.print();
    }
  }

  // Initial render
  window.addEventListener('DOMContentLoaded', () => {
    updateView();
  });
</script>
</body>
</html>
"""

    workbench_html = workbench_template.replace("__DATA_PLACEHOLDER__", json.dumps(variants))

    out_file = Path("web/public/sitrep_preview.html")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(workbench_html, encoding="utf-8")
    print(f"Successfully generated workbench at: {out_file.resolve()}")
    print(f"File size: {out_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build_workbench()
