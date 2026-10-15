// Situation report page (/report). Structure follows the OCHA situation report template
// (highlights, key figures, situation overview, sections, sources), map figures follow common
// humanitarian mapping practice (title, legend, sources and date, disclaimer), and text and tables
// follow USWDS guidance (16 px body, 45–90 character lines, captioned tables with header scopes,
// stacked rows on narrow screens). Every number is a computed fact from the job; a missing fact
// is left out, never filled in.
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, ExternalLink } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { api, apiUrl, type SitrepData } from '@/lib/api'
import { jobLabel } from '@/lib/job'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { PdfButton } from './pdf-button'

type Lang = 'en' | 'ne'

/** File name for a saved report, e.g. Dhara_SitRep_2026-08-26_पाल्ते_68d2b5869583 (`_NE` for Nepali).
 *  Same rule as `dhara.sitrep.pdf.report_filename` on the server. */
/** Absolute link for the report's footer. A PDF printed on the server is opened elsewhere, so the
 *  server passes its public address as ?origin= and the dashboard's as ?web= (when hosted apart).
 *  In the browser, API links go to the API and the dashboard link stays relative. */
function publicLink(path: string, to: 'api' | 'web' = 'api'): string {
  const q = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null
  const ok = (v: string | null | undefined) => (v && /^https?:\/\/[^/?#]+$/.test(v) ? v : null)
  const origin = ok(q?.get('origin'))
  if (!origin) return to === 'api' ? apiUrl(path) : path
  return `${to === 'web' ? (ok(q?.get('web')) ?? origin) : origin}${path}`
}

/** Report map image: the server sends data URLs (WebP); older responses sent bare PNG base64. */
const mapSrc = (v: string) => (v.startsWith('data:') ? v : `data:image/png;base64,${v}`)

function reportFileName(facts: Record<string, string>, jobId: string, lang: Lang): string {
  const day = (facts.event ?? '').slice(0, 10)
  const place = (facts.target_location ?? '').split(' (')[0].replace(/[^\p{L}\p{M}\p{N}]+/gu, '-').replace(/^-+|-+$/g, '')
  const where = place || (facts.target_lat && facts.target_lon ? `${facts.target_lat}N-${facts.target_lon}E` : '')
  return ['Dhara_SitRep', day, where, jobId].filter(Boolean).join('_') + (lang === 'ne' ? '_NE' : '')
}
type Row = Record<string, string>

// Severity levels in Nepali (गम्भीर severe, उच्च high, मध्यम moderate, न्यून low).
const LEVEL_NE: Record<string, string> = { SEVERE: 'गम्भीर', HIGH: 'उच्च', MODERATE: 'मध्यम', LOW: 'न्यून' }
const DEVANAGARI = '०१२३४५६७८९'
const neDigits = (s: string) => s.replace(/[0-9]/g, (d) => DEVANAGARI[Number(d)])

const T = {
  en: {
    kind: 'Situation report',
    prototype: 'Educational prototype. Not for operational use.',
    titleTail: 'flood and debris impact',
    asOf: 'As of',
    analysis: 'Analysis',
    highlights: 'Highlights',
    keyFigures: 'Key figures',
    overview: 'Situation overview',
    where: 'Where the flood hit',
    what: 'What was damaged',
    who: 'Who is cut off',
    access: 'Access and isolation',
    damage: 'Damage to buildings, roads and bridges',
    repairs: 'Restoration priorities',
    change: 'What changed between the images',
    imagery: 'Satellite images',
    flow: 'Flood paths traced in this area',
    hospitals: 'Hospitals and clinics',
    method: 'Method, limitations and sources',
    kfCut: 'communities with no road out',
    kfHosp: 'more communities without road access to a hospital',
    kfBld: 'buildings in mapped flood or debris',
    kfBridge: 'bridges likely lost',
    kfRoad: 'km of road and trail impassable',
    of: 'of',
    mapped: 'mapped',
    sources: 'Sources',
    created: 'Created',
    disclaimer: 'Boundaries and names shown do not imply official endorsement or acceptance. Produced automatically from satellite data; not field-verified.',
    print: 'Print or save as PDF',
    download: 'Download PDF',
    back: 'Back to map',
    loading: 'Building the report from the analysis…',
    pickJob: 'Choose an analysis',
    noJobs: 'No finished analysis yet. Run one from the map first.',
    error: 'The report could not be built for this analysis.',
    status: {
      cut: 'Cut off',
      hosp: 'No hospital access',
      lost: 'Likely lost',
      risk: 'At risk',
      ok: 'Accessible',
      iso: 'Isolated',
    },
  },
  ne: {
    kind: 'स्थिति प्रतिवेदन',
    prototype: 'शैक्षिक प्रोटोटाइप। सञ्चालनका लागि होइन।',
    titleTail: 'बाढी तथा लेदोको प्रभाव',
    asOf: 'मिति',
    analysis: 'विश्लेषण',
    highlights: 'मुख्य बुँदा',
    keyFigures: 'मुख्य तथ्याङ्क',
    overview: 'स्थितिको सारांश',
    where: 'बाढी कहाँ पुग्यो',
    what: 'के क्षति भयो',
    who: 'को सम्पर्कविहीन छ',
    access: 'पहुँच तथा सम्पर्कविच्छेद',
    damage: 'भवन, सडक र पुलको क्षति',
    repairs: 'पुनःस्थापनाका प्राथमिकता',
    change: 'दुई तस्बिरबीच के बदलियो',
    imagery: 'उपग्रह तस्बिर',
    flow: 'यस क्षेत्रमा कोरिएका बाढी मार्ग',
    hospitals: 'अस्पताल तथा क्लिनिक',
    method: 'विधि, सीमा तथा स्रोत',
    kfCut: 'बाहिर निस्कने सडक नभएका बस्ती',
    kfHosp: 'थप बस्तीको अस्पतालसम्म सडक पहुँच छैन',
    kfBld: 'बाढी वा लेदो क्षेत्रका भवन',
    kfBridge: 'सम्भवतः बगेका पुल',
    kfRoad: 'कि.मि. सडक तथा गोरेटो अवरुद्ध',
    of: 'मध्ये',
    mapped: 'नक्साङ्कित',
    sources: 'स्रोत',
    created: 'तयार',
    disclaimer: 'नक्सामा देखाइएका सिमाना र नामले आधिकारिक मान्यता जनाउँदैनन्। उपग्रह तथ्याङ्कबाट स्वचालित रूपमा तयार; स्थलगत रूपमा पुष्टि गरिएको छैन।',
    print: 'प्रिन्ट वा PDF',
    download: 'PDF डाउनलोड',
    back: 'नक्सामा फर्कनुहोस्',
    loading: 'विश्लेषणबाट प्रतिवेदन तयार हुँदैछ…',
    pickJob: 'विश्लेषण छान्नुहोस्',
    noJobs: 'अहिलेसम्म कुनै विश्लेषण पूरा भएको छैन।',
    error: 'यो विश्लेषणको प्रतिवेदन बन्न सकेन।',
    status: {
      cut: 'सम्पर्कविहीन',
      hosp: 'अस्पताल पहुँच छैन',
      lost: 'सम्भवतः बगेको',
      risk: 'जोखिममा',
      ok: 'पहुँचयोग्य',
      iso: 'सम्पर्कविहीन',
    },
  },
}

/** Fact text, with Devanagari digits in Nepali; null when the fact is missing. */
function useFacts(data: SitrepData | undefined, lang: Lang) {
  return useMemo(() => {
    const facts = data?.facts ?? {}
    const values = data?.values ?? {}
    const t = (k: string): string | null => {
      const v = facts[k]
      if (v == null || v === '') return null
      return lang === 'ne' ? neDigits(v) : v
    }
    const n = (k: string): number | null => {
      const v = values[k]
      const x = typeof v === 'number' ? v : Number(String(v ?? '').replace(/,/g, ''))
      return v == null || v === '' || Number.isNaN(x) ? null : x
    }
    return { t, n, digits: (s: string) => (lang === 'ne' ? neDigits(s) : s) }
  }, [data, lang])
}

const MONTHS_NE = ['जनवरी', 'फेब्रुअरी', 'मार्च', 'अप्रिल', 'मे', 'जुन', 'जुलाई', 'अगस्ट', 'सेप्टेम्बर', 'अक्टोबर', 'नोभेम्बर', 'डिसेम्बर']

/** A UTC date as "26 Aug 2026", or in Nepali "२६ अगस्ट २०२६" (Gregorian dates, Nepali month names). */
const fmtDay = (iso: string | null | undefined, lang: Lang = 'en') => {
  if (!iso) return null
  const d = new Date(iso.length <= 10 ? `${iso}T00:00:00Z` : iso)
  if (Number.isNaN(d.getTime())) return null
  if (lang === 'ne') return neDigits(`${d.getUTCDate()} ${MONTHS_NE[d.getUTCMonth()]} ${d.getUTCFullYear()}`)
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
}

// OSM place and road types, and pass directions, for the Nepali report. Unknown values stay as they are.
const PLACE_NE: Record<string, string> = { hamlet: 'टोल', locality: 'स्थान', village: 'गाउँ', town: 'सहर', city: 'सहर', isolated_dwelling: 'एकल बस्ती', suburb: 'उपनगर', neighbourhood: 'टोल' }
const ROAD_NE: Record<string, string> = {
  path: 'गोरेटो',
  footway: 'पैदल मार्ग',
  track: 'कच्ची बाटो',
  unclassified: 'साना सडक',
  residential: 'आवासीय सडक',
  service: 'सेवा सडक',
  tertiary: 'स्थानीय सडक',
  secondary: 'सहायक सडक',
  primary: 'मुख्य सडक',
  trunk: 'राजमार्ग',
  roadway: 'सडक',
}
const placeLabel = (v: string | undefined, lang: Lang) => {
  const k = String(v ?? '')
  return lang === 'ne' ? (PLACE_NE[k] ?? k.replace(/_/g, ' ')) : k.replace(/_/g, ' ')
}
const roadLabel = (v: string | undefined, lang: Lang) => {
  const k = String(v ?? '')
  return lang === 'ne' ? (ROAD_NE[k] ?? k.replace(/_/g, ' ')) : k.replace(/_/g, ' ')
}
/** "Bridge …" / "Road …" from the repair ranking, with the kind word in Nepali when needed. */
const assetLabel = (v: string | undefined, lang: Lang) => {
  const k = String(v ?? '')
  return lang === 'ne' ? k.replace(/^Bridge\b/, 'पुल').replace(/^Road\b/, 'सडक') : k
}

// ------------------------------------------------------------------------------- building blocks

type Col = {
  key: string
  label: string
  num?: boolean
  render?: (r: Row) => ReactNode
}

/** USWDS-style table: visible caption, header scopes, right-aligned tabular numbers, striped rows,
 *  stacked label/value rows on narrow screens (data-label). */
function DataTable({ caption, cols, rows, note, ne = false }: { caption: string; cols: Col[]; rows: Row[]; note?: string; ne?: boolean }) {
  if (rows.length === 0) return null
  return (
    <div className='sr-tablewrap'>
      <table className='sr-table'>
        <caption>{caption}</caption>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c.key} scope='col' className={c.num ? 'num' : ''}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c, j) =>
                j === 0 ? (
                  <th key={c.key} scope='row' data-label={c.label}>
                    {c.render ? c.render(r) : ne && c.num ? neDigits(String(r[c.key] ?? '')) : r[c.key]}
                  </th>
                ) : (
                  <td key={c.key} data-label={c.label} className={c.num ? 'num' : ''}>
                    {c.render ? c.render(r) : ne && c.num ? neDigits(String(r[c.key] ?? '')) : r[c.key]}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
      {note && <p className='sr-source'>{note}</p>}
    </div>
  )
}

/** Status shown as text with a shape marker, so it does not rely on colour alone. */
function Status({ tone, children }: { tone: 'bad' | 'warn' | 'ok'; children: ReactNode }) {
  const c = tone === 'bad' ? 'var(--sr-bad)' : tone === 'warn' ? 'var(--sr-warn)' : 'var(--sr-ok)'
  return (
    <span className='inline-flex items-center gap-1.5 font-semibold whitespace-nowrap' style={{ color: c }}>
      <span
        aria-hidden
        style={{
          width: 8,
          height: 8,
          background: c,
          borderRadius: tone === 'ok' ? 8 : 1,
          transform: tone === 'warn' ? 'rotate(45deg)' : undefined,
        }}
      />
      {children}
    </span>
  )
}

const MapLink = ({ href, label = 'Map' }: { href?: string; label?: string }) =>
  href ? (
    <a href={href} target='_blank' rel='noreferrer' className='sr-link whitespace-nowrap'>
      {label} <ExternalLink className='inline size-3' aria-hidden />
      <span className='sr-only'> (opens Google Maps in a new tab)</span>
    </a>
  ) : null

// Section names and labels as specified for this report (the version staged in git).
const LABELS = {
  en: {
    brand: 'DHARA',
    subBrand: 'SITUATION REPORT',
    sitrepNo: 'NO. 1',
    prototype: 'Educational prototype. Not for operational use.',
    title: 'Flood & Debris Situation Report',
    target: 'Location',
    event: 'Event',
    area: 'Area',
    generated: 'Issued',
    kpis: '1. Key figures',
    hazard: 'Area hit',
    cutOff: 'Cut off',
    noHospital: 'No hospital access',
    bridges: 'Bridges lost',
    buildings: 'Buildings hit',
    detours: 'Longer route only',
    secAssessment: '2. Situation',
    secExtent: '3. Situation map',
    secSatellite: '4. Before and after',
    secCutoff: '5. Cut-off communities',
    secBridges: '6. Bridges',
    secRepairs: '7. Reopen first',
    dataUsed: 'Data',
    limits: 'Limitations',
    credits: 'Credits',
    links: 'Links',
    qWhere: 'Where did the flood hit?',
    qWhat: 'What was damaged?',
    qWho: 'Who is cut off?',
    qRecon: 'What to reopen first?',
    how: 'How we estimate',
    severity: 'Severity',
    mapDisclaimer: 'Boundaries and names shown do not imply official endorsement. Map data: OpenStreetMap (pre-event); images: Copernicus Sentinel-1/2.',
  },
  ne: {
    brand: 'DHARA',
    subBrand: 'स्थिति प्रतिवेदन',
    sitrepNo: 'नं. १',
    prototype: 'शैक्षिक प्रोटोटाइप। सञ्चालनका लागि होइन।',
    title: 'बाढी तथा लेदो स्थिति प्रतिवेदन',
    target: 'स्थान',
    event: 'घटना',
    area: 'क्षेत्र',
    generated: 'जारी',
    kpis: '१. मुख्य तथ्याङ्क',
    hazard: 'प्रभावित क्षेत्र',
    cutOff: 'सम्पर्कविहीन',
    noHospital: 'अस्पताल पहुँच छैन',
    bridges: 'बगेका पुल',
    buildings: 'प्रभावित भवन',
    detours: 'लामो बाटो मात्र',
    secAssessment: '२. स्थिति',
    secExtent: '३. स्थिति नक्सा',
    secSatellite: '४. पहिले र पछि',
    secCutoff: '५. सम्पर्कविहीन बस्ती',
    secBridges: '६. पुल',
    secRepairs: '७. पहिले खोल्ने',
    dataUsed: 'तथ्याङ्क',
    limits: 'सीमा',
    credits: 'आभार',
    links: 'लिङ्क',
    qWhere: 'बाढी कहाँ पुग्यो?',
    qWhat: 'के क्षति भयो?',
    qWho: 'को सम्पर्कविहीन छ?',
    qRecon: 'पहिले के खोल्ने?',
    how: 'कसरी अनुमान गरियो',
    severity: 'गम्भीरता',
    mapDisclaimer: 'नक्सामा देखाइएका सिमाना र नामले आधिकारिक मान्यता जनाउँदैनन्। नक्सा: ओपनस्ट्रिटम्याप (घटनाअघि); तस्बिर: कोपर्निकस सेन्टिनेल-१/२।',
  },
}

// ----------------------------------------------------------------------------------- the report

/** The one-page situation report (Track B: "a one-page situation report"), with the sections
 *  specified for it; every number is a computed fact, and a missing fact is shown as a dash. */
function Report({ data, lang }: { data: SitrepData; lang: Lang }) {
  const L = LABELS[lang]
  const U = T[lang]
  const { t, n, digits } = useFacts(data, lang)
  const tables = data.tables as Record<string, Row[] | undefined>
  const maps = data.maps
  const en = lang === 'en'
  const dash = '—'
  const loc = data.facts.target_location ?? null
  const place = loc ? loc.replace(/\s*\([^)]*\)\s*$/, '') : null
  const day = (iso: string | null | undefined) => fmtDay(iso, lang)
  const created = day(data.facts.generated_utc)
  const osm = day(data.facts.osm_snapshot)
  const s1 = tables.s1 ?? []
  const track = (tr: string) => (en ? tr : tr.replace('ascending', 'आरोही').replace('descending', 'अवरोही'))
  const s1Text = s1.map((r) => `${track(r.track)} (${day(r.pre)} → ${day(r.post)})`).join(' · ')
  const s2Text = data.facts.s2_pair
    ? data.facts.s2_pair
        .split(' → ')
        .map((d) => day(d) ?? d)
        .join(' → ')
    : null
  const level = data.facts.verdict_level ?? null

  const cut = n('settlements_cut_off')
  const hasCutoff = (cut ?? 0) > 0
  const repair = (tables.repairs ?? [])[0]
  const statusOf = (st: string) => (st.toLowerCase().includes('cut off') && !st.toLowerCase().includes('hospital') ? 'cut' : 'hosp')
  const bridgeName = (r: Row) => {
    const nm = r.name ?? ''
    const generic = ['primary', 'secondary', 'tertiary', 'trunk', 'unclassified', 'residential', 'track', 'path', 'footway', 'service', '']
    if (!generic.includes(nm.toLowerCase())) return nm
    return en ? `Unnamed${nm ? ` (${nm})` : ''}` : `नाम नभएको${nm ? ` (${roadLabel(nm, lang)})` : ''}`
  }

  const kpis: {
    value: string | null
    unit?: string
    label: string
    sub: string | null
    tone?: 'bad' | 'warn'
  }[] = [
    {
      value: t('area_debris_km2'),
      unit: 'km²',
      label: L.hazard,
      sub: en
        ? `${t('area_flood_km2') ?? dash} km² new water · ${t('area_not_observable_km2') ?? dash} km² not observable`
        : `${t('area_flood_km2') ?? dash} वर्ग कि.मि. नयाँ पानी`,
    },
    {
      value: hasCutoff ? t('settlements_cut_off') : t('settlements_hospital_cut_off'),
      label: hasCutoff ? L.cutOff : L.noHospital,
      sub: hasCutoff
        ? en
          ? `+${t('settlements_hospital_cut_off') ?? dash} without hospital access`
          : `थप ${t('settlements_hospital_cut_off') ?? dash} अस्पताल पहुँचविहीन`
        : en
          ? `of ${t('settlements_total') ?? dash} mapped settlements`
          : `${t('settlements_total') ?? dash} नक्साङ्कित बस्तीमध्ये`,
      tone: 'bad',
    },
    {
      value: t('bridges_lost'),
      label: L.bridges,
      sub: en
        ? `of ${t('bridges_total') ?? dash} checked · ${t('bridges_at_risk') ?? dash} at risk`
        : `${t('bridges_total') ?? dash} मध्ये · ${t('bridges_at_risk') ?? dash} जोखिममा`,
    },
    {
      value: t('buildings_hit'),
      label: L.buildings,
      sub: en
        ? `of ${t('buildings_total') ?? dash} mapped${t('buildings_in_cut_off') ? ` · ${t('buildings_in_cut_off')} near cut-off communities` : ''}`
        : `${t('buildings_total') ?? dash} नक्साङ्कितमध्ये`,
      tone: 'warn',
    },
    {
      value: t('settlements_detour'),
      label: L.detours,
      sub: t('max_detour_hours') ? (en ? `longest extra time ${t('max_detour_hours')} h` : `बढीमा ${t('max_detour_hours')} घण्टा थप`) : null,
    },
  ]

  return (
    <article lang={lang} className='sr-doc'>
      {/* 1 header and metadata */}
      {/* not <header>/<main>: the dashboard's global print CSS hides those elements */}
      <div className='sr-header'>
        <div className='sr-kicker'>
          <span>
            <strong className='sr-brand'>{L.brand}</strong> · {L.subBrand} {L.sitrepNo}
          </span>
          {level && (
            <span className='sr-kicker-right'>
              {L.severity}: <span className={`sr-level sr-level-${level.toLowerCase()}`}>{lang === 'ne' ? (LEVEL_NE[level] ?? level) : level}</span>
            </span>
          )}
        </div>
        <div className='sr-titlerow'>
          <h1 className='sr-h1' title={place ? `${L.title} · ${place}` : L.title}>
            {L.title}
            {place && <span className='sr-h1-place'> · {place}</span>}
          </h1>
          <span className='sr-ref'>
            {en ? 'Analysis' : 'विश्लेषण'} <span className='font-mono'>{data.job_id}</span>
          </span>
        </div>
        <dl className='sr-meta'>
          {(
            [
              [L.target, loc ? `${loc}, ${en ? 'Nepal' : 'नेपाल'}` : null],
              [L.event, day(data.facts.event)],
              [L.area, t('aoi_area_km2') ? `${t('aoi_area_km2')} km²` : null],
              [L.generated, created],
              [en ? 'Baseline OSM' : 'आधार OSM', osm],
              [en ? 'Grid' : 'ग्रिड', t('resolution_m') ? `${t('resolution_m')} m` : null],
            ] as [string, string | null][]
          )
            .filter(([, v]) => v)
            .map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{digits(v ?? '')}</dd>
              </div>
            ))}
        </dl>
        {data.facts.verdict_headline && (
          <p className={`sr-verdict sr-verdict-${(level ?? '').toLowerCase()}`}>{lang === 'ne' ? neDigits(data.facts.verdict_headline_ne ?? data.facts.verdict_headline) : data.facts.verdict_headline}</p>
        )}
      </div>

      {/* 2 key metrics */}
      <section aria-labelledby='sr-kpis' className='sr-box'>
        <h2 id='sr-kpis' className='sr-h2'>
          {L.kpis}
        </h2>
        <ul className='sr-kpis'>
          {kpis.map((k) => (
            <li key={k.label}>
              <span className={`sr-kpi-num ${k.tone ? `sr-${k.tone}` : ''}`}>
                {k.value ?? dash}
                {k.value && k.unit ? <span className='sr-kpi-unit'> {k.unit}</span> : null}
              </span>
              <span className='sr-kpi-label'>{k.label}</span>
              {k.sub && <span className='sr-kpi-sub'>{k.sub}</span>}
            </li>
          ))}
        </ul>
        {t('change_width_before_m') && (
          <div className='sr-chg-row'>
            <span className='sr-chg-head'>
              {en ? 'What changed' : 'के बदलियो'}
              <span className='sr-chg-sub'>{en ? ` clear-sky optical, ${t('change_clear_pct')}% of area` : ` सफा आकाश, ${t('change_clear_pct')}% क्षेत्र`}</span>
            </span>
            <ul className='sr-change'>
              <li>
                <span className='sr-chg-label'>{en ? 'River width' : 'नदीको चौडाइ'}</span>
                <span className='sr-chg-val'>
                  {t('change_width_before_m')} → <b>{t('change_width_after_m')} m</b> <span className='sr-chg-delta'>{t('change_width_delta_m')} m</span>
                </span>
                <span className='sr-chg-sub'>
                  {en ? `${t('change_stations')} points · ${t('change_widened')} widened ≥ ${t('change_widened_threshold_m')} m` : `${t('change_stations')} बिन्दु`}
                </span>
              </li>
              {t('change_shift_max_m') && (
                <li>
                  <span className='sr-chg-label'>{en ? 'River moved sideways' : 'नदी सरेको'}</span>
                  <span className='sr-chg-val'>
                    <b>{t('change_shift_max_m')} m</b> {en ? 'at most' : 'बढीमा'}
                  </span>
                  <span className='sr-chg-sub'>{en ? `median ${t('change_shift_median_m')} m` : `मध्यिका ${t('change_shift_median_m')} मि.`}</span>
                </li>
              )}
              <li>
                <span className='sr-chg-label'>{en ? 'Open water' : 'खुला पानी'}</span>
                <span className='sr-chg-val'>
                  {t('change_water_before_km2')} → <b>{t('change_water_after_km2')} km²</b> <span className='sr-chg-delta'>{t('change_water_pct') ?? ''}</span>
                </span>
                <span className='sr-chg-sub'>{en ? 'water surface seen from space' : 'अन्तरिक्षबाट देखिने पानी'}</span>
              </li>
              <li>
                <span className='sr-chg-label'>{en ? 'Bare channel' : 'खाली च्यानल'}</span>
                <span className='sr-chg-val'>
                  {t('change_channel_before_km2')} → <b>{t('change_channel_after_km2')} km²</b> <span className='sr-chg-delta'>{t('change_channel_pct') ?? ''}</span>
                </span>
                <span className='sr-chg-sub'>{en ? 'water, mud and fresh sediment, no plants' : 'पानी, लेदो र बालुवा'}</span>
              </li>
            </ul>
          </div>
        )}
      </section>

      {/* 3 situation overview */}
      <section aria-labelledby='sr-assess' className='sr-box'>
        <h2 id='sr-assess' className='sr-h2'>
          {L.secAssessment}
        </h2>
        <div className='sr-assess'>
          <p>
            <strong className='sr-q'>{L.qWhere}</strong>
            {en ? (
              <>
                Comparing images from before and after {day(data.facts.event) ?? 'the event'}, we estimate <b>{t('area_debris_km2') ?? dash} km²</b> of debris and scour and{' '}
                <b>{t('area_flood_km2') ?? dash} km²</b> of new water in {t('aoi_area_km2') ?? dash} km². Terrain hid {t('area_not_observable_km2') ?? dash} km² from the radar:
                unknown, not safe.
              </>
            ) : (
              <>
                उपग्रह तस्बिरको तुलनाले <b>{t('area_debris_km2') ?? dash} वर्ग कि.मि.</b> लेदो तथा कटान र <b>{t('area_flood_km2') ?? dash} वर्ग कि.मि.</b> नयाँ पानी भएको अनुमान छ।
                राडारले नदेखेको {t('area_not_observable_km2') ?? dash} वर्ग कि.मि. क्षेत्र सुरक्षित भन्न सकिँदैन।
              </>
            )}
            <span className='sr-how'>
              {L.how}:{' '}
              {en
                ? 'radar change between the before and after passes, stronger than normal pass-to-pass change, backed by optical plant loss where clear; low ground beside the mapped flow and below its level is filled in at low confidence.'
                : 'घटनाअघि र पछिका राडार तस्बिरबीचको परिवर्तन र सफा आकाशमा अप्टिकल प्रमाण; देखिएको बहावछेउको तल्लो जमिन कम विश्वासका साथ थपिन्छ।'}
            </span>
          </p>
          <p>
            <strong className='sr-q'>{L.qWhat}</strong>
            {en ? (
              <>
                <b>{t('buildings_hit') ?? dash}</b> mapped buildings are likely hit. About <b>{t('network_failed_km') ?? dash} km</b> of road and trail (
                {t('road_ways_hit') ?? dash} segments) is likely impassable. Of {t('bridges_total') ?? dash} mapped bridges, <b>{t('bridges_lost') ?? dash}</b> are likely lost and{' '}
                <b>{t('bridges_at_risk') ?? dash}</b> possibly at risk.
              </>
            ) : (
              <>
                <b>{t('buildings_hit') ?? dash}</b> भवन बाढी वा लेदो क्षेत्रमा परेको अनुमान छ। <b>{t('network_failed_km') ?? dash} कि.मि.</b> सडक अवरुद्ध भएको हुनसक्छ।{' '}
                {t('bridges_total') ?? dash} पुलमध्ये <b>{t('bridges_lost') ?? dash}</b> सम्भवतः बगेका छन्।
              </>
            )}
            <span className='sr-how'>
              {L.how}:{' '}
              {en
                ? 'a building, road or bridge counts as hit when it overlaps mapped flood or debris; bridges by radar change over the span.'
                : 'मापित बाढी वा लेदो क्षेत्रमा पर्ने भवन, सडक वा पुल प्रभावित मानिन्छ।'}
            </span>
          </p>
          <p>
            <strong className='sr-q'>{L.qWho}</strong>
            {en ? (
              <>
                <b>{t('settlements_cut_off') ?? dash}</b> {t('settlements_cut_off') === '1' ? 'community is' : 'communities are'} likely cut off by road and{' '}
                <b>{t('settlements_hospital_cut_off') ?? dash}</b> more likely cannot reach a hospital
                {t('hospitals_total') ? ` (${t('hospitals_isolated') ?? dash} of ${t('hospitals_total')} health facilities likely isolated)` : ''}.{' '}
                <b>{t('settlements_detour') ?? dash}</b> {t('settlements_detour') === '1' ? 'community' : 'communities'} likely need a longer route
                {t('max_detour_hours') ? `, up to ${t('max_detour_hours')} h extra` : ''}.
              </>
            ) : (
              <>
                <b>{t('settlements_cut_off') ?? dash}</b> बस्तीको सबै सडक बन्द भएको हुनसक्छ र थप <b>{t('settlements_hospital_cut_off') ?? dash}</b> बस्तीबाट अस्पताल पुग्न नसकिने
                अनुमान छ।
              </>
            )}
            <span className='sr-how'>
              {L.how}:{' '}
              {en
                ? 'routes on the pre-event road map are recomputed without impassable sections; cut off means no route to a town or hospital.'
                : 'अवरुद्ध खण्ड हटाएर घटनाअघिको सडक सञ्जालमा बाटो पुनः गणना गरिन्छ।'}
            </span>
          </p>
          <p>
            <strong className='sr-q'>{L.qRecon}</strong>
            {repair ? (
              en ? (
                <>
                  Reopening <b>{repair.what}</b> ({repair.highway}) first would likely restore road access to <b>{repair.settlements}</b>{' '}
                  {Number(repair.settlements) === 1 ? 'community' : 'communities'} and <b>{repair.buildings}</b>{' '}
                  {Number(repair.buildings) === 1 ? 'building' : 'buildings'}, more than any other single section.
                </>
              ) : (
                <>
                  <b>{assetLabel(repair.what, lang)}</b> ({roadLabel(repair.highway, lang)}) पहिले खोल्दा <b>{digits(repair.settlements)}</b> बस्ती र <b>{digits(repair.buildings)}</b> भवनको सडक पहुँच फर्कने अनुमान
                  छ।
                </>
              )
            ) : en ? (
              <>No single road section would reconnect isolated communities on its own.</>
            ) : (
              <>एउटै सडक खण्डले मात्र सम्पर्क फर्काउने देखिँदैन।</>
            )}
            <span className='sr-how'>
              {L.how}:{' '}
              {en
                ? 'each impassable section is ranked by how many communities and buildings regain a route if it alone reopens.'
                : 'प्रत्येक खण्ड एक्लै खोल्दा कति बस्ती जोडिन्छन् भनेर क्रम मिलाइएको।'}
            </span>
          </p>
        </div>
      </section>

      {/* maps and tables in two independent columns, so neither column leaves a gap */}
      <div className='sr-visual'>
        <div className='sr-stack'>
          <section aria-labelledby='sr-extent' className='sr-box'>
            <h2 id='sr-extent' className='sr-h2'>
              <span>
                <span className='sr-maptag'>{en ? 'Map 1' : 'नक्सा १'}</span> {L.secExtent}
              </span>
            </h2>
            {maps.overview && (
              <img
                src={mapSrc(maps.overview)}
                alt={en ? 'Map of the analysed area showing mapped debris and new water, impassable roads, settlements by road access and hospitals.' : 'विश्लेषित क्षेत्रको नक्सा'}
                className='sr-map'
              />
            )}
            <ul className='sr-legend' aria-label='Legend'>
              {(
                [
                  ['#c27803', en ? 'Debris & scour' : 'लेदो तथा कटान', 'fill'],
                  ['#1d4ed8', en ? 'New water' : 'नयाँ पानी', 'fill'],
                  ['#dc2626', en ? 'Impassable road' : 'अवरुद्ध सडक', 'line'],
                  ['#b91c1c', en ? 'Cut off' : 'सम्पर्कविहीन', 'dot'],
                  ['#ea580c', en ? 'No hospital access' : 'अस्पताल पहुँच छैन', 'dot'],
                  ['#a8a29e', en ? 'Not observable' : 'देख्न नसकिएको', 'fill'],
                ] as [string, string, 'fill' | 'line' | 'dot'][]
              ).map(([c, label, shape]) => (
                <li key={label}>
                  <span
                    aria-hidden
                    style={
                      shape === 'line'
                        ? { width: 12, height: 2.5, background: c }
                        : shape === 'dot'
                          ? {
                              width: 8,
                              height: 8,
                              borderRadius: 8,
                              background: c,
                            }
                          : { width: 9, height: 9, background: c }
                    }
                  />
                  {label}
                </li>
              ))}
            </ul>
            <p className='sr-source'>{L.mapDisclaimer}</p>
          </section>
          <DataTable
            ne={!en}
            caption={`${L.secCutoff} · ${en ? `top ${Math.min(5, (tables.cut_off ?? []).length)} of ${(tables.cut_off ?? []).length}` : ''}`}
            cols={[
              {
                key: 'name',
                label: en ? 'Settlement' : 'बस्ती',
                render: (r) => (
                  <span className='inline-flex items-start gap-1.5'>
                    <span className={`sr-rank ${statusOf(r.status) === 'cut' ? 'sr-rank-cut' : 'sr-rank-hosp'}`} aria-label={`Map 1 marker ${r.rank}`}>
                      {r.rank}
                    </span>
                    <span className='sr-clip' style={{ maxWidth: '27mm' }} title={en ? r.name : r.name_ne || r.name}>
                      {en ? r.name : r.name_ne || r.name} <span className='sr-muted'>({placeLabel(r.place, lang)})</span>
                    </span>
                  </span>
                ),
              },
              {
                key: 'status',
                label: en ? 'Access status' : 'पहुँच',
                render: (r) => (statusOf(r.status) === 'cut' ? <Status tone='bad'>{U.status.cut}</Status> : <Status tone='warn'>{U.status.hosp}</Status>),
              },
              { key: 'buildings', label: en ? 'Bldgs' : 'भवन', num: true },
              {
                key: 'km_pre',
                label: en ? 'Pre dist (km)' : 'पहिलेको दूरी',
                num: true,
              },
              {
                key: 'gmaps_url',
                label: en ? 'Maps' : 'नक्सा',
                render: (r) => <MapLink href={r.gmaps_url} label={en ? 'Map' : 'नक्सा'} />,
              },
            ]}
            rows={(tables.cut_off ?? []).slice(0, 5)}
          />
        </div>

        <div className='sr-stack'>
          <section aria-labelledby='sr-sat' className='sr-box'>
            <h2 id='sr-sat' className='sr-h2'>
              <span>
                <span className='sr-maptag'>{en ? 'Map 2' : 'नक्सा २'}</span> {L.secSatellite}
              </span>
            </h2>
            {maps.satellite_pair ? (
              <img
                src={mapSrc(maps.satellite_pair)}
                alt={en ? 'Before and after satellite images of the analysed area, side by side.' : 'पहिले र पछिका उपग्रह तस्बिर'}
                className='sr-sat'
              />
            ) : (
              <p className='sr-empty'>{en ? 'No usable before/after image pair.' : 'प्रयोगयोग्य तस्बिर जोडी छैन।'}</p>
            )}
            <p className='sr-source'>{data.imagery
                ? `${en ? data.imagery.sensor : data.imagery.sensor.replace('true colour', 'प्राकृतिक रङ')} · ${day(data.imagery.before)} (${en ? 'before' : 'पहिले'}) → ${day(data.imagery.after)} (${en ? 'after' : 'पछि'})`
                : s1Text}</p>
          </section>
          <ul className='sr-areas' aria-label={en ? 'Mapped areas' : 'नक्साङ्कित क्षेत्र'}>
            {(
              [
                ['area_debris_km2', en ? 'Debris & scour' : 'लेदो तथा कटान'],
                ['area_flood_km2', en ? 'New water' : 'नयाँ पानी'],
                ['area_permanent_water_km2', en ? 'Water before' : 'पहिलेको पानी'],
                ['area_not_observable_km2', en ? 'Not observable' : 'देख्न नसकिएको'],
              ] as [string, string][]
            ).map(([k, label]) => (
              <li key={k}>
                <span className='sr-area-num'>{t(k) ? `${t(k)} km²` : dash}</span>
                <span className='sr-area-label'>{label}</span>
              </li>
            ))}
          </ul>
          <DataTable
            ne={!en}
            caption={`${L.secBridges} · ${t('bridges_lost') ?? dash} ${en ? 'lost' : 'बगेका'} · ${t('bridges_at_risk') ?? dash} ${en ? 'at risk' : 'जोखिममा'}`}
            cols={[
              {
                key: 'name',
                label: en ? 'Bridge / highway' : 'पुल',
                render: (r) => (
                  <span className='sr-clip' style={{ maxWidth: '25mm' }} title={bridgeName(r)}>
                    {bridgeName(r)}
                  </span>
                ),
              },
              {
                key: 'status',
                label: en ? 'Status' : 'अवस्था',
                render: (r) => (r.status === 'likely lost' ? <Status tone='bad'>{U.status.lost}</Status> : <Status tone='warn'>{U.status.risk}</Status>),
              },
              {
                key: 'drop_db',
                label: en ? 'Radar drop (dB)' : 'राडार (dB)',
                num: true,
              },
              {
                key: 'length_m',
                label: en ? 'Span (m)' : 'लम्बाइ (मि.)',
                num: true,
              },
              {
                key: 'gmaps_url',
                label: en ? 'Maps' : 'नक्सा',
                render: (r) => <MapLink href={r.gmaps_url} label={en ? 'Map' : 'नक्सा'} />,
              },
            ]}
            rows={(tables.bridges ?? []).slice(0, 3)}
          />
          <DataTable
            ne={!en}
            caption={L.secRepairs}
            cols={[
              { key: 'rank', label: '#', num: true },
              {
                key: 'what',
                label: en ? 'Asset / roadway' : 'सडक',
                render: (r) => (
                  <span className='sr-clip' style={{ maxWidth: '40mm' }} title={r.what}>
                    {assetLabel(r.what, lang)} <span className='sr-muted'>({roadLabel(r.highway, lang)})</span>
                  </span>
                ),
              },
              {
                key: 'settlements',
                label: en ? 'Restores villages' : 'बस्ती',
                num: true,
              },
              {
                key: 'buildings',
                label: en ? 'Restores bldgs' : 'भवन',
                num: true,
              },
            ]}
            rows={(tables.repairs ?? []).slice(0, 2)}
          />
        </div>
      </div>

      {/* 6 footer: data, limitations, credits, provenance */}
      <footer className='sr-foot'>
        <div className='sr-foot-cols'>
          <div>
            <h3>{L.dataUsed}</h3>
            <p>
              <b>{en ? 'Radar' : 'राडार'}:</b> Sentinel-1 {s1Text || dash}
            </p>
            {s2Text && (
              <p>
                <b>{en ? 'Optical' : 'अप्टिकल'}:</b> Sentinel-2 {s2Text}
              </p>
            )}
            <p>
              <b>{en ? 'DEM & OSM' : 'DEM र OSM'}:</b> {data.facts.dem_name ?? 'Copernicus DEM GLO-30'} · OSM {osm ?? dash}
            </p>
            <p>
              <b>{en ? 'Grid' : 'ग्रिड'}:</b> {t('resolution_m') ?? dash} m, {en ? 'local UTM' : 'स्थानीय UTM'}
            </p>
          </div>
          <div>
            <h3>{L.limits}</h3>
            <ul className='sr-list'>
              <li>{en ? 'Images are days apart: damage between them is mapped together, and nothing is forecast.' : 'तस्बिर केही दिनको फरकमा लिइन्छन्; पूर्वानुमान गरिँदैन।'}</li>
              <li>{en ? 'Ground hidden from the radar by steep terrain is unknown, not undamaged.' : 'राडारले नदेखेको क्षेत्रको क्षति अज्ञात छ।'}</li>
              <li>{en ? 'Optical evidence only where the sky was clear; cloud elsewhere.' : 'अप्टिकल प्रमाण बादल नभएको ठाउँमा मात्र।'}</li>
              <li>
                {en
                  ? '10 m pixels: a building counts as hit when its footprint overlaps mapped change; damage to a single house is not graded.'
                  : '१० मि. पिक्सेल: एउटा घरको क्षति मूल्याङ्कन गरिँदैन।'}
              </li>
              <li>
                {en
                  ? 'Low-confidence areas (including valley floor filled in from the observed flood level) are inferred, not seen; severity uses high-confidence evidence only.'
                  : 'कम विश्वासका क्षेत्र (देखिएको बाढी तहबाट थपिएको उपत्यकाको भुइँ सहित) अनुमान हुन्, प्रत्यक्ष देखिएका होइनन्।'}
              </li>
              <li>
                {en
                  ? 'Roads, bridges and buildings missing from the pre-event map are not assessed; routes assume every mapped road was open before.'
                  : 'घटनाअघिको नक्सामा नभएका संरचना जाँच गरिँदैनन्।'}
              </li>
              <li>
                {en
                  ? 'Buildings are counted, not people: no population data is used, so people displaced is not measured.'
                  : 'भवन गनिएका हुन्, मानिस होइनन्; विस्थापित मानिस नापिएको छैन।'}
              </li>
              <li>{en ? 'Not field-verified. Confirm on the ground before acting.' : 'स्थलगत पुष्टि गरिएको छैन।'}</li>
            </ul>
          </div>
          <div>
            <h3>{L.links}</h3>
            <p>
              <a className='sr-link' href={publicLink(`/api/jobs/${data.job_id}/files/provenance.json`)} target='_blank' rel='noreferrer'>
                provenance.json (scenes &amp; parameters)
              </a>
            </p>
            <p>
              <a className='sr-link' href={publicLink(`/api/jobs/${data.job_id}/download.zip`)}>
                download.zip (all GIS layers)
              </a>
            </p>
            <p>
              <a className='sr-link' href={publicLink(`/?job=${data.job_id}`, 'web')} target='_blank' rel='noreferrer'>
                {en ? 'Interactive map dashboard' : 'अन्तरक्रियात्मक नक्सा'}
              </a>
            </p>
          </div>
        </div>
        <p className='sr-credits-line'>
          <b>{L.credits}:</b>{' '}
          {['attr_s', 'attr_dem', 'attr_osm', 'attr_train']
            .map((k) => data.facts[k])
            .filter(Boolean)
            .join(' ')}
        </p>
        <p className='sr-foot-line'>
          <span>
            {en
              ? 'Generated deterministically by the Dhara pipeline. Every statistic is computed from the satellite data and checked against the analysis results.'
              : 'Dhara प्रणालीले स्वचालित रूपमा तयार गरेको। सबै तथ्याङ्क उपग्रह तथ्याङ्कबाट गणना गरिएका हुन्।'}
          </span>
          <span className='sr-proto'>{L.prototype}</span>
        </p>
      </footer>
    </article>
  )
}

// ------------------------------------------------------------------------------------- page

/** The A4 report sheet with its styles. It keeps to one page by shrinking only the main map, and
 *  sets the page title to the report's file name (browsers name a printed PDF after it).
 *  `fluid`: on narrow screens the sheet reflows into one column (the full page); otherwise it stays
 *  a fixed A4 page (the zoomable dialog). */
export function ReportSheet({ data, jobId, lang, fluid = false }: { data: SitrepData; jobId: string; lang: Lang; fluid?: boolean }) {
  const root = useRef<HTMLDivElement>(null)

  const fileName = reportFileName(data.facts, jobId, lang)
  useEffect(() => {
    const before = document.title
    document.title = fileName
    return () => {
      document.title = before
    }
  }, [fileName])

  useEffect(() => {
    const el = root.current?.querySelector<HTMLElement>('.sr-doc')
    if (!el) return
    const MAP_MAX = 272 // px ≈ 72 mm
    const MAP_MIN = 170 // px ≈ 45 mm
    const check = () => {
      if (fluid && window.innerWidth <= 820 && !window.matchMedia('print').matches) return
      el.style.setProperty('--sr-map-h', `${MAP_MAX}px`)
      const over = el.scrollHeight - el.clientHeight
      if (over > 0) el.style.setProperty('--sr-map-h', `${Math.max(MAP_MIN, MAP_MAX - over - 2)}px`)
    }
    const ro = new ResizeObserver(check)
    ro.observe(el)
    const imgs = [...el.querySelectorAll('img')]
    imgs.forEach((i) => i.addEventListener('load', check))
    check()
    const mq = window.matchMedia('print')
    mq.addEventListener('change', check)
    window.addEventListener('beforeprint', check)
    return () => {
      mq.removeEventListener('change', check)
      window.removeEventListener('beforeprint', check)
      ro.disconnect()
      imgs.forEach((i) => i.removeEventListener('load', check))
    }
  }, [data, lang, fluid])

  return (
    <div ref={root} className={fluid ? 'sr-fluid' : 'sr-fixed'}>
      <style>{CSS}</style>
      <Report data={data} lang={lang} />
    </div>
  )
}

export function ReportPage({ onBack }: { onBack?: () => void }) {
  const app = useApp()
  const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null
  const [lang, setLang] = useState<Lang>(params?.get('lang') === 'ne' ? 'ne' : 'en')
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  const done = (jobs.data ?? []).filter((j) => j.kind === 'analysis' && j.status === 'done')
  const fromUrl = params?.get('job') ?? null
  const [jobId, setJobId] = useState<string | null>(fromUrl ?? app.jobId ?? null)
  useEffect(() => {
    if (!jobId && done.length) setJobId(done[0].id)
  }, [jobId, done])
  useEffect(() => {
    if (!jobId) return
    const u = new URL(window.location.href)
    u.searchParams.set('job', jobId)
    if (lang === 'ne') u.searchParams.set('lang', 'ne')
    else u.searchParams.delete('lang')
    window.history.replaceState(null, '', u)
  }, [jobId, lang])

  const report = useQuery({
    queryKey: ['sitrepData', jobId],
    queryFn: () => api.sitrepData(jobId!),
    enabled: !!jobId,
    staleTime: 5 * 60_000,
  })
  const L = T[lang]

  return (
    <div className='sr-page flex h-svh w-screen flex-col bg-muted/40 print:block print:h-auto print:bg-white'>
      <header className='flex h-12 shrink-0 items-center gap-1 border-b bg-background px-2 sm:gap-2 sm:px-3 print:hidden'>
        <Button variant='ghost' size='sm' onClick={onBack ?? (() => (window.location.href = '/'))} className='gap-1.5'>
          <ArrowLeft className='size-4' aria-hidden /> <span className='hidden sm:inline'>{L.back}</span>
        </Button>
        <div className='mx-1 h-5 w-px bg-border' />
        <span className='hidden text-sm font-semibold sm:inline'>{L.kind}</span>
        <div className='ml-auto flex min-w-0 items-center gap-1.5 sm:gap-2'>
          <Select value={jobId ?? undefined} onValueChange={setJobId}>
            <SelectTrigger size='sm' className='w-28 sm:w-60' aria-label={L.pickJob}>
              <SelectValue placeholder={L.pickJob} />
            </SelectTrigger>
            <SelectContent>
              {jobId && !done.some((j) => j.id === jobId) && (
                <SelectItem value={jobId}>
                  {L.analysis} {jobId}
                </SelectItem>
              )}
              {done.map((j) => {
                const l = jobLabel(j)
                return (
                  <SelectItem key={j.id} value={j.id}>
                    {l.title} · {l.sub}
                  </SelectItem>
                )
              })}
            </SelectContent>
          </Select>
          <div className='flex rounded-md border p-0.5' role='group' aria-label='Language'>
            {(['en', 'ne'] as const).map((k) => (
              <button
                key={k}
                type='button'
                aria-pressed={lang === k}
                onClick={() => setLang(k)}
                className={`rounded px-2 py-1 text-xs font-medium ${lang === k ? 'bg-foreground text-background' : 'text-muted-foreground hover:text-foreground'}`}
              >
                {k === 'en' ? (
                  <>
                    <span className='sm:hidden'>EN</span>
                    <span className='hidden sm:inline'>English</span>
                  </>
                ) : (
                  'नेपाली'
                )}
              </button>
            ))}
          </div>
          <PdfButton jobId={jobId} lang={lang} label={L.download} />
        </div>
      </header>
      <div role='main' className='min-h-0 flex-1 overflow-y-auto px-3 py-6 print:overflow-visible print:p-0'>
        {!jobId && !jobs.isLoading && <p className='mx-auto max-w-prose text-sm text-muted-foreground'>{L.noJobs}</p>}
        {report.isLoading && (
          <p className='mx-auto max-w-prose text-sm text-muted-foreground' role='status'>
            {L.loading}
          </p>
        )}
        {report.isError && <p className='mx-auto max-w-prose text-sm text-destructive'>{L.error}</p>}
        {report.data && jobId && <ReportSheet data={report.data} jobId={jobId} lang={lang} fluid />}
      </div>
    </div>
  )
}

// One A4 page, on screen and in print (same sizes in pt, so the screen shows what prints).
const CSS = `
.sr-doc{--sr-ink:#1b1b1b;--sr-muted:#565c65;--sr-line:#dfe1e2;--sr-accent:#1a4480;--sr-bad:#b50909;--sr-warn:#a54e00;--sr-ok:#00704a;--sr-stripe:#f3f4f5;--sr-box:#f7f8f9;
  width:210mm;height:297mm;overflow:hidden;margin:0 auto;background:#fff;color:var(--sr-ink);padding:8mm 9mm;box-sizing:border-box;
  box-shadow:0 1px 3px rgba(0,0,0,.12);display:flex;flex-direction:column;gap:1.8mm;
  font-family:'IBM Plex Sans','IBM Plex Sans Devanagari','Noto Sans Devanagari','Noto Sans SC','Noto Sans CJK SC','PingFang SC','Microsoft YaHei',system-ui,sans-serif;font-size:7.4pt;line-height:1.33}
.sr-doc *{box-sizing:border-box}
.sr-doc :focus-visible{outline:2px solid var(--sr-accent);outline-offset:2px}
.sr-header{border-bottom:2pt solid var(--sr-ink);padding-bottom:1.6mm}
.sr-kicker{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:2mm;font-size:6.8pt;font-weight:600;color:var(--sr-muted);text-transform:uppercase;letter-spacing:.05em}
.sr-brand{color:var(--sr-ink);font-weight:800;font-size:8pt;letter-spacing:.02em}
.sr-sitrepno{color:var(--sr-accent)}
.sr-kicker-right{display:inline-flex;gap:1.2mm;align-items:center;color:var(--sr-ink)}
.sr-proto-tag{border:.75pt solid var(--sr-line);border-radius:2px;padding:.2mm 1.2mm;font-size:6.2pt;color:var(--sr-muted)}
.sr-level{border-radius:2px;padding:.3mm 1.6mm;font-size:6.8pt;font-weight:800;color:#fff;background:var(--sr-warn)}
.sr-level-severe{background:var(--sr-bad)}.sr-level-high{background:#c2410c}.sr-level-moderate{background:#b45309}.sr-level-low{background:var(--sr-ok)}
.sr-titlerow{display:flex;justify-content:space-between;align-items:baseline;gap:4mm;margin:1mm 0 .8mm}
.sr-doc .font-mono{font-family:'JetBrains Mono',ui-monospace,monospace}
.sr-h1{font-size:15pt;line-height:1.15;font-weight:800;color:var(--sr-ink);margin:0;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sr-clip{display:inline-block;vertical-align:top;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.sr-h1-place{color:var(--sr-accent)}
.sr-ref{font-size:6.8pt;color:var(--sr-muted);white-space:nowrap}
.sr-meta{display:flex;flex-wrap:wrap;gap:.4mm 3mm;margin:0;font-size:6.6pt}
.sr-meta div{display:flex;gap:1.2mm}.sr-meta dt{color:var(--sr-muted)}.sr-meta dd{margin:0;font-weight:700}
.sr-verdict-high{border-left-color:#c2410c;background:#fdf1ea}.sr-verdict-moderate{border-left-color:#b45309;background:#fdf6e9}.sr-verdict-low{border-left-color:var(--sr-ok);background:#ecf6f1}
.sr-verdict{margin:1.2mm 0 0;padding:.9mm 2mm;border-left:2.5pt solid var(--sr-bad);background:#f8eff0;font-weight:600;font-size:7.4pt}
.sr-box{border:.75pt solid var(--sr-line);border-radius:3px;padding:1.2mm 2mm}
.sr-h2{display:flex;justify-content:space-between;gap:2mm;font-size:7.2pt;font-weight:800;text-transform:uppercase;letter-spacing:.05em;color:var(--sr-ink);margin:0 0 1.4mm}
.sr-h2-aside{font-weight:500;text-transform:none;letter-spacing:0;color:var(--sr-muted)}
.sr-kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));margin:0;padding:0;list-style:none}
.sr-kpis>li{display:flex;flex-direction:column;padding:0 2.2mm;border-left:.75pt solid var(--sr-line)}
.sr-kpis>li:first-child{border-left:0;padding-left:0}
.sr-kpi-num{font-size:15pt;line-height:1.05;font-weight:800;color:var(--sr-ink);font-variant-numeric:tabular-nums}
.sr-kpi-unit{font-size:8pt;font-weight:600;color:var(--sr-muted)}
.sr-kpi-label{font-size:7.4pt;font-weight:700;margin-top:.6mm}
.sr-kpi-sub{font-size:6.6pt;color:var(--sr-muted);line-height:1.3;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;overflow:hidden}
.sr-bad{color:var(--sr-bad)}.sr-warn{color:var(--sr-warn)}
.sr-assess{display:grid;grid-template-columns:1fr 1fr;gap:1.1mm 5mm}
.sr-assess p{margin:0}
.sr-q{display:block;font-size:7.8pt;color:var(--sr-accent);margin-bottom:.2mm}
.sr-how{display:block;margin-top:.3mm;font-size:6.3pt;line-height:1.25;color:var(--sr-muted);font-style:italic}
.sr-maptag{display:inline-block;background:var(--sr-ink);color:#fff;border-radius:2px;padding:0 1.2mm;margin-right:1mm;font-size:6.6pt;letter-spacing:.03em}
.sr-rank{display:inline-flex;align-items:center;justify-content:center;min-width:3.6mm;height:3.6mm;border-radius:9px;color:#fff;font-size:6pt;font-weight:800;flex-shrink:0}
.sr-rank-cut{background:#b91c1c}.sr-rank-hosp{background:#ea580c}
.sr-chg-row{display:grid;grid-template-columns:auto 1fr;gap:3mm;align-items:start;margin-top:1.6mm;padding-top:1.4mm;border-top:.75pt solid var(--sr-line)}
.sr-chg-head{font-size:7pt;font-weight:800;text-transform:uppercase;letter-spacing:.04em;display:flex;flex-direction:column;max-width:26mm}
.sr-chg-head .sr-chg-sub{text-transform:none;letter-spacing:0;font-weight:500}
.sr-change{display:grid;grid-template-columns:repeat(4,1fr);gap:1mm 3mm;margin:0;padding:0;list-style:none}
.sr-change li{display:flex;flex-direction:column}
.sr-chg-label{font-size:6.5pt;color:var(--sr-muted);text-transform:uppercase;letter-spacing:.03em}
.sr-chg-val{font-variant-numeric:tabular-nums}
.sr-chg-delta{color:var(--sr-accent);font-weight:700;margin-left:1mm}
.sr-chg-sub{font-size:6.3pt;color:var(--sr-muted)}
.sr-visual{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:1.8mm;align-items:start}
.sr-stack{display:flex;flex-direction:column;gap:1.8mm;min-width:0}
.sr-map{display:block;width:100%;max-height:var(--sr-map-h,72mm);object-fit:contain;background:#f8fafc;border:.5pt solid var(--sr-line)}
.sr-sat{display:block;width:100%;max-height:40mm;object-fit:contain;background:#0f172a}
.sr-legend{display:flex;flex-wrap:wrap;gap:.4mm 2.6mm;margin:1mm 0 .4mm;padding:0;list-style:none;font-size:6.6pt}
.sr-legend li{display:inline-flex;align-items:center;gap:1mm}
.sr-source{font-size:6.2pt;color:var(--sr-muted);margin:.6mm 0 0;font-style:italic}
.sr-empty{color:var(--sr-muted);margin:0}
.sr-areas{display:grid;grid-template-columns:repeat(4,1fr);margin:0;padding:0;list-style:none;border:.75pt solid var(--sr-line);border-radius:3px}
.sr-areas li{display:flex;flex-direction:column;align-items:center;padding:1.2mm .6mm;border-left:.75pt solid var(--sr-line)}
.sr-areas li:first-child{border-left:0}
.sr-area-num{font-size:9pt;font-weight:800;font-variant-numeric:tabular-nums;text-align:center}
.sr-area-label{font-size:6.4pt;color:var(--sr-muted);text-transform:uppercase;letter-spacing:.03em}
.sr-tablewrap{margin:0;overflow:hidden;border:.75pt solid var(--sr-line);border-radius:3px}
.sr-table{width:100%;border-collapse:collapse;font-size:6.8pt;font-variant-numeric:tabular-nums}
.sr-table caption{caption-side:top;text-align:left;font-weight:800;font-size:7pt;text-transform:uppercase;letter-spacing:.04em;padding:1.2mm 2mm .8mm;background:var(--sr-box)}
.sr-table th,.sr-table td{padding:.7mm 1.6mm;text-align:left;vertical-align:top;border-bottom:.5pt solid var(--sr-line)}
.sr-table thead th{font-size:6.5pt;font-weight:700;color:var(--sr-muted);background:var(--sr-box)}
.sr-table tbody th{font-weight:600}
.sr-table tbody tr:nth-child(even){background:var(--sr-stripe)}
.sr-table tbody tr:last-child th,.sr-table tbody tr:last-child td{border-bottom:0}
.sr-table .num{text-align:right;white-space:nowrap}
.sr-muted{color:var(--sr-muted);font-weight:400}
.sr-link{color:var(--sr-accent);text-decoration:underline;text-underline-offset:1px}
.sr-foot{margin-top:auto;border-top:2pt solid var(--sr-ink);padding-top:1.4mm;font-size:6.5pt;line-height:1.3;color:var(--sr-ink)}
.sr-foot-cols{display:grid;grid-template-columns:.9fr 2.3fr .75fr;gap:3mm;padding-bottom:1.2mm;border-bottom:.5pt solid var(--sr-line)}
.sr-foot h3{font-size:6.6pt;font-weight:800;text-transform:uppercase;letter-spacing:.05em;margin:0 0 .6mm}
.sr-foot p{margin:0 0 .4mm}.sr-list{margin:0;padding-left:1.1em;list-style:disc outside;columns:2;column-gap:4mm}.sr-list li{break-inside:avoid}.sr-list li{margin:0 0 .2mm;display:list-item}
.sr-credits-line{margin:.8mm 0 0;font-size:6.2pt;color:var(--sr-muted);line-height:1.3}
.sr-foot-line{display:flex;justify-content:space-between;gap:3mm;padding-top:1mm;color:var(--sr-muted);font-size:6.3pt}
.sr-proto{color:var(--sr-warn);font-weight:700}
@media screen and (max-width:820px){
.sr-fluid{
  .sr-doc{width:auto;height:auto;overflow:visible;padding:16px;font-size:10.5pt}
  .sr-kpis,.sr-assess,.sr-visual,.sr-foot-cols,.sr-chg-row,.sr-change{grid-template-columns:1fr}
  .sr-kpis>li{border-left:0;padding:6px 0;border-top:1px solid var(--sr-line)}
  .sr-areas{grid-template-columns:1fr 1fr}
  .sr-h1,.sr-clip{white-space:normal}
  .sr-clip{max-width:none!important}
  .sr-h1{font-size:19pt}.sr-table,.sr-foot,.sr-meta,.sr-legend{font-size:9.5pt}
  .sr-table thead{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
  .sr-table tr{display:block;border-bottom:2px solid var(--sr-line);padding:4px 8px}
  .sr-table tbody tr:nth-child(even){background:#fff}
  .sr-table th,.sr-table td{display:flex;justify-content:space-between;gap:12px;border:0;padding:3px 0;text-align:right}
  .sr-table th::before,.sr-table td::before{content:attr(data-label);font-weight:700;text-align:left;color:var(--sr-muted)}
}
}
@media print{
  @page{size:A4;margin:0}
  html,body,#root{height:auto !important;overflow:visible !important;background:#fff !important}
  body *{visibility:hidden !important}
  .sr-doc,.sr-doc *{visibility:visible !important}
  .sr-doc{position:fixed;left:0;top:0;margin:0;box-shadow:none;-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .sr-link{color:var(--sr-ink)}
  /* Printed from the zoomable dialog: undo the dialog and zoom transforms, which would otherwise
     anchor the fixed page to them instead of to the paper. */
  body *:has(.sr-doc){transform:none !important;translate:none !important;scale:none !important;rotate:none !important;filter:none !important;contain:none !important;will-change:auto !important;animation:none !important;overflow:visible !important}
}
`
