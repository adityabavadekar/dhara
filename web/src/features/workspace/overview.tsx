import { useState } from 'react'
import {
  ArrowRight,
  Building2,
  ChevronDown,
  ChevronUp,
  Download,
  MapPin,
  Waves,
} from 'lucide-react'
import { useApp } from '@/context/app-state'
import { apiUrl } from '@/lib/api'
import { fixed, STATUS_COLOR, type useJobData } from '@/lib/job'
import { Button } from '@/components/ui/button'
import { PdfButton } from '@/features/report/pdf-button'
import { Dot, List, Note, Row, Section, Stack, StatusBadge } from './common'

type Data = ReturnType<typeof useJobData>

// Mirrors the verdict rules in src/dhara/sitrep/facts.py; keep the two in step.
const SEVERITY_RULES: [string, string][] = [
  ['SEVERE', 'High-confidence: ≥ 1 km² flood or debris, ≥ 250 buildings hit, or ≥ 5 bridges lost'],
  ['HIGH', 'High-confidence: ≥ 0.1 km², ≥ 25 buildings, or ≥ 2 bridges lost'],
  ['MODERATE', 'Any confidence: a community cut off, without hospital access or without a road to one, ≥ 50 buildings, or ≥ 0.25 km²'],
  ['LOW', 'None of the above'],
]

function Mini({ label, value, sub, note }: { label: string; value: string | null; sub?: string; note?: string }) {
  return (
    <div className='min-w-0'>
      <div className='truncate text-[0.6875rem] text-foreground/70'>{label}</div>
      <div className='text-base font-semibold tabular-nums'>
        {value ?? '—'}
        {sub && <span className='ml-1 text-xs font-normal text-muted-foreground'>{sub}</span>}
      </div>
      {note && <div className='truncate text-[0.6875rem] text-muted-foreground tabular-nums'>{note}</div>}
    </div>
  )
}

// "High confidence" means two or more independent methods agree; the severity level counts only these.
const sure = (v: string | null) => (v != null ? `${v} high confidence` : undefined)

export function Overview({ d }: { d: Data }) {
  const app = useApp()
  const { t, n } = d.facts
  const id = d.job.data!.id
  // Runs made before road access was measured fall back to the on-foot "cut off" count.
  // Also when no settlement had a road to a hospital before the event (nothing to compare).
  const hasRoad = t('settlements_vehicle_lost') != null && (n('settlements_vehicle_before') ?? 0) > 0
  const roadLost = (s: (typeof d.settlements)[number]) => s.raw.vehicle_access === 'lost'
  const urgent = d.settlements
    .filter((s) => s.status === 'cut_off' || s.status === 'hospital_cut_off' || roadLost(s))
    .sort((a, b) => b.buildings - a.buildings)
  const topRepair = d.repairs[0]
  const lost = d.bridges.filter((b) => b.status === 'likely_lost')
  const change = d.change
  const openWaterDelta = change ? change.water_after_km2 - change.water_before_km2 : null
  const openWaterPct = change && change.water_before_km2 > 0 ? Math.round((openWaterDelta! / change.water_before_km2) * 100) : null
  const riverWidthDelta =
    change?.width_before_median_m != null && change.width_after_median_m != null
      ? change.width_after_median_m - change.width_before_median_m
      : null
  const [signalOpen, setSignalOpen] = useState(false)

  // Severity and its one-line reason come from the facts the report uses, so both always agree.
  const level = (t('verdict_level') ?? '').toUpperCase()
  const hazardKm2 = (n('area_debris_km2') ?? 0) + (n('area_flood_km2') ?? 0)
  const badgeColor =
    level === 'SEVERE'
      ? 'bg-red-600 text-white'
      : level === 'HIGH'
        ? 'bg-orange-600 text-white'
        : level === 'MODERATE'
          ? 'bg-amber-500 text-white'
          : 'bg-emerald-600 text-white'

  return (
    <div className='space-y-2 p-3'>
      {/* Severity */}
      <div className='overflow-hidden rounded-lg bg-card shadow-xs'>
        <button
          type='button'
          onClick={() => setSignalOpen(!signalOpen)}
          className='flex w-full items-center justify-between gap-2 px-3 py-2 text-left transition-colors hover:bg-muted/30'
          aria-expanded={signalOpen}
        >
          <span className='text-xs font-semibold tracking-wider text-muted-foreground uppercase'>Severity</span>
          <div className='flex items-center gap-1.5'>
            {level && <span className={`shrink-0 rounded px-2 py-0.5 text-[0.625rem] font-bold tracking-wider uppercase ${badgeColor}`}>{level}</span>}
            {signalOpen ? <ChevronUp className='size-3.5 text-muted-foreground' /> : <ChevronDown className='size-3.5 text-muted-foreground' />}
          </div>
        </button>
        {t('verdict_headline') && <p className='border-t px-3 py-2 text-xs leading-snug text-foreground/85'>{t('verdict_headline')}</p>}
        {signalOpen && (
          <div className='space-y-2 border-t bg-muted/20 px-3 py-2.5 text-xs'>
            <div>
              <div className='font-semibold text-foreground/90'>Mapped area by confidence</div>
              <div className='mt-1 leading-relaxed text-muted-foreground'>
                High <b className='font-semibold text-foreground'>{t('area_tier_high_km2') ?? '—'} km²</b> · Moderate{' '}
                <b className='font-semibold text-foreground'>{t('area_tier_moderate_km2') ?? '—'} km²</b> · Low{' '}
                <b className='font-semibold text-foreground'>{t('area_tier_low_km2') ?? '—'} km²</b>
              </div>
              <div className='mt-1 text-[0.6875rem] leading-snug text-muted-foreground/80'>
                Severity counts only high-confidence change (two or more independent signals agree).
              </div>
            </div>
            <div>
              <div className='font-semibold text-foreground/90'>How the level is set</div>
              <dl className='mt-1 space-y-0.5 text-[0.6875rem] leading-snug text-muted-foreground'>
                {SEVERITY_RULES.map(([lv, rule]) => (
                  <div key={lv} className='flex gap-2'>
                    <dt className={`w-16 shrink-0 font-semibold ${lv === level ? 'text-foreground' : ''}`}>{lv}</dt>
                    <dd>{rule}</dd>
                  </div>
                ))}
              </dl>
              <div className='mt-1 text-[0.6875rem] leading-snug text-muted-foreground/80'>
                The first level that matches applies. Thresholds rank runs; they are design choices, not calibrated against outcomes.
              </div>
            </div>
            <div className='text-[0.6875rem] leading-snug text-muted-foreground/80'>
              People displaced: <i>not measured</i> (no population data is used; buildings hit is the closest proxy).
            </div>
          </div>
        )}
      </div>

      <div className='space-y-2'>
        {/* 1. Where did the flood hit? */}
        <div className='rounded-lg bg-amber-500/[0.04] p-3 shadow-xs dark:bg-amber-500/[0.05]'>
          <div className='flex items-center gap-2 pb-2.5 text-xs font-semibold text-amber-800 dark:text-amber-300'>
            <Waves className='size-3.5 text-amber-600' />
            <span>Where did the flood hit?</span>
          </div>
          <button
            type='button'
            onClick={() => app.setView('flooding')}
            className='w-full rounded-md bg-amber-500/[0.08] p-2.5 text-left transition-colors hover:bg-amber-500/[0.14] dark:bg-background/90 dark:hover:bg-amber-500/[0.08]'
          >
            <div className='flex items-baseline justify-between'>
              <span className='text-[0.6875rem] font-medium text-foreground/80'>Mapped flood and debris</span>
              <span className='text-xs text-muted-foreground'>of {t('aoi_area_km2') ?? '—'} km²</span>
            </div>
            <div className='mt-1 flex items-baseline gap-1.5'>
              <span className='text-xl font-bold tabular-nums sm:text-2xl text-foreground'>{hazardKm2 > 0 && hazardKm2 < 0.005 ? '<0.01' : hazardKm2.toLocaleString('en', { maximumFractionDigits: 2 })}</span>
              <span className='text-sm text-muted-foreground'>
                km² · {t('area_debris_km2') ?? '—'} debris and scour, {t('area_flood_km2') ?? '—'} new water
              </span>
            </div>
            <div className='mt-2'>
              <Stack
                parts={[
                  { value: n('area_debris_km2') ?? 0, color: '#c27803', label: 'Debris' },
                  { value: n('area_flood_km2') ?? 0, color: '#2563eb', label: 'New water' },
                ]}
              />
            </div>
            <div className='mt-1.5 text-[0.6875rem] text-muted-foreground'>
              {(n('area_not_observable_km2') ?? 0) >= 0.01 ? (
                <>
                  {t('area_not_observable_km2')} km² hidden from the radar by terrain: <b className='font-semibold'>unknown, not safe</b>.
                </>
              ) : (
                <>The radar saw the whole area.</>
              )}
            </div>
          </button>
          {change && (
            <div className='mt-2 space-y-1 rounded-md bg-amber-500/[0.06] p-2 dark:bg-white/[0.02] text-[0.6875rem] text-muted-foreground'>
              <div className='space-y-0.5'>
                {riverWidthDelta != null && riverWidthDelta > 0 && (
                  <div>
                    • River widened by <b className='font-semibold text-foreground'>+{riverWidthDelta} m</b> ({change.width_before_median_m} m → {change.width_after_median_m} m)
                  </div>
                )}
                {openWaterDelta != null && openWaterDelta > 0 && (
                  <div>
                    • Open water <b className='font-semibold text-foreground'>+{openWaterPct}%</b> ({fixed(change.water_before_km2, 2)} → {fixed(change.water_after_km2, 2)} km²)
                  </div>
                )}
                {change.shift_max_m != null && change.shift_max_m > 0 && (
                  <div>
                    • River moved sideways up to <b className='font-semibold text-foreground'>{change.shift_max_m} m</b> (median {change.shift_median_m ?? 0} m)
                  </div>
                )}
              </div>
              {change.clear_both_pct != null && (
                <div className='border-t border-foreground/10 pt-1 text-[0.625rem] text-muted-foreground/80'>
                  Measured on the {change.clear_both_pct.toFixed(1)}% of the area clear in both optical images
                </div>
              )}
            </div>
          )}
        </div>

        {/* 2. What was damaged? */}
        <div className='rounded-lg bg-orange-500/[0.04] p-3 shadow-xs dark:bg-orange-500/[0.05]'>
          <div className='flex items-center gap-2 pb-2.5 text-xs font-semibold text-orange-800 dark:text-orange-300'>
            <Building2 className='size-3.5 text-orange-600' />
            <span>What was damaged?</span>
          </div>
          <button
            type='button'
            onClick={() => app.setView('damage')}
            className='grid w-full grid-cols-3 gap-2 rounded-md bg-orange-500/[0.08] p-2.5 text-left transition-colors hover:bg-orange-500/[0.14] dark:bg-background/90 dark:hover:bg-orange-500/[0.08]'
          >
            <Mini label='Buildings hit' value={t('buildings_hit')} sub={t('buildings_total') ? `of ${t('buildings_total')}` : undefined} note={sure(t('buildings_hit_high'))} />
            <Mini label='Impassable road' value={t('network_failed_km') != null ? `${t('network_failed_km')} km` : null} />
            <Mini label='Bridges lost' value={t('bridges_lost')} sub={(n('bridges_at_risk') ?? 0) > 0 ? `+${t('bridges_at_risk')} at risk` : undefined} note={sure(t('bridges_lost_high'))} />
          </button>
        </div>

        {/* 3. Who is cut off? */}
        <div className='rounded-lg bg-red-500/[0.04] p-3 shadow-xs dark:bg-red-500/[0.05]'>
          <div className='flex items-center gap-2 pb-2.5 text-xs font-semibold text-red-800 dark:text-red-300'>
            <MapPin className='size-3.5 text-red-600' />
            <span>Who is cut off?</span>
          </div>
          <div className='grid grid-cols-2 gap-2'>
            <button
              type='button'
              onClick={() => app.setView('cutoff')}
              className='group rounded-md bg-red-500/[0.08] p-2.5 text-left transition-colors hover:bg-red-500/[0.14] dark:bg-background/90 dark:hover:bg-red-500/[0.08]'
            >
              <div className='flex items-center justify-between text-[0.6875rem] font-medium text-foreground/80'>
                <span>{hasRoad ? 'Lost road to a hospital' : 'No way out'}</span>
                <ArrowRight className='size-3 text-muted-foreground transition-transform group-hover:translate-x-0.5' />
              </div>
              <div className={`mt-1 text-xl font-bold tabular-nums sm:text-2xl ${(n(hasRoad ? 'settlements_vehicle_lost' : 'settlements_cut_off') ?? 0) > 0 ? 'text-red-700 dark:text-red-400' : 'text-foreground'}`}>
                {t(hasRoad ? 'settlements_vehicle_lost' : 'settlements_cut_off')}
                <span className='ml-1 text-xs font-normal text-muted-foreground'>of {t(hasRoad ? 'settlements_vehicle_before' : 'settlements_total')}</span>
              </div>
              <div className='mt-0.5 text-[0.6875rem] text-muted-foreground'>
                {hasRoad
                  ? `communities that had one · ${t('settlements_cut_off') ?? '—'} with no way out, even on foot`
                  : `communities · ${t('settlements_hospital_cut_off') ?? '—'} more without a hospital`}
              </div>
            </button>
            <button
              type='button'
              onClick={() => app.setView('cutoff')}
              className='group rounded-md bg-red-500/[0.08] p-2.5 text-left transition-colors hover:bg-red-500/[0.14] dark:bg-background/90 dark:hover:bg-red-500/[0.08]'
            >
              <div className='flex items-center justify-between text-[0.6875rem] font-medium text-foreground/80'>
                <span>{hasRoad ? 'Buildings in them' : 'Buildings nearby'}</span>
                <ArrowRight className='size-3 text-muted-foreground transition-transform group-hover:translate-x-0.5' />
              </div>
              <div className='mt-1 text-xl font-bold tabular-nums sm:text-2xl text-foreground'>{t(hasRoad ? 'buildings_in_vehicle_lost' : 'buildings_in_cut_off')}</div>
              <div className='mt-0.5 text-[0.6875rem] text-muted-foreground'>within {t('exposure_radius_m') ?? '—'} m · no population data</div>
            </button>
          </div>
        </div>
      </div>

      <Section title='Act on first' aside={urgent.length ? `${urgent.length} settlements` : undefined}>
        {urgent.length === 0 && !topRepair && lost.length === 0 ? (
          <Note>No settlement lost its road connection and no bridge is classed as likely lost in the mapped network.</Note>
        ) : (
          <List>
            {urgent.slice(0, 5).map((s) => (
              <Row
                key={s.id}
                at={s.at}
                onClick={() => {
                  app.setView('cutoff')
                  app.select({ kind: 'settlement', props: s.raw, lngLat: s.at }, 14)
                }}
              >
                <div className='flex items-center justify-between gap-2'>
                  <span className='truncate font-medium'>{s.name}</span>
                  <StatusBadge status={s.status !== 'cut_off' && s.status !== 'hospital_cut_off' && roadLost(s) ? 'road_lost' : s.status} />
                </div>
                <div className='mt-0.5 text-[0.6875rem] text-muted-foreground'>{s.buildings} buildings within {t('exposure_radius_m') ?? '—'} m</div>
              </Row>
            ))}
            {topRepair && (
              <Row
                onClick={() => {
                  app.setView('cutoff')
                  const b = d.bridges.find((x) => x.id === topRepair.osmId)
                  if (b) app.select({ kind: 'bridge', props: b.raw, lngLat: b.at }, 15)
                  else {
                    const road = d.roadFeatures.find((x) => String(x.raw.osm_id) === topRepair.osmId)
                    if (road) app.select({ kind: 'road', props: road.raw, lngLat: road.at }, 15)
                  }
                }}
              >
                <div className='flex items-center gap-2'>
                  <Dot color='#16a34a' />
                  <span className='truncate font-medium'>
                    Repair {topRepair.isBridge ? 'bridge' : 'road'} {topRepair.name || topRepair.highway}
                  </span>
                </div>
                <div className='mt-0.5 text-[0.6875rem] text-muted-foreground'>
                  Reconnects {topRepair.settlements} settlement{topRepair.settlements === 1 ? '' : 's'} and {topRepair.buildings} buildings
                </div>
              </Row>
            )}
            {lost.slice(0, 3).map((b) => (
              <Row key={b.id} at={b.at} onClick={() => app.select({ kind: 'bridge', props: b.raw, lngLat: b.at }, 15)}>
                <div className='flex items-center justify-between gap-2'>
                  <span className='truncate font-medium'>Bridge {b.name || b.highway || b.id}</span>
                  <StatusBadge status={b.status} />
                </div>
              </Row>
            ))}
          </List>
        )}
      </Section>

      <Section title='How far to trust this'>
        <ul className='space-y-1.5 text-[0.6875rem] leading-relaxed text-muted-foreground'>
          <li className='flex gap-2'>
            <Dot color={STATUS_COLOR.not_observable} className='mt-0.5' />
            <span>Grey areas were hidden from the radar (shadow or layover). They are not known to be safe.</span>
          </li>
          <li className='flex gap-2'>
            <Dot color='#b45309' className='mt-0.5' />
            <span>
              At {t('resolution_m') ?? '—'} m a house is one or two pixels, so a building "hit" means its footprint overlaps mapped
              flood or debris, not a confirmed collapse.
            </span>
          </li>
          <li className='flex gap-2'>
            <Dot color='#475569' className='mt-0.5' />
            <span>
              Roads and buildings come from OpenStreetMap as of {(t('osm_snapshot') ?? '').slice(0, 10) || '—'}. Paths missing from the map
              cannot be checked.
            </span>
          </li>
          <li className='flex gap-2'>
            <Dot color='#2563eb' className='mt-0.5' />
            <span>
              Evidence: {t('s1_track_count') ?? '—'} Sentinel-1 track{t('s1_track_count') === '1' ? '' : 's'}, Sentinel-2{' '}
              {t('s2_pair') ?? 'not used'}.
            </span>
          </li>
        </ul>
      </Section>

      <Section title='Export'>
        <div className='grid grid-cols-2 gap-1.5'>
          <PdfButton jobId={id} lang='en' label='Situation report (PDF)' />
          <Button asChild variant='outline' size='sm' className='h-8 justify-start text-xs'>
            <a href={apiUrl(`/api/jobs/${id}/download.zip`)}>
              <Download className='size-3.5' /> GeoJSON bundle
            </a>
          </Button>
        </div>
      </Section>
    </div>
  )
}
