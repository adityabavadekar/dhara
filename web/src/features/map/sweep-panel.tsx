// Side panel of the before/after sweep: measured differences between the two dates, and which
// overlays to show on both halves of the map.
import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import type { ChangeSummary } from '@/lib/api'
import { fixed, useFacts } from '@/lib/job'

export type SweepOpt = 'outlines' | 'width' | 'water' | 'buildings' | 'settlements' | 'hospitals' | 'roads' | 'flowpath'
export type SweepSide = 'before' | 'after'

/** Overlays for the sweep. `side` layers appear only on that half (before = left, after = right). */
export const SWEEP_OPTS: { id: SweepOpt; label: string; swatch: string[]; layers?: string[]; side?: Record<SweepSide, string[]> }[] = [
  {
    id: 'width',
    label: 'River width',
    swatch: ['#f8fafc', '#22d3ee'],
    side: { before: ['c-width-before-casing', 'c-width-before', 'c-label-before'], after: ['c-width-after-casing', 'c-width-after', 'c-label-after'] },
  },
  { id: 'water', label: 'Water edge', swatch: ['#f8fafc', '#22d3ee'], side: { before: ['c-water-before'], after: ['c-water-after'] } },
  { id: 'outlines', label: 'Mapped damage', swatch: ['#fbbf24', '#38bdf8'], layers: ['hz-outline-casing', 'hz-outline'] },
  { id: 'buildings', label: 'Buildings', swatch: ['#b45309'], layers: ['v-buildings'] },
  { id: 'roads', label: 'Roads and bridges', swatch: ['#dc2626'], layers: ['v-roads', 'v-failed', 'v-bridges-casing', 'v-bridges'] },
  { id: 'settlements', label: 'Settlements', swatch: ['#94a3b8'], layers: ['v-settlements', 'v-settlements-label'] },
  { id: 'hospitals', label: 'Hospitals', swatch: ['#dc2626'], layers: ['v-hospitals', 'v-hospitals-symbol', 'v-hospitals-label'] },
  { id: 'flowpath', label: 'Flood path', swatch: ['#7c3aed'], layers: ['f-corridor', 'f-path', 'f-pulse', 'f-settlements', 'f-settlements-label', 'f-start'] },
]

export const SWEEP_DEFAULT: Record<SweepOpt, boolean> = {
  width: true,
  water: false,
  outlines: true,
  buildings: false,
  roads: false,
  settlements: false,
  hospitals: false,
  flowpath: true,
}

/** Layers visible on one half of the sweep: imagery, the area boundary and the chosen overlays. */
export function sweepLayers(opts: Record<SweepOpt, boolean>, side: SweepSide): Set<string> {
  const keep = new Set(['basemap', side === 'before' ? 'img-before' : 'img-after', 'aoi-fill', 'aoi-casing', 'aoi-line'])
  for (const o of SWEEP_OPTS) if (opts[o.id]) for (const l of [...(o.layers ?? []), ...(o.side?.[side] ?? [])]) keep.add(l)
  return keep
}

const km2 = (v: number) => `${fixed(v, 2)} km²`
const rel = (b: number, a: number) => (b > 0 ? `${a >= b ? '+' : ''}${Math.round(((a - b) / b) * 100)}%` : null)

function Diff({ label, note, before, after, delta }: { label: string; note?: string; before: string; after: string; delta?: string | null }) {
  return (
    <div className='py-1.5'>
      <div className='flex items-baseline justify-between gap-2'>
        <span className='text-white/75'>{label}</span>
        {delta && <span className='font-semibold tabular-nums text-cyan-300'>{delta}</span>}
      </div>
      <div className='flex items-baseline gap-1.5 tabular-nums'>
        <span className='text-white/80'>{before}</span>
        <span className='text-white/45'>→</span>
        <span className='text-sm font-semibold text-white'>{after}</span>
      </div>
      {note && <div className='text-[0.625rem] text-white/50'>{note}</div>}
    </div>
  )
}

export function SweepPanel({
  jobId,
  change,
  opts,
  setOpts,
  hasFlow,
}: {
  jobId: string | null
  change: ChangeSummary | undefined
  opts: Record<SweepOpt, boolean>
  setOpts: (o: Record<SweepOpt, boolean>) => void
  hasFlow: boolean
}) {
  // Collapsed on phones, where the open panel would cover the divider.
  const [open, setOpen] = useState(() => typeof window === 'undefined' || window.matchMedia('(min-width: 640px)').matches)
  const { t } = useFacts(jobId, !!jobId)
  const c = change
  const avail = SWEEP_OPTS.filter((o) => (o.id === 'flowpath' ? hasFlow : o.side ? !!c : true))

  return (
    <div className='pointer-events-auto absolute top-28 left-3 z-30 flex max-h-[calc(100%-10rem)] w-60 flex-col rounded-md bg-neutral-900/90 text-[0.6875rem] text-white shadow-lg'>
      <button type='button' onClick={() => setOpen(!open)} className='flex w-full items-center justify-between px-3 py-2 text-xs font-semibold'>
        What changed
        <ChevronDown className={`size-3.5 transition-transform ${open ? '' : '-rotate-90'}`} />
      </button>
      {open && (
        <div className='min-h-0 overflow-y-auto px-3 pb-3'>
          {c ? (
            <>
              <div className='divide-y divide-white/10'>
                {c.width_before_median_m != null && c.width_after_median_m != null && (
                  <Diff
                    label='River width'
                    note={`Median of ${c.stations} points along the main rivers. ${c.widened_stations} of them widened by 50 m or more.`}
                    before={`${c.width_before_median_m} m`}
                    after={`${c.width_after_median_m} m`}
                    delta={`${c.width_after_median_m - c.width_before_median_m >= 0 ? '+' : ''}${c.width_after_median_m - c.width_before_median_m} m`}
                  />
                )}
                {c.shift_median_m != null && (
                  <div className='py-1.5'>
                    <div className='text-white/75'>River moved sideways</div>
                    <div className='tabular-nums'>
                      <span className='text-sm font-semibold'>{c.shift_max_m} m</span> <span className='text-white/60'>at most · median {c.shift_median_m} m</span>
                    </div>
                  </div>
                )}
                <Diff label='Open water' before={km2(c.water_before_km2)} after={km2(c.water_after_km2)} delta={rel(c.water_before_km2, c.water_after_km2)} />
                <Diff
                  label='Bare channel'
                  note='Water, mud and fresh sediment: ground with no plants.'
                  before={km2(c.channel_before_km2)}
                  after={km2(c.channel_after_km2)}
                  delta={rel(c.channel_before_km2, c.channel_after_km2)}
                />
              </div>
              <p className='mt-1 text-[0.625rem] leading-snug text-white/50'>
                Measured only where both optical images are cloud-free ({c.clear_both_pct}% of the area). Elsewhere: not measured.
              </p>
            </>
          ) : (
            <p className='text-white/60'>No cloud-free optical pair for this run, so water and river width were not measured.</p>
          )}

          <div className='mt-2 border-t border-white/10 pt-2'>
            <div className='mb-1 text-white/75'>Whole area (radar, sees through cloud)</div>
            <div className='grid grid-cols-[1fr_auto] gap-x-2 gap-y-0.5 tabular-nums'>
              <span className='text-white/60'>Debris and scour</span>
              <span className='font-semibold'>{t('area_debris_km2') ?? '—'} km²</span>
              <span className='text-white/60'>Buildings touched</span>
              <span className='font-semibold'>{t('buildings_hit') ?? '—'}</span>
              <span className='text-white/60'>Communities cut off</span>
              <span className='font-semibold'>{t('settlements_cut_off') ?? '—'}</span>
              <span className='text-white/60'>People displaced</span>
              <span className='text-white/60'>not measured</span>
            </div>
            <p className='mt-1 text-[0.625rem] leading-snug text-white/50'>
              No population data is used. Buildings touched is the closest measure of homes affected.
            </p>
          </div>

          <div className='mt-2 border-t border-white/10 pt-2'>
            <div className='mb-1 text-white/75'>Show on both sides</div>
            <div className='flex flex-wrap gap-1'>
              {avail.map((o) => (
                <button
                  key={o.id}
                  type='button'
                  onClick={() => setOpts({ ...opts, [o.id]: !opts[o.id] })}
                  className={`flex items-center gap-1 rounded border px-1.5 py-0.5 ${opts[o.id] ? 'border-white bg-white text-neutral-900' : 'border-white/25 text-white/80 hover:bg-white/10'}`}
                >
                  <span className='flex'>
                    {o.swatch.map((s) => (
                      <span key={s} className='size-2 rounded-full border border-neutral-900/30' style={{ background: s }} />
                    ))}
                  </span>
                  {o.label}
                </button>
              ))}
            </div>
            {c && (opts.width || opts.water) && (
              <p className='mt-1.5 text-[0.625rem] leading-snug text-white/50'>White = before (left), cyan = after (right).</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
