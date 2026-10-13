import type { ReactNode } from 'react'
import { Bot, Copy, Waves, X } from 'lucide-react'
import { toast } from 'sonner'
import { useApp } from '@/context/app-state'
import { distanceKm, fmtHours, fmtKm, inRing, jobAoiRing, num, parseBridge, parseHospital, parseSettlement, pct, str, useJobData } from '@/lib/job'
import { GMapsLink } from '@/components/gmaps-link'
import { Button } from '@/components/ui/button'
import { StatusBadge } from '@/features/workspace/common'

function Line({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className='flex items-baseline justify-between gap-3 py-0.5'>
      <span className='text-muted-foreground'>{k}</span>
      <span className='text-right font-medium tabular-nums'>{v}</span>
    </div>
  )
}

function tagValue(raw: unknown, key: string): string {
  if (typeof raw !== 'string') return ''
  try {
    return String((JSON.parse(raw) as Record<string, unknown>)[key] ?? '')
  } catch {
    return ''
  }
}

export function SelectionCard() {
  const app = useApp()
  const d = useJobData(app.jobId)
  const s = app.selection
  if (!s) return null
  const [lon, lat] = s.lngLat
  const p = s.props

  const traceFrom = () => {
    app.setFlowPoint([+lon.toFixed(5), +lat.toFixed(5)])
    app.setView('flowpath')
  }
  const copy = () => {
    navigator.clipboard.writeText(`${lat.toFixed(5)}, ${lon.toFixed(5)}`)
    toast.success('Coordinates copied')
  }

  let kicker = ''
  let title = ''
  let badge: string | null = null
  let body: ReactNode = null
  let ask: string | null = null

  if (s.kind === 'settlement') {
    const x = parseSettlement(p, s.lngLat)
    const broken = x.status === 'cut_off' || x.status === 'hospital_cut_off'
    kicker = x.place ? `Settlement · ${x.place}` : 'Settlement'
    title = x.name
    badge = x.status
    ask = `Why is ${x.name} classed as "${x.status.replaceAll('_', ' ')}", and what would reconnect it?`
    body = (
      <>
        <Line k='Buildings within 500 m' v={x.buildings} />
        {x.pCutOff != null && <Line k='Chance of no road out at all' v={pct(x.pCutOff)} />}
        {x.hospital && <Line k='Nearest hospital' v={x.hospital} />}
        <div className='my-1.5 grid grid-cols-2 gap-px overflow-hidden rounded-md border bg-border'>
          <div className='bg-card p-2'>
            <div className='text-[0.625rem] text-muted-foreground'>Before</div>
            <div className='font-medium tabular-nums'>{fmtKm(x.kmBefore)}</div>
            <div className='text-muted-foreground tabular-nums'>{fmtHours(x.hoursBefore)}</div>
          </div>
          <div className='bg-card p-2'>
            <div className='text-[0.625rem] text-muted-foreground'>After</div>
            {broken ? (
              <div className='font-medium text-red-700 dark:text-red-400'>No road route</div>
            ) : (
              <>
                <div className='font-medium tabular-nums'>{fmtKm(x.kmAfter)}</div>
                <div className='text-muted-foreground tabular-nums'>{fmtHours(x.hoursAfter)}</div>
              </>
            )}
          </div>
        </div>
        {x.breaks.length > 0 && <p className='text-muted-foreground'>Red marks on the map show where the pre-event route crosses flood or debris.</p>}
        {!x.pathBefore.length && <p className='text-muted-foreground'>No mapped road route to a hospital.</p>}
      </>
    )
  } else if (s.kind === 'bridge') {
    const b = parseBridge(p, s.lngLat)
    const repair = d.repairs.findIndex((r) => r.osmId === b.id)
    kicker = `Bridge${b.highway ? ` · ${b.highway}` : ''}`
    title = b.name || 'Unnamed bridge'
    badge = b.status
    ask = `What is the evidence that bridge ${b.name || b.id} is ${b.status.replaceAll('_', ' ')}?`
    body = (
      <>
        <Line k='Chance lost' v={pct(b.pLost)} />
        <Line k='Flood or debris at both ends' v={pct(num(p.p_hazard))} />
        {b.dropDb != null && <Line k='Radar return' v={`${b.dropDb >= 0 ? 'fell' : 'rose'} ${Math.abs(b.dropDb).toFixed(1)} dB`} />}
        {b.lengthM != null && <Line k='Span' v={`${Math.round(b.lengthM)} m${str(p.below_resolution) === 'True' ? ' (below grid size)' : ''}`} />}
        <Line k='Confidence' v={b.confidence || '—'} />
        {repair >= 0 && (
          <div className='mt-1.5 rounded-md border border-emerald-600/30 bg-emerald-600/5 px-2 py-1.5 text-emerald-800 dark:text-emerald-300'>
            Repair priority {repair + 1}: reconnects {d.repairs[repair].settlements} settlement{d.repairs[repair].settlements === 1 ? '' : 's'},{' '}
            {d.repairs[repair].buildings} buildings
          </div>
        )}
      </>
    )
  } else if (s.kind === 'hospital') {
    const h = parseHospital(p, s.lngLat)
    const served = d.settlements.filter((x) => x.hospital && x.hospital === h.name)
    const lost = served.filter((x) => x.status === 'cut_off' || x.status === 'hospital_cut_off')
    kicker = h.amenity ? `Health · ${h.amenity}` : 'Health facility'
    title = h.name
    badge = h.isolated ? 'cut_off' : h.status
    ask = `Which settlements lost road access to ${h.name}?`
    body = (
      <>
        <Line k='Nearest hospital for' v={`${served.length} settlements`} />
        <Line k='Of those, lost road access' v={lost.length} />
      </>
    )
  } else if (s.kind === 'building' || s.kind === 'road') {
    const hit = str(p.status) === 'hit'
    kicker = s.kind === 'road' ? `Road · ${tagValue(p.tags, 'highway') || 'way'}` : `Building${str(p.facility_type) && str(p.facility_type) !== 'building' ? ` · ${str(p.facility_type)}` : ''}`
    title = str(p.name) || (s.kind === 'road' ? 'Unnamed road' : 'Building')
    badge = hit ? 'hit' : 'not_hit'
    body = (
      <>
        {hit && <Line k='Hazard' v={str(p.hazard) || '—'} />}
        <Line k='Peak probability' v={pct(num(p.p_max))} />
        <Line k='Share of footprint in hazard' v={pct(num(p.frac_hit))} />
        <Line k='Confidence' v={str(p.confidence) || '—'} />
      </>
    )
  } else if (s.kind === 'flow-settlement') {
    kicker = `On the flood path · #${str(p.order)}`
    title = str(p.name) || 'Unnamed settlement'
    body = (
      <>
        <Line k='Distance down the path' v={fmtKm(num(p.distance_km))} />
        <Line k='Height above the river' v={num(p.delta_z_river_m) != null ? `${num(p.delta_z_river_m)!.toFixed(0)} m` : '—'} />
        <Line k='Height above drainage' v={num(p.hand_m) != null ? `${num(p.hand_m)!.toFixed(0)} m` : '—'} />
        {num(p.arrival_time_min) != null && <Line k='Reached in (rough)' v={`${Math.round(num(p.arrival_time_min)!)} min`} />}
      </>
    )
  } else {
    kicker = 'Location'
    title = `${lat.toFixed(5)}°N, ${lon.toFixed(5)}°E`
    const nearest = <T extends { at: [number, number] }>(xs: T[]) =>
      xs.reduce<{ x: T; km: number } | null>((best, x) => {
        const km = distanceKm(s.lngLat, x.at)
        return !best || km < best.km ? { x, km } : best
      }, null)
    const ns = nearest(d.settlements)
    const nh = nearest(d.hospitals)
    const ring = jobAoiRing(d.job.data)
    body = (
      <>
        {ring && <Line k='Analysed area' v={inRing(s.lngLat, ring) ? 'Inside' : 'Outside'} />}
        {ns && (
          <div className='flex items-center justify-between gap-2 py-0.5'>
            <span className='text-muted-foreground'>Nearest settlement</span>
            <button
              type='button'
              className='flex min-w-0 items-center gap-1.5 font-medium hover:underline'
              onClick={() => app.select({ kind: 'settlement', props: ns.x.raw, lngLat: ns.x.at })}
            >
              <span className='truncate'>{ns.x.name}</span>
              <span className='shrink-0 text-muted-foreground tabular-nums'>{ns.km.toFixed(1)} km</span>
            </button>
          </div>
        )}
        {ns && (
          <div className='flex justify-end pb-0.5'>
            <StatusBadge status={ns.x.status} />
          </div>
        )}
        {nh && (
          <div className='flex items-center justify-between gap-2 py-0.5'>
            <span className='text-muted-foreground'>Nearest hospital</span>
            <button
              type='button'
              className='flex min-w-0 items-center gap-1.5 font-medium hover:underline'
              onClick={() => app.select({ kind: 'hospital', props: nh.x.raw, lngLat: nh.x.at })}
            >
              <span className='truncate'>{nh.x.name}</span>
              <span className='shrink-0 text-muted-foreground tabular-nums'>{nh.km.toFixed(1)} km</span>
            </button>
          </div>
        )}
        {(ns || nh) && <p className='pt-1 text-[0.6875rem] text-muted-foreground'>Distances are straight-line, not by road.</p>}
      </>
    )
  }

  return (
    <div className='overflow-hidden rounded-lg border bg-background shadow-lg'>
      <div className='flex items-start gap-2 border-b px-3 py-2.5'>
        <div className='min-w-0 flex-1'>
          <div className='truncate text-[0.6875rem] text-muted-foreground capitalize'>{kicker}</div>
          <div className='truncate text-sm font-semibold'>{title}</div>
        </div>
        {badge && <StatusBadge status={badge} className='mt-0.5' />}
        <GMapsLink at={s.lngLat} className='-mt-0.5' />
        <button type='button' onClick={() => app.select(null)} className='text-muted-foreground hover:text-foreground' aria-label='Close'>
          <X className='size-4' />
        </button>
      </div>
      {body && <div className='px-3 py-2 text-xs'>{body}</div>}
      <div className='flex flex-wrap items-center gap-1 border-t px-2 py-1.5'>
        {ask && app.jobId && (
          <Button variant='ghost' size='sm' className='h-7 gap-1.5 px-2 text-xs' onClick={() => app.askAssistant(ask!)}>
            <Bot className='size-3.5' /> Ask
          </Button>
        )}
        {s.kind !== 'flow-settlement' && (
          <Button variant='ghost' size='sm' className='h-7 gap-1.5 px-2 text-xs' onClick={traceFrom}>
            <Waves className='size-3.5' /> Trace flood path from here
          </Button>
        )}
        <Button variant='ghost' size='sm' className='ml-auto h-7 gap-1.5 px-2 text-xs text-muted-foreground' onClick={copy}>
          <Copy className='size-3.5' /> {lat.toFixed(4)}, {lon.toFixed(4)}
        </Button>
      </div>
    </div>
  )
}
