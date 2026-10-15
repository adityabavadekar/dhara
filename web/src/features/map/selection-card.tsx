import type { ReactNode } from 'react'
import { pinyin } from 'pinyin-pro'
import Sanscript from '@indic-transliteration/sanscript'
import { ArrowLeft, ArrowRight, Bot, Copy, MapPin, Waves, X } from 'lucide-react'
import { toast } from 'sonner'
import { useApp } from '@/context/app-state'
import { distanceKm, fmtHours, fmtKm, inRing, jobAoiRing, num, parseBridge, parseHospital, parseSettlement, pct, score, str, useJobData, useSettlementRoute } from '@/lib/job'
import { GMapsLink } from '@/components/gmaps-link'
import { ElevationProfile } from '@/components/elevation-profile'
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
  const withRoute = useSettlementRoute(app.jobId, s)
  if (!s) return null
  const [lon, lat] = s.lngLat
  const p = withRoute ?? s.props
  const settlementName = s.kind === 'settlement' ? parseSettlement(p, s.lngLat).name : ''
  const pinyinName = settlementName && /\p{Script=Han}/u.test(settlementName)
    ? pinyin(settlementName, { toneType: 'symbol' })
    : ''
  const devanagariName = settlementName && /\p{Script=Devanagari}/u.test(settlementName)
    ? Sanscript.t(settlementName, 'devanagari', 'iast').replace(/~/g, '\u0303')
    : ''

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

  if (p.suitability && p.slope_deg != null) {
    kicker = 'Possible helicopter landing area'
    title = `Near ${str(p.settlement) || 'cut-off settlement'}`
    badge = null
    ask = `Which settlements near ${str(p.settlement)} have no road access?`
    body = (
      <>
        <Line k='Distance to settlement' v={`${num(p.distance_m)} m`} />
        <Line k='Ground slope' v={`${num(p.slope_deg)}°`} />
        <Line k='Flat open area' v={`${num(p.area_m2)} m²`} />
        <Line k='Ground elevation' v={`${num(p.elevation_m)} m`} />
        <p className='mt-1.5 text-muted-foreground'>
          Gentle, dry ground outside mapped flood and debris, from a 30 m elevation model. Trees, wires, buildings and soft
          ground are not checked: confirm on site before use.
        </p>
      </>
    )
  } else if (s.kind === 'settlement') {
    const x = parseSettlement(p, s.lngLat)
    const broken = x.status === 'cut_off' || x.status === 'hospital_cut_off'
    kicker = x.place ? `Settlement · ${x.place}` : 'Settlement'
    title = x.name
    badge = x.status
    ask = `Why is ${x.name} classed as "${x.status.replaceAll('_', ' ')}", and what would reconnect it?`
    const MEANS: Record<string, string> = {
      cut_off: 'After the flood no mapped road or trail leads out of here to a town, a health facility or beyond the area.',
      hospital_cut_off: 'Roads still lead out, but after the flood none of them reaches a hospital.',
      detour: 'Still connected by road, but only by a longer route than before the flood.',
      connected: 'Its usual road route to the hospital still works after the flood.',
      no_baseline_route:
        'The pre-flood map shows no road from here to a town or hospital, so road access cannot be judged. It is probably reached by unmapped foot trails.',
      not_on_network: 'No mapped road comes near this place, so road access cannot be judged.',
    }
    body = (
      <>
        <p className='mb-1.5 leading-snug text-foreground/85'>{MEANS[x.status] ?? ''}</p>
        <Line k={`Buildings within ${d.facts.t('exposure_radius_m') ?? '—'} m`} v={x.buildings} />
        {x.pCutOff != null && x.pathBefore.length > 0 && <Line k='Cut-off frequency in simulations' v={pct(x.pCutOff)} />}
        {x.hospital && x.pathBefore.length > 0 && (
          <div className='flex items-center justify-between gap-2 py-0.5'>
            <span className='text-muted-foreground'>Hospital by road</span>
            <button
              type='button'
              className='flex items-center gap-1 font-medium text-primary hover:underline'
              onClick={() => {
                const route = !broken && x.pathAfter.length > 1 ? x.pathAfter : x.pathBefore
                // Many OSM hospitals share a generic name ("Hospital"), so find the one at the end
                // of this settlement's actual road route, not the first with the same name.
                const end = route[route.length - 1]
                const targetHosp = end
                  ? d.hospitals.reduce<(typeof d.hospitals)[number] | undefined>(
                      (best, h) => (!best || distanceKm(h.at, end) < distanceKm(best.at, end) ? h : best),
                      undefined
                    )
                  : d.hospitals.find((h) => h.name === x.hospital)
                app.select({
                  kind: 'hospital',
                  props: targetHosp ? targetHosp.raw : { name: x.hospital },
                  lngLat: targetHosp ? targetHosp.at : (route[route.length - 1] ?? x.at),
                  originLngLat: x.at,
                  routePath: route,
                })
              }}
            >
              <span>{x.hospital}</span>
              <ArrowRight className='size-3 shrink-0' />
            </button>
          </div>
        )}
        {x.pathBefore.length > 0 && (
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
        )}
        {x.breaks.length > 0 && <p className='text-muted-foreground'>Red marks on the map show where the pre-event route crosses flood or debris.</p>}
        {s.originLngLat && (
          <Button
            variant='outline'
            size='sm'
            className='mt-2 h-7 w-full gap-1.5 text-[0.6875rem]'
            onClick={() => app.select({ kind: 'point', props: {}, lngLat: s.originLngLat! })}
          >
            <ArrowLeft className='size-3' />
            <span>Back</span>
          </Button>
        )}
        {app.jobId && ((!broken && x.pathAfter.length > 1) || x.pathBefore.length > 1) && (
          <div className='mt-2.5'>
            <ElevationProfile
              jobId={app.jobId}
              path={!broken && x.pathAfter.length > 1 ? x.pathAfter : x.pathBefore}
              title={!broken && x.pathAfter.length > 1 ? 'Detour Route Elevation' : 'Hospital Route Elevation'}
              subtitle='Topography along route from Copernicus DEM'
            />
          </div>
        )}
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
        <Line k='Loss model score' v={score(b.pLost)} />
        <Line k='Flood or debris score at both ends' v={score(num(p.p_hazard))} />
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
    kicker = s.originLngLat ? 'Medical Evacuation Destination' : (h.amenity ? `Health · ${h.amenity}` : 'Health facility')
    title = h.name
    badge = h.isolated ? 'cut_off' : h.status
    ask = `Which settlements lost road access to ${h.name}?`
    const distFromOrigin = s.originLngLat ? distanceKm(s.originLngLat, s.lngLat) : null
    const routeCoords = s.routePath && s.routePath.length > 1 ? s.routePath : []

    body = (
      <>
        {s.originLngLat && (
          <div className='mb-2 rounded-md border border-primary/20 bg-primary/5 p-2'>
            <div className='flex items-center justify-between text-[0.6875rem]'>
              <span className='flex items-center gap-1 font-medium text-primary'>
                <MapPin className='size-3' />
                <span>From</span>
              </span>
              <span className='font-mono tabular-nums text-muted-foreground'>
                {s.originLngLat[1].toFixed(4)}°N, {s.originLngLat[0].toFixed(4)}°E
              </span>
            </div>
            <div className='mt-1 flex items-center justify-between text-xs'>
              <span className='text-muted-foreground'>Straight-line distance</span>
              <span className='font-semibold tabular-nums text-foreground'>{distFromOrigin?.toFixed(1)} km</span>
            </div>
            <Button
              variant='outline'
              size='sm'
              className='mt-2 h-7 w-full gap-1.5 text-[0.6875rem]'
              onClick={() => app.select({ kind: 'point', props: {}, lngLat: s.originLngLat! })}
            >
              <ArrowLeft className='size-3' />
              <span>Back</span>
            </Button>
          </div>
        )}
        <Line k='Nearest hospital for' v={`${served.length} settlements`} />
        <Line k='Of those, lost road access' v={lost.length} />
        {app.jobId && routeCoords.length > 1 && (
          <div className='mt-2.5'>
            <ElevationProfile
              jobId={app.jobId}
              path={routeCoords}
              title='Road route elevation'
              subtitle='Topography from Copernicus DEM'
            />
          </div>
        )}
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
        <Line k='Peak model score' v={score(num(p.p_max))} />
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
            <span className='text-muted-foreground'>Closest settlement</span>
            <button
              type='button'
              className='flex min-w-0 items-center gap-1.5 font-medium text-primary hover:underline'
              onClick={() => app.select({ kind: 'settlement', props: ns.x.raw, lngLat: ns.x.at, originLngLat: s.lngLat })}
            >
              <span className='truncate'>{ns.x.name}</span>
              <span className='shrink-0 text-muted-foreground tabular-nums'>{ns.km.toFixed(1)} km</span>
              <ArrowRight className='size-3 shrink-0' />
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
            <span className='text-muted-foreground'>Closest hospital</span>
            <button
              type='button'
              className='flex min-w-0 items-center gap-1.5 font-medium text-primary hover:underline'
              onClick={() => app.select({ kind: 'hospital', props: nh.x.raw, lngLat: nh.x.at, originLngLat: s.lngLat })}
            >
              <span className='truncate'>{nh.x.name}</span>
              <span className='shrink-0 text-muted-foreground tabular-nums'>{nh.km.toFixed(1)} km</span>
              <ArrowRight className='size-3 shrink-0' />
            </button>
          </div>
        )}
        {(ns || nh) && <p className='pt-1 text-[0.6875rem] text-muted-foreground'>Distances are straight-line. Whether a road gets there is the settlement's status: open it to see its road route.</p>}
      </>
    )
  }

  return (
    <div className='overflow-hidden rounded-lg border bg-background shadow-lg'>
      <div className='flex items-start gap-2 border-b px-3 py-2.5'>
        <div className='min-w-0 flex-1'>
          <div className='truncate text-[0.6875rem] text-muted-foreground capitalize'>{kicker}</div>
          <div className='truncate text-sm font-semibold'>{title}</div>
          {pinyinName && <div className='truncate text-[0.6875rem] text-muted-foreground'>Pinyin · {pinyinName}</div>}
          {devanagariName && <div className='[overflow-wrap:anywhere] text-[0.6875rem] leading-snug text-muted-foreground'>Latin transliteration · {devanagariName}</div>}
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
          <Button
            variant='ghost'
            size='sm'
            className='h-7 gap-1.5 px-2 text-xs'
            onClick={() => app.askAssistant(ask!, { id: str(p.osm_id), kind: s.kind, coordinates: s.lngLat })}
          >
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
