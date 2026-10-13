import { ArrowRight, Download, FileText } from 'lucide-react'
import { useApp, type View } from '@/context/app-state'
import { apiUrl } from '@/lib/api'
import { STATUS_COLOR, type useJobData } from '@/lib/job'
import { Button } from '@/components/ui/button'
import { Dot, List, Note, Row, Section, Stack, StatusBadge } from './common'

type Data = ReturnType<typeof useJobData>

function Tile({
  tag,
  view,
  label,
  value,
  unit,
  of,
  tone,
  sub,
  bar,
}: {
  tag: string
  view: View
  label: string
  value: string | null
  unit?: string
  of?: string | null
  tone?: string
  sub: string
  bar?: { value: number; color: string; label: string }[]
}) {
  const app = useApp()
  return (
    <button
      type='button'
      onClick={() => app.setView(view)}
      className='group min-w-0 border-b border-r p-3 text-left transition-colors even:border-r-0 hover:bg-muted/40 [&:nth-child(n+3)]:border-b-0'
    >
      <div className='flex items-center justify-between gap-1'>
        <span className='truncate text-xs font-medium text-foreground/80'>{label}</span>
        <span className='flex shrink-0 items-center gap-0.5 text-[0.625rem] font-semibold text-muted-foreground uppercase group-hover:text-foreground'>
          {tag} <ArrowRight className='size-3' />
        </span>
      </div>
      <div className='mt-1 flex items-baseline gap-1'>
        <span className={`text-2xl font-semibold tabular-nums tracking-tight ${tone ?? ''}`}>{value ?? '—'}</span>
        {unit && value != null && <span className='text-sm text-muted-foreground'>{unit}</span>}
        {of && <span className='text-sm text-muted-foreground'>of {of}</span>}
      </div>
      {bar && (
        <div className='mt-1.5'>
          <Stack parts={bar} />
        </div>
      )}
      <div className='mt-1 text-xs leading-snug text-foreground/70'>{sub}</div>
    </button>
  )
}

function Mini({ label, value, sub }: { label: string; value: string | null; sub?: string }) {
  return (
    <div className='min-w-0'>
      <div className='truncate text-[0.6875rem] text-foreground/70'>{label}</div>
      <div className='text-base font-semibold tabular-nums'>
        {value ?? '—'}
        {sub && <span className='ml-1 text-xs font-normal text-muted-foreground'>{sub}</span>}
      </div>
    </div>
  )
}

export function Overview({ d }: { d: Data }) {
  const app = useApp()
  const { t, n } = d.facts
  const id = d.job.data!.id
  const urgent = d.settlements.filter((s) => s.status === 'cut_off' || s.status === 'hospital_cut_off')
  const topRepair = d.repairs[0]
  const lost = d.bridges.filter((b) => b.status === 'likely_lost')

  return (
    <div className='space-y-5 p-3'>
      <div className='overflow-hidden rounded-lg border bg-card'>
        <div className='grid grid-cols-2'>
          <Tile
            tag='Access'
            view='cutoff'
            label='No road out'
            value={t('settlements_cut_off')}
            of={t('settlements_total')}
            tone={(n('settlements_cut_off') ?? 0) > 0 ? 'text-red-700 dark:text-red-400' : ''}
            sub={`communities · ${t('settlements_hospital_cut_off') ?? '—'} others can't reach a hospital`}
          />
          <Tile
            tag='Access'
            view='cutoff'
            label='Nearby buildings'
            value={t('buildings_in_cut_off')}
            sub={`within ${t('exposure_radius_m') ?? '—'} m · no population data`}
          />
          <Tile
            tag='Ground'
            view='flooding'
            label='Debris and water'
            value={t('area_debris_km2')}
            unit='km²'
            sub={`debris and scour · ${t('area_flood_km2') ?? '—'} km² new water`}
            bar={[
              { value: n('area_debris_km2') ?? 0, color: '#c27803', label: 'Debris' },
              { value: n('area_flood_km2') ?? 0, color: '#2563eb', label: 'New water' },
            ]}
          />
          <Tile
            tag='Assets'
            view='damage'
            label='Buildings touched'
            value={t('buildings_hit')}
            of={t('buildings_total')}
            tone='text-amber-700 dark:text-amber-400'
            sub='overlap mapped debris or water'
          />
        </div>
        <button
          type='button'
          onClick={() => app.setView('damage')}
          className='grid w-full grid-cols-3 gap-2 border-t px-3 py-2.5 text-left hover:bg-muted/40'
        >
          <Mini label='Impassable road' value={t('network_failed_km') != null ? `${t('network_failed_km')} km` : null} />
          <Mini
            label='Bridges washed out'
            value={t('bridges_lost')}
            sub={(n('bridges_at_risk') ?? 0) > 0 ? `+${t('bridges_at_risk')} unsure` : undefined}
          />
          <Mini label='Clinics isolated' value={t('hospitals_isolated')} sub={t('hospitals_total') ? `of ${t('hospitals_total')}` : undefined} />
        </button>
        <div className='border-t bg-muted/30 px-3 py-2 text-xs leading-relaxed text-foreground/80'>
          {(n('area_not_observable_km2') ?? 0) >= 0.01 ? (
            <>
              <b>{t('area_not_observable_km2')} km²</b> of the {t('aoi_area_km2') ?? '—'} km² area was out of the radar's view: unknown,
              not safe.{' '}
            </>
          ) : (
            <>The radar saw the whole {t('aoi_area_km2') ?? '—'} km² area. </>
          )}
          These are lower bounds: places with nothing mapped may still be affected.
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
                  <StatusBadge status={s.status} />
                </div>
                <div className='mt-0.5 text-[0.6875rem] text-muted-foreground'>{s.buildings} buildings within 500 m</div>
              </Row>
            ))}
            {topRepair && (
              <Row
                onClick={() => {
                  app.setView('cutoff')
                  const b = d.bridges.find((x) => x.id === topRepair.osmId)
                  if (b) app.select({ kind: 'bridge', props: b.raw, lngLat: b.at }, 15)
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
          <Button asChild variant='outline' size='sm' className='h-8 justify-start text-xs'>
            <a href={apiUrl(`/api/jobs/${id}/files/sitrep_en.pdf`)} target='_blank' rel='noreferrer'>
              <FileText className='size-3.5' /> Situation report
            </a>
          </Button>
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
