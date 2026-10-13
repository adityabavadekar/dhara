import type { ReactNode } from 'react'
import { ArrowRight, Building2, Download, FileText, Route, Waves } from 'lucide-react'
import { useApp, type View } from '@/context/app-state'
import { apiUrl } from '@/lib/api'
import { STATUS_COLOR, type useJobData } from '@/lib/job'
import { Button } from '@/components/ui/button'
import { Dot, List, Note, Row, Section, Stack, StatusBadge } from './common'

type Data = ReturnType<typeof useJobData>

function Answer({
  n,
  question,
  view,
  icon,
  children,
}: {
  n: number
  question: string
  view: View
  icon: ReactNode
  children: ReactNode
}) {
  const app = useApp()
  return (
    <button
      type='button'
      onClick={() => app.setView(view)}
      className='group block w-full rounded-lg border bg-card p-3 text-left transition-colors hover:border-foreground/20 hover:bg-muted/30'
    >
      <div className='mb-2.5 flex items-center gap-2'>
        <span className='flex size-5.5 items-center justify-center rounded-md bg-muted text-foreground/90 shrink-0'>{icon}</span>
        <span className='text-xs font-semibold text-foreground tracking-tight'>
          {n}. {question}
        </span>
        <ArrowRight className='ml-auto size-3.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100' />
      </div>
      {children}
    </button>
  )
}

const Big = ({ v, unit, tone }: { v: string | null; unit?: string; tone?: string }) => (
  <span className={`text-2xl font-semibold tabular-nums tracking-tight ${tone ?? ''}`}>
    {v ?? '—'}
    {unit && v != null && <span className='ml-1 text-sm font-normal text-muted-foreground'>{unit}</span>}
  </span>
)

export function Overview({ d }: { d: Data }) {
  const app = useApp()
  const { t, n } = d.facts
  const id = d.job.data!.id
  const urgent = d.settlements.filter((s) => s.status === 'cut_off' || s.status === 'hospital_cut_off')
  const topRepair = d.repairs[0]
  const lost = d.bridges.filter((b) => b.status === 'likely_lost')

  return (
    <div className='space-y-5 p-3'>
      <div className='space-y-2'>
        <Answer n={1} question='Where did the flood hit?' view='flooding' icon={<Waves className='size-3.5' />}>
          <div className='flex items-baseline gap-4'>
            <div>
              <Big v={t('area_flood_km2')} unit='km²' />
              <div className='text-xs text-muted-foreground'>new water</div>
            </div>
            <div>
              <Big v={t('area_debris_km2')} unit='km²' />
              <div className='text-xs text-muted-foreground'>debris and scour</div>
            </div>
          </div>
          <div className='mt-2.5'>
            <Stack
              parts={[
                { value: n('area_flood_km2') ?? 0, color: '#2563eb', label: 'New water' },
                { value: n('area_debris_km2') ?? 0, color: '#c27803', label: 'Debris' },
                { value: n('area_not_observable_km2') ?? 0, color: '#a8a29e', label: 'Not observable' },
              ]}
            />
          </div>
          <p className='mt-1.5 text-xs text-muted-foreground leading-relaxed'>
            {(n('area_not_observable_km2') ?? 0) >= 0.01 ? (
              <>
                {t('area_not_observable_km2')} km² of the {t('aoi_area_km2') ?? '—'} km² area was hidden from the radar and is not known to be
                safe.
              </>
            ) : (
              <>All of the {t('aoi_area_km2') ?? '—'} km² area was visible to the radar.</>
            )}
          </p>
        </Answer>

        <Answer n={2} question='What was damaged?' view='damage' icon={<Building2 className='size-3.5' />}>
          <div className='grid grid-cols-3 gap-2'>
            <div>
              <Big v={t('buildings_hit')} tone='text-amber-700 dark:text-amber-400' />
              <div className='text-xs text-muted-foreground'>of {t('buildings_total') ?? '—'} buildings</div>
            </div>
            <div>
              <Big v={t('network_failed_km')} unit='km' />
              <div className='text-xs text-muted-foreground'>road impassable</div>
            </div>
            <div>
              <Big v={t('bridges_lost')} tone={(n('bridges_lost') ?? 0) > 0 ? 'text-red-700 dark:text-red-400' : ''} />
              <div className='text-xs text-muted-foreground'>of {t('bridges_total') ?? '—'} bridges lost</div>
            </div>
          </div>
          {(n('bridges_at_risk') ?? 0) > 0 && (
            <p className='mt-1.5 text-xs text-muted-foreground'>{t('bridges_at_risk')} more bridges at risk.</p>
          )}
        </Answer>

        <Answer n={3} question='Who is cut off?' view='cutoff' icon={<Route className='size-3.5' />}>
          <div className='flex items-baseline gap-4'>
            <div>
              <Big v={t('settlements_cut_off')} tone={(n('settlements_cut_off') ?? 0) > 0 ? 'text-red-700 dark:text-red-400' : ''} />
              <div className='text-xs text-muted-foreground'>of {t('settlements_total') ?? '—'} settlements cut off</div>
            </div>
            <div>
              <Big v={t('settlements_hospital_cut_off')} />
              <div className='text-xs text-muted-foreground'>lost hospital access</div>
            </div>
            <div>
              <Big v={t('settlements_detour')} />
              <div className='text-xs text-muted-foreground'>longer route</div>
            </div>
          </div>
          {(n('buildings_in_cut_off') ?? 0) > 0 && (
            <p className='mt-1.5 text-[0.6875rem] text-muted-foreground'>
              {t('buildings_in_cut_off')} buildings within {t('exposure_radius_m') ?? '—'} m of cut-off settlements.
            </p>
          )}
        </Answer>
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
