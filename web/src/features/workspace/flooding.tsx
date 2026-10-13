import { useQuery } from '@tanstack/react-query'
import { Eye, EyeOff } from 'lucide-react'
import { useApp, type LayerId } from '@/context/app-state'
import { api } from '@/lib/api'
import type { useJobData } from '@/lib/job'
import { Dot, Note, Section, Stack, ViewIntro } from './common'

type Data = ReturnType<typeof useJobData>

const MEMBER_LABEL: Record<string, string> = {
  s1_water_logratio: 'Radar change: new open water',
  s1_change_bidirectional: 'Radar change: debris and scour',
  ml_unet_kurosiwo: 'Segmentation model trained on Kuro Siwo',
  s2_indices: 'Optical: water, vegetation loss, bare soil',
  s2_mndwi: 'Optical: new water',
  s2_ndvi_bsi: 'Optical: plant loss and new bare ground',
  s1_zscore_change: 'Radar against several earlier dates: debris and scour',
  s1_zscore_water: 'Radar against several earlier dates: new water',
}

function LayerRow({ id, color, label, value, unit, note }: { id: LayerId; color: string; label: string; value: string | null; unit: string; note?: string }) {
  const app = useApp()
  const on = app.visible[id]
  return (
    <button
      type='button'
      onClick={() => app.toggle(id, !on)}
      className='flex w-full items-center gap-2.5 px-3 py-2 text-left text-xs hover:bg-muted/60'
      title={on ? 'Hide on map' : 'Show on map'}
    >
      <Dot color={color} className={on ? '' : 'opacity-30'} />
      <div className='min-w-0 flex-1'>
        <div className={on ? 'font-medium' : 'font-medium text-muted-foreground'}>{label}</div>
        {note && <div className='text-[0.6875rem] text-muted-foreground'>{note}</div>}
      </div>
      <span className='tabular-nums font-medium'>
        {value ?? '—'}
        {value != null && <span className='ml-0.5 font-normal text-muted-foreground'>{unit}</span>}
      </span>
      {on ? <Eye className='size-3.5 text-muted-foreground' /> : <EyeOff className='size-3.5 text-muted-foreground/50' />}
    </button>
  )
}

export function Flooding({ d }: { d: Data }) {
  const { t, n } = d.facts
  const model = useQuery({ queryKey: ['model'], queryFn: api.model, staleTime: Infinity })
  const members = (t('members') ?? '').split(',').map((m) => m.trim()).filter(Boolean)
  const usesModel = members.some((m) => m.startsWith('ml_'))
  const m = model.data?.deployed

  return (
    <div className='space-y-5 p-3'>
      <ViewIntro
        question='Where did the flood hit?'
        answer={
          <>
            Radar change between the before and after images marks <b>{t('area_debris_km2') ?? '—'} km²</b> of debris and scour
            and <b>{t('area_flood_km2') ?? '—'} km²</b> of new water. Click a row to show or hide it on the map.
          </>
        }
      />
      <Section title='Mapped areas' aside='click to show or hide'>
        <div className='divide-y overflow-hidden rounded-lg border bg-card'>
          <LayerRow id='flood' color='#2563eb' label='New water' value={t('area_flood_km2')} unit='km²' note='Water that was not there before the event' />
          <LayerRow id='debris' color='#c27803' label='Debris and scour' value={t('area_debris_km2')} unit='km²' note='Mud, rock or eroded ground' />
          <LayerRow id='water' color='#7dd3fc' label='Water before the event' value={t('area_permanent_water_km2')} unit='km²' note='Rivers and lakes, left out of the flood' />
          <LayerRow id='unobs' color='#a8a29e' label='Not observable' value={t('area_not_observable_km2')} unit='km²' note='Hidden from the radar. Not known to be safe.' />
        </div>
      </Section>

      <Section title='Confidence' aside={<LayerToggleText id='confidence' />}>
        <div className='rounded-lg border bg-card p-3'>
          <Stack
            parts={[
              { value: n('area_tier_high_km2') ?? 0, color: '#b91c1c', label: 'High' },
              { value: n('area_tier_moderate_km2') ?? 0, color: '#ea580c', label: 'Moderate' },
              { value: n('area_tier_low_km2') ?? 0, color: '#fbbf24', label: 'Low' },
            ]}
          />
          <div className='mt-2 space-y-1 text-[0.6875rem]'>
            {(
              [
                ['High', 'area_tier_high_km2', '#b91c1c'],
                ['Moderate', 'area_tier_moderate_km2', '#ea580c'],
                ['Low', 'area_tier_low_km2', '#fbbf24'],
              ] as const
            ).map(([label, key, c]) => (
              <div key={key} className='flex items-center gap-1.5'>
                <Dot color={c} />
                <span className='text-muted-foreground'>{label}</span>
                <span className='ml-auto font-medium tabular-nums'>{t(key) ?? '—'} km²</span>
              </div>
            ))}
          </div>
          <p className='mt-2 text-[0.6875rem] leading-relaxed text-muted-foreground'>
            High means probability at least 0.8 and two or more methods agree. Moderate means probability at least 0.65 or
            two methods agree. Low is a lead to check, not a finding.
          </p>
        </div>
      </Section>

      <Section title='Evidence used'>
        <div className='space-y-1.5 rounded-lg border bg-card p-3 text-xs'>
          {members.length === 0 && <span className='text-muted-foreground'>—</span>}
          {members.map((mm) => (
            <div key={mm} className='flex items-center gap-2'>
              <span className='size-1.5 rounded-full bg-foreground/60' />
              <span>{MEMBER_LABEL[mm] ?? mm.replaceAll('_', ' ')}</span>
            </div>
          ))}
          <div className='mt-2 grid grid-cols-2 gap-x-3 gap-y-1 border-t pt-2 text-[0.6875rem] text-muted-foreground'>
            <span>Sentinel-1 tracks</span>
            <span className='text-right font-medium text-foreground'>{t('s1_track_count') ?? '—'}</span>
            <span>Sentinel-2 pair</span>
            <span className='text-right font-medium text-foreground'>{t('s2_pair') ?? '—'}</span>
            <span>Grid</span>
            <span className='text-right font-medium text-foreground'>{t('resolution_m') ?? '—'} m</span>
          </div>
        </div>
      </Section>

      {usesModel && m && (
        <Section title='Segmentation model'>
          {m.mountain_held_out ? (
            <div className='rounded-lg border bg-card p-3 text-xs'>
              <div className='text-[0.6875rem] text-muted-foreground'>Flood F1 on high-relief events never seen in training</div>
              <div className='text-lg font-semibold tabular-nums'>{m.mountain.f1_flood?.toFixed(2) ?? '—'}</div>
              {m.mountain.event_f1_flood_ci95 && (
                <div className='text-[0.6875rem] text-muted-foreground'>
                  Per-event 95% interval {m.mountain.event_f1_flood_ci95[0].toFixed(2)} to {m.mountain.event_f1_flood_ci95[1].toFixed(2)}
                </div>
              )}
            </div>
          ) : (
            <Note tone='warn'>
              The model in use has no clean held-out score yet: its training split let the high-relief test tiles into training.
              Treat its contribution as unvalidated. See About.
            </Note>
          )}
          <Note>Model scores come from Kuro Siwo, not from this area.</Note>
        </Section>
      )}
    </div>
  )
}

function LayerToggleText({ id }: { id: LayerId }) {
  const app = useApp()
  const on = app.visible[id]
  return (
    <button type='button' className='underline-offset-2 hover:underline' onClick={() => app.toggle(id, !on)}>
      {on ? 'Hide on map' : 'Show on map'}
    </button>
  )
}
