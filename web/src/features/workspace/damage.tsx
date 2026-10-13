import { useState } from 'react'
import { GraduationCap, HeartPulse, Landmark } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { pct, type useJobData } from '@/lib/job'
import { Dot, Empty, Figure, List, Note, Row, Section, Stack, StatusBadge } from './common'

type Data = ReturnType<typeof useJobData>

const FAC_ICON = { school: GraduationCap, health: HeartPulse, community: Landmark } as const

export function Damage({ d }: { d: Data }) {
  const app = useApp()
  const { t, n } = d.facts
  const [showAllBridges, setShowAllBridges] = useState(false)
  const sel = app.selection
  const facHit = d.facilities.filter((f) => f.hit)
  const bridges = showAllBridges ? d.bridges : d.bridges.slice(0, 8)

  return (
    <div className='space-y-5 p-3'>
      <div className='grid grid-cols-3 gap-3 rounded-lg border bg-card p-3'>
        <Figure label='Buildings hit' value={t('buildings_hit')} of={t('buildings_total')} tone='text-amber-700 dark:text-amber-400' />
        <Figure label='Road impassable' value={t('network_failed_km')} unit='km' hint={`of ${t('network_km') ?? '—'} km`} />
        <Figure label='Bridges lost' value={t('bridges_lost')} of={t('bridges_total')} hint={`${t('bridges_at_risk') ?? '—'} at risk`} />
      </div>

      <Section title='Buildings in the hazard zone' aside='by confidence'>
        <div className='rounded-lg border bg-card p-3'>
          <Stack
            parts={[
              { value: n('buildings_hit_high') ?? 0, color: '#b91c1c', label: 'High' },
              { value: n('buildings_hit_moderate') ?? 0, color: '#ea580c', label: 'Moderate' },
              { value: n('buildings_hit_low') ?? 0, color: '#fbbf24', label: 'Low' },
            ]}
          />
          <div className='mt-2 space-y-1 text-[0.6875rem]'>
            {(
              [
                ['High', 'buildings_hit_high', '#b91c1c'],
                ['Moderate', 'buildings_hit_moderate', '#ea580c'],
                ['Low', 'buildings_hit_low', '#fbbf24'],
              ] as const
            ).map(([label, key, c]) => (
              <div key={key} className='flex items-center gap-1.5'>
                <Dot color={c} />
                <span className='text-muted-foreground'>{label}</span>
                <span className='ml-auto font-medium tabular-nums'>{t(key) ?? '—'}</span>
              </div>
            ))}
          </div>
          {(n('buildings_unobserved') ?? 0) > 0 && (
            <p className='mt-2 text-[0.6875rem] text-muted-foreground'>
              {t('buildings_unobserved')} {n('buildings_unobserved') === 1 ? 'building sits' : 'buildings sit'} where the radar could not
              see. Their state is unknown.
            </p>
          )}
        </div>
      </Section>

      <Section title='Schools and health facilities' aside={d.facilities.length ? `${facHit.length} of ${d.facilities.length} in hazard zone` : undefined}>
        {d.facilities.length === 0 ? (
          <Empty>No schools or health facilities are tagged in the pre-event map of this area.</Empty>
        ) : (
          <List className='max-h-64 overflow-y-auto'>
            {d.facilities.slice(0, 60).map((f) => {
              const Icon = FAC_ICON[f.type as keyof typeof FAC_ICON] ?? Landmark
              return (
                <Row
                  key={f.id}
                  at={f.at}
                  active={sel?.kind === 'building' && sel.props.osm_id === f.raw.osm_id}
                  onClick={() => app.select({ kind: 'building', props: f.raw, lngLat: f.at }, 17)}
                >
                  <div className='flex items-center gap-2'>
                    <Icon className='size-3.5 shrink-0 text-muted-foreground' />
                    <span className='truncate font-medium'>{f.name}</span>
                    <StatusBadge status={f.hit ? 'hit' : 'not_hit'} className='ml-auto' />
                  </div>
                  {f.hit && (
                    <div className='mt-0.5 pl-5.5 text-[0.6875rem] text-muted-foreground'>
                      {f.hazard || 'hazard'} · {f.confidence || '—'} confidence · peak probability {pct(f.pMax)}
                    </div>
                  )}
                </Row>
              )
            })}
          </List>
        )}
      </Section>

      <Section title='Bridges' aside={`${d.bridges.length} mapped`}>
        {d.bridges.length === 0 ? (
          <Empty>No bridges are mapped in this area.</Empty>
        ) : (
          <>
            <List>
              {bridges.map((b) => (
                <Row
                  key={b.id}
                  at={b.at}
                  active={sel?.kind === 'bridge' && sel.props.osm_id === b.raw.osm_id}
                  onClick={() => app.select({ kind: 'bridge', props: b.raw, lngLat: b.at }, 16)}
                >
                  <div className='flex items-center gap-2'>
                    <span className='truncate font-medium'>{b.name || `Unnamed ${b.highway || 'bridge'}`}</span>
                    <StatusBadge status={b.status} className='ml-auto' />
                  </div>
                  <div className='mt-0.5 text-[0.6875rem] text-muted-foreground'>
                    Chance lost {pct(b.pLost)}
                    {b.dropDb != null && <> · radar return {b.dropDb >= 0 ? 'fell' : 'rose'} {Math.abs(b.dropDb).toFixed(1)} dB</>}
                    {b.lengthM != null && <> · {Math.round(b.lengthM)} m span</>}
                  </div>
                </Row>
              ))}
            </List>
            {d.bridges.length > 8 && (
              <button type='button' className='text-[0.6875rem] text-muted-foreground underline-offset-2 hover:underline' onClick={() => setShowAllBridges((v) => !v)}>
                {showAllBridges ? 'Show fewer' : `Show all ${d.bridges.length}`}
              </button>
            )}
            <Note>
              A standing bridge reflects radar strongly. A fall of about 3 dB or more in that return is the main sign of loss. Flood
              at both ends alone only makes a bridge "at risk". Spans shorter than two pixels are kept at low confidence.
            </Note>
          </>
        )}
      </Section>
    </div>
  )
}
