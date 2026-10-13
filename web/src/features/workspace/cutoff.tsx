import { useMemo, useState } from 'react'
import { HeartPulse, Search } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { STATUS_COLOR, STATUS_LABEL, fmtHours, type Settlement, type useJobData } from '@/lib/job'
import { Input } from '@/components/ui/input'
import { nameMatches } from '@/lib/translit'
import { cn } from '@/lib/utils'
import { Dot, Empty, List, Note, Row, Section, StatusBadge } from './common'

type Data = ReturnType<typeof useJobData>
const FILTERS = ['cut_off', 'hospital_cut_off', 'detour', 'connected', 'no_baseline_route', 'not_on_network'] as const

function access(s: Settlement): string {
  if (s.status === 'cut_off') return 'No road out to a town, health facility or beyond the area'
  if (s.status === 'hospital_cut_off') return `No road to ${s.hospital || 'a hospital'}`
  if (s.status === 'detour')
    return `${fmtHours(s.hoursAfter)} to ${s.hospital || 'hospital'}${s.delayHours ? `, ${fmtHours(s.delayHours)} longer` : ''}`
  if (s.status === 'connected') return `${fmtHours(s.hoursAfter ?? s.hoursBefore)} to ${s.hospital || 'hospital'}`
  if (s.status === 'no_baseline_route') return 'No road route existed before the event'
  return 'Not on the mapped road network'
}

export function CutOff({ d }: { d: Data }) {
  const app = useApp()
  const { t } = d.facts
  const counts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const s of d.settlements) c[s.status] = (c[s.status] ?? 0) + 1
    return c
  }, [d.settlements])
  const firstAffected = FILTERS.find((f) => (counts[f] ?? 0) > 0 && ['cut_off', 'hospital_cut_off', 'detour'].includes(f))
  const [filter, setFilter] = useState<string | 'all'>(firstAffected ?? 'all')
  const [q, setQ] = useState('')
  const sel = app.selection
  const rows = d.settlements.filter(
    (s) => (filter === 'all' || s.status === filter) && nameMatches(s.name, q)
  )

  const selectSettlement = (s: Settlement) => app.select({ kind: 'settlement', props: s.raw, lngLat: s.at }, 13)

  return (
    <div className='space-y-5 p-3'>
      <Section title='Settlements' aside={`${d.settlements.length} mapped`}>
        <div className='flex flex-wrap gap-1'>
          <Chip active={filter === 'all'} onClick={() => setFilter('all')}>
            All <span className='tabular-nums text-muted-foreground'>{d.settlements.length}</span>
          </Chip>
          {FILTERS.filter((f) => counts[f]).map((f) => (
            <Chip key={f} active={filter === f} onClick={() => setFilter(f)}>
              <Dot color={STATUS_COLOR[f]} className='size-2' />
              {STATUS_LABEL[f]} <span className='tabular-nums text-muted-foreground'>{counts[f]}</span>
            </Chip>
          ))}
        </div>
        <div className='relative'>
          <Search className='absolute top-2 left-2.5 size-3.5 text-muted-foreground' />
          <Input className='h-8 pl-8 text-xs' placeholder='Find a settlement' value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {rows.length === 0 ? (
          <Empty>{q ? 'No settlement matches.' : 'None in this group.'}</Empty>
        ) : (
          <List className='max-h-[23.75rem] overflow-y-auto'>
            {rows.slice(0, 300).map((s) => (
              <Row key={s.id} at={s.at} active={sel?.kind === 'settlement' && sel.props.osm_id === s.raw.osm_id} onClick={() => selectSettlement(s)}>
                <div className='flex items-center gap-2'>
                  <span className='truncate font-medium'>{s.name}</span>
                  <span className='ml-auto shrink-0 text-[0.6875rem] tabular-nums text-muted-foreground'>{s.buildings} bldg</span>
                  <StatusBadge status={s.status} />
                </div>
                <div className='mt-0.5 truncate text-[0.6875rem] text-muted-foreground'>{access(s)}</div>
              </Row>
            ))}
          </List>
        )}
        <p className='text-[0.6875rem] text-muted-foreground'>
          Ranked by severity, then by buildings within {t('exposure_radius_m') ?? '—'} m. Population is not recorded in the inputs.
        </p>
      </Section>

      <Section title='Repairs that reconnect the most' aside={d.repairs.length ? `${d.repairs.length} candidates` : undefined}>
        {d.repairs.length === 0 ? (
          <Empty>No single road or bridge repair reconnects a cut-off settlement.</Empty>
        ) : (
          <List>
            {d.repairs.slice(0, 10).map((r, i) => {
              const b = d.bridges.find((x) => x.id === r.osmId)
              const s = d.settlements.find((x) => r.settlementNames.includes(x.name))
              return (
                <Row
                  key={r.osmId}
                  onClick={() => {
                    if (b) app.select({ kind: 'bridge', props: b.raw, lngLat: b.at }, 15)
                    else if (s) selectSettlement(s)
                  }}
                >
                  <div className='flex items-center gap-2'>
                    <span className='flex size-5 shrink-0 items-center justify-center rounded bg-muted text-[0.6875rem] font-semibold tabular-nums'>{i + 1}</span>
                    <span className='truncate font-medium'>
                      {r.isBridge ? 'Bridge' : 'Road'} {r.name || `(${r.highway || 'unnamed'})`}
                    </span>
                  </div>
                  <div className='mt-0.5 pl-7 text-[0.6875rem] text-muted-foreground'>
                    Reconnects {r.settlements} settlement{r.settlements === 1 ? '' : 's'} · {r.buildings} buildings
                    {r.settlementNames.length > 0 && <> · {r.settlementNames.slice(0, 3).join(', ')}</>}
                  </div>
                </Row>
              )
            })}
          </List>
        )}
      </Section>

      <Section title='Hospitals and clinics' aside={`${d.hospitals.length} mapped`}>
        {d.hospitals.length === 0 ? (
          <Empty>No hospital or clinic is mapped in or near this area, so hospital access cannot be judged.</Empty>
        ) : (
          <List>
            {d.hospitals.map((h) => (
              <Row key={h.id} at={h.at} active={sel?.kind === 'hospital' && sel.props.osm_id === h.raw.osm_id} onClick={() => app.select({ kind: 'hospital', props: h.raw, lngLat: h.at }, 15)}>
                <div className='flex items-center gap-2'>
                  <HeartPulse className='size-3.5 shrink-0 text-red-600' />
                  <span className='truncate font-medium'>{h.name}</span>
                  <span className='text-[0.6875rem] text-muted-foreground'>{h.amenity}</span>
                  <StatusBadge status={h.isolated ? 'cut_off' : h.status} className='ml-auto' />
                </div>
              </Row>
            ))}
          </List>
        )}
      </Section>

      <Note>
        A settlement is cut off when no road or trail in the pre-event map still links it to a town, a health facility or the edge of
        the analysed area, once links crossed by flood or debris are removed. Travel times assume typical speeds per road type and
        are a guide only.
      </Note>
    </div>
  )
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type='button'
      onClick={onClick}
      className={cn(
        'inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-[0.6875rem] font-medium transition-colors',
        active ? 'border-foreground/30 bg-foreground/5 text-foreground' : 'text-muted-foreground hover:bg-muted hover:text-foreground'
      )}
    >
      {children}
    </button>
  )
}
