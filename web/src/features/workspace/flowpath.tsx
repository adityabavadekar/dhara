import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Crosshair, Loader2, Play, X } from 'lucide-react'
import { toast } from 'sonner'
import { useApp } from '@/context/app-state'
import { api } from '@/lib/api'
import { num, str, useJob } from '@/lib/job'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { ElevationProfile } from '@/components/elevation-profile'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'
import { Empty, List, Note, Row, Section, ViewIntro } from './common'

type Meta = {
  length_km?: number
  drop_m?: number
  settlements?: number
  stopped_because?: string
  simulation?: { total_transit_time_min?: number }
}

const EXPOSURE: Record<string, { label: string; color: string }> = {
  direct_path: { label: 'In the path', color: '#b91c1c' },
  fringe: { label: 'Edge of path', color: '#d97706' },
  ridge_safe: { label: 'Well above river', color: '#16a34a' },
}

export function FlowPath() {
  const app = useApp()
  const [len, setLen] = useState(40)
  const [corr, setCorr] = useState(600)
  const id = app.flowJobId
  const job = useJob(id)
  const done = job.data?.status === 'done'
  const sett = useQuery({ queryKey: ['fp', id], queryFn: () => api.geojson(id!, 'settlements.geojson'), enabled: !!id && done, staleTime: Infinity })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  const traces = (jobs.data ?? []).filter((j) => j.kind === 'flowpath' && j.status === 'done').slice(0, 6)

  const run = useMutation({
    mutationFn: () => api.createFlowPath({ lon: app.flowPoint![0], lat: app.flowPoint![1], length_km: len, corridor_m: corr }),
    onSuccess: ({ id }) => app.setFlowJobId(id),
    onError: (e) => toast.error((e as Error).message),
  })
  const meta = (job.data?.summary ?? {}) as Meta
  const busy = run.isPending || job.data?.status === 'running' || job.data?.status === 'queued'
  const picking = app.drawMode === 'flowpoint'
  const rows = ((sett.data?.features ?? []) as GeoJSON.Feature[]).map((f) => ({
    p: f.properties as Record<string, unknown>,
    at: (f.geometry as GeoJSON.Point).coordinates as [number, number],
  }))

  return (
    <div className='space-y-5 p-3'>
      <ViewIntro
        question='Where would a surge go?'
        answer={
          <>
            Pick a point upstream. The path follows the steepest way downhill on the Copernicus DEM, along the valley floor
            (not along roads), and lists the settlements beside it in order.
          </>
        }
      />
      <Section title='Trace a flood path'>
        <p className='text-[0.6875rem] leading-relaxed text-muted-foreground'>
          Pick a point upstream, such as a glacier lake or landslide dam. The path follows the steepest descent on the Copernicus DEM
          and lists the settlements along it, in order.
        </p>
        <div className='flex gap-1.5'>
          <Button
            size='sm'
            variant={picking ? 'default' : 'outline'}
            className='h-8 flex-1 justify-start gap-1.5 text-xs'
            onClick={() => app.setDrawMode(picking ? 'none' : 'flowpoint')}
          >
            <Crosshair className='size-3.5' />
            {picking ? 'Click the map…' : app.flowPoint ? `${app.flowPoint[1].toFixed(4)}°N ${app.flowPoint[0].toFixed(4)}°E` : 'Pick start on map'}
          </Button>
          {app.flowPoint && (
            <Button size='icon' variant='ghost' className='size-8' onClick={() => app.setFlowPoint(null)} title='Clear start point'>
              <X className='size-3.5' />
            </Button>
          )}
        </div>
        <div className='grid grid-cols-2 gap-2'>
          <div className='space-y-1'>
            <Label className='text-[0.6875rem] text-muted-foreground'>Length (km)</Label>
            <Input type='number' min={2} max={120} value={len} onChange={(e) => setLen(Number(e.target.value))} className='h-8 text-xs' />
          </div>
          <div className='space-y-1'>
            <Label className='text-[0.6875rem] text-muted-foreground'>Search width each side (m)</Label>
            <Input type='number' min={50} max={3000} value={corr} onChange={(e) => setCorr(Number(e.target.value))} className='h-8 text-xs' />
          </div>
        </div>
        <Button className='h-8 w-full gap-1.5 text-xs' disabled={!app.flowPoint || busy} onClick={() => run.mutate()}>
          {busy ? <Loader2 className='size-3.5 animate-spin' /> : <Play className='size-3.5' />}
          {busy ? 'Tracing…' : 'Trace path'}
        </Button>
        {job.data?.status === 'failed' && (
          <Alert variant='destructive'>
            <AlertDescription className='text-xs'>{job.data.error}</AlertDescription>
          </Alert>
        )}
      </Section>

      {done && (
        <Section title='Along the path' aside={`${meta.settlements ?? rows.length} settlements`}>
          <div className='grid grid-cols-3 gap-px overflow-hidden rounded-lg border bg-border text-xs'>
            <Cell label='Length' value={meta.length_km != null ? `${meta.length_km.toFixed(1)} km` : '—'} />
            <Cell label='Drop' value={meta.drop_m != null ? `${Math.round(meta.drop_m)} m` : '—'} />
            <Cell label='Stopped at' value={meta.stopped_because ?? '—'} />
          </div>

          {id && (
            <ElevationProfile
              jobId={id}
              path={[]}
              title='Debris Surge Elevation Profile'
              subtitle='Steepest-descent river bed cross-section'
            />
          )}

          {rows.length === 0 ? (
            <Empty>No mapped settlement lies within {corr} m of this path.</Empty>
          ) : (
            <List className='max-h-[22.5rem] overflow-y-auto'>
              {rows.map(({ p, at }) => {
                const ex = EXPOSURE[str(p.exposure_status)]
                const arr = num(p.arrival_time_min)
                return (
                  <Row key={str(p.osm_id) || str(p.order)} at={at} onClick={() => app.select({ kind: 'flow-settlement', props: p, lngLat: at }, 14)}>
                    <div className='flex items-center gap-2'>
                      <span className='flex size-5 shrink-0 items-center justify-center rounded bg-violet-600/10 text-[0.6875rem] font-semibold text-violet-700 tabular-nums dark:text-violet-300'>
                        {str(p.order)}
                      </span>
                      <span className='truncate font-medium'>{str(p.name) || 'Unnamed settlement'}</span>
                      {ex && (
                        <span className='ml-auto shrink-0 text-[0.6875rem] font-medium' style={{ color: ex.color }}>
                          {ex.label}
                        </span>
                      )}
                    </div>
                    <div className='mt-0.5 pl-7 text-[0.6875rem] text-muted-foreground'>
                      {num(p.distance_km)?.toFixed(1) ?? '—'} km down · {num(p.delta_z_river_m)?.toFixed(0) ?? '—'} m above river
                      {arr != null && <> · reached in roughly {Math.round(arr)} min</>}
                    </div>
                  </Row>
                )
              })}
            </List>
          )}
          <Note tone='warn'>
            Arrival times use an assumed slope-to-speed rule that has not been calibrated against observed floods. Use them to order
            warnings, not as forecasts. Sentinel satellites revisit every few days, so this cannot warn of a collapse as it happens.
          </Note>
        </Section>
      )}

      {traces.length > 0 && (
        <Section title='Earlier traces'>
          <List>
            {traces.map((j) => {
              const r = j.request as { lon?: number; lat?: number; length_km?: number }
              return (
                <Row
                  key={j.id}
                  active={j.id === id}
                  onClick={() => {
                    app.setFlowJobId(j.id)
                    if (r.lon != null && r.lat != null) app.setFlowPoint([r.lon, r.lat])
                  }}
                >
                  <div className={cn('flex items-center gap-2', j.id === id && 'font-medium')}>
                    <span className='tabular-nums'>
                      {r.lat?.toFixed(3)}°N {r.lon?.toFixed(3)}°E
                    </span>
                    <span className='ml-auto text-[0.6875rem] text-muted-foreground'>
                      {(j.summary as Meta | null)?.settlements ?? '—'} settlements · {r.length_km ?? '—'} km
                    </span>
                  </div>
                </Row>
              )
            })}
          </List>
        </Section>
      )}
    </div>
  )
}

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <div className='min-w-0 bg-card p-2.5'>
      <div className='text-[0.6875rem] text-muted-foreground'>{label}</div>
      <div className='truncate font-medium tabular-nums'>{value}</div>
    </div>
  )
}
