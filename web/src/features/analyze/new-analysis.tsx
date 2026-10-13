import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ArrowLeft, Check, Loader2, MousePointerSquareDashed, Pentagon, Play, Satellite, Search, X } from 'lucide-react'
import { toast } from 'sonner'
import { useApp } from '@/context/app-state'
import { api, type Place, type Polygon } from '@/lib/api'
import { bounds } from '@/lib/job'
import { OFFSETS, eventIso } from '@/lib/time'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Calendar as CalendarPicker } from '@/components/ui/calendar'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'

/** Approximate geodesic area (km²) of a lon/lat polygon ring (EPSG:4326), for the size check only. */
function areaKm2(p: Polygon): number {
  const R = 6371.0088
  const ring = p.coordinates[0]
  let s = 0
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1, y1] = ring[i]
    const [x2, y2] = ring[i + 1]
    s += (((x2 - x1) * Math.PI) / 180) * (2 + Math.sin((y1 * Math.PI) / 180) + Math.sin((y2 * Math.PI) / 180))
  }
  return Math.abs((s * R * R) / 2)
}

function bboxPolygon(text: string): Polygon | null {
  const v = text.split(/[\s,]+/).filter(Boolean).map(Number)
  if (v.length !== 4 || v.some((n) => !Number.isFinite(n))) return null
  const [x0, y0, x1, y1] = v
  if (!(x0 < x1 && y0 < y1)) return null
  return { type: 'Polygon', coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]] }
}

function Step({ n, title, done, children }: { n: number; title: string; done: boolean; children: React.ReactNode }) {
  return (
    <section className='space-y-2.5'>
      <div className='flex items-center gap-2'>
        <span
          className={cn(
            'flex size-5 items-center justify-center rounded-full text-[0.6875rem] font-semibold',
            done ? 'bg-emerald-600 text-white' : 'bg-muted text-foreground'
          )}
        >
          {done ? <Check className='size-3' /> : n}
        </span>
        <h3 className='text-xs font-semibold tracking-tight'>{title}</h3>
      </div>
      <div className='space-y-2 pl-7'>{children}</div>
    </section>
  )
}

export function NewAnalysis() {
  const app = useApp()
  const cfg = useQuery({ queryKey: ['config'], queryFn: api.config })
  const [day, setDay] = useState<Date | undefined>()
  const [time, setTime] = useState('')
  const [offset, setOffset] = useState('+05:45')
  const [bbox, setBbox] = useState('')
  const [placeQuery, setPlaceQuery] = useState('')
  const [showPlaces, setShowPlaces] = useState(false)
  const [useS2, setUseS2] = useState(true)
  const [useMl, setUseMl] = useState(true)

  const places = useQuery({ queryKey: ['places', placeQuery], queryFn: () => api.places(placeQuery), enabled: showPlaces })
  const event = day ? eventIso(day, time, offset) : null
  const area = useMemo(() => (app.aoi ? areaKm2(app.aoi) : 0), [app.aoi])
  const cap = cfg.data?.max_aoi_km2 ?? 0
  const tooBig = !!app.aoi && cap > 0 && area > cap
  const ready = !!app.aoi && !!event && !tooBig

  const preview = useMutation({ mutationFn: () => api.scenes(app.aoi!, event!) })
  const run = useMutation({
    mutationFn: () => api.createJob({ aoi: app.aoi!, event: event!, use_s2: useS2, use_ml: useMl && !!cfg.data?.ml_model_available }),
    onSuccess: ({ id }) => {
      app.openJob(id)
      app.setView('overview')
    },
    onError: (e) => toast.error((e as Error).message),
  })

  const pickPlace = (p: Place) => {
    app.setAoi(p.polygon)
    app.fitTo(p.bbox)
    setPlaceQuery(p.name)
    setShowPlaces(false)
  }

  const caseStudy = () => {
    const p: Polygon = { type: 'Polygon', coordinates: [[[85.25, 27.95], [85.4, 27.95], [85.4, 28.2], [85.25, 28.2], [85.25, 27.95]]] }
    app.setAoi(p)
    app.fitTo(bounds(p.coordinates[0]))
    setDay(new Date(2026, 7, 26))
    setTime('')
    setBbox('85.25, 27.95, 85.40, 28.20')
  }

  return (
    <div className='flex min-h-0 flex-1 flex-col'>
      <div className='flex items-center gap-2 border-b px-3 py-2.5'>
        <Button variant='ghost' size='icon' className='size-7' onClick={() => app.setMode('workspace')} title='Back' aria-label='Back to the open analysis'>
          <ArrowLeft className='size-4' />
        </Button>
        <div className='min-w-0'>
          <div className='text-sm font-semibold tracking-tight'>New analysis</div>
          <div className='text-[0.6875rem] text-muted-foreground'>Any area, any flood date</div>
        </div>
        <button type='button' onClick={caseStudy} className='ml-auto text-[0.6875rem] text-muted-foreground underline-offset-2 hover:text-foreground hover:underline'>
          Load case study
        </button>
      </div>

      <div className='min-h-0 flex-1 overflow-y-auto'>
        <div className='space-y-6 p-3'>
          <Step n={1} title='Area' done={!!app.aoi && !tooBig}>
            <div className='grid grid-cols-2 gap-1.5'>
              <Button
                size='sm'
                variant={app.drawMode === 'box' ? 'default' : 'outline'}
                className='h-8 justify-start gap-1.5 text-xs'
                onClick={() => app.setDrawMode(app.drawMode === 'box' ? 'none' : 'box')}
              >
                <MousePointerSquareDashed className='size-3.5' /> {app.drawMode === 'box' ? 'Drag on map…' : 'Draw box'}
              </Button>
              <Button
                size='sm'
                variant={app.drawMode === 'polygon' ? 'default' : 'outline'}
                className='h-8 justify-start gap-1.5 text-xs'
                onClick={() => app.setDrawMode(app.drawMode === 'polygon' ? 'none' : 'polygon')}
              >
                <Pentagon className='size-3.5' /> {app.drawMode === 'polygon' ? 'Double-click ends' : 'Draw polygon'}
              </Button>
            </div>
            <div className='relative'>
              <Search className='absolute top-2.5 left-2.5 size-3.5 text-muted-foreground' />
              <Input
                placeholder='Search a place'
                value={placeQuery}
                onChange={(e) => {
                  setPlaceQuery(e.target.value)
                  setShowPlaces(true)
                }}
                onFocus={() => setShowPlaces(true)}
                onBlur={() => setTimeout(() => setShowPlaces(false), 150)}
                className='h-8 pl-8 text-xs'
              />
              {showPlaces && (places.data?.length ?? 0) > 0 && (
                <div className='absolute z-50 mt-1 max-h-56 w-full overflow-auto rounded-md border bg-popover p-1 shadow-md'>
                  {places.data!.map((p) => (
                    <button
                      type='button'
                      key={p.id}
                      className='flex w-full items-start justify-between gap-2 rounded px-2 py-1.5 text-left text-xs hover:bg-accent'
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => pickPlace(p)}
                    >
                      <span>
                        <span className='font-medium'>{p.name}</span>
                        <span className='block text-[0.6875rem] text-muted-foreground'>{p.description}</span>
                      </span>
                      <span className='text-[0.625rem] text-muted-foreground uppercase'>{p.category}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className='flex gap-1.5'>
              <Input
                placeholder='or bounding box: W, S, E, N'
                value={bbox}
                onChange={(e) => setBbox(e.target.value)}
                className='h-8 font-mono text-xs'
              />
              <Button
                size='sm'
                variant='secondary'
                className='h-8 shrink-0 text-xs'
                onClick={() => {
                  const p = bboxPolygon(bbox)
                  if (!p) return toast.error('Enter four numbers: west, south, east, north (degrees)')
                  app.setAoi(p)
                  app.fitTo(bounds(p.coordinates[0]))
                }}
              >
                Apply
              </Button>
            </div>
            {app.aoi && (
              <div className={cn('flex items-center justify-between text-[0.6875rem]', tooBig ? 'text-destructive' : 'text-muted-foreground')}>
                <span>
                  {area.toFixed(1)} km² {tooBig && `is over the ${cap.toFixed(0)} km² limit for a live run`}
                </span>
                <button type='button' className='inline-flex items-center gap-1 hover:text-foreground' onClick={() => app.setAoi(null)}>
                  <X className='size-3' /> Clear
                </button>
              </div>
            )}
          </Step>

          <Step n={2} title='Flood date' done={!!event}>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant='outline' size='sm' className='h-8 w-full justify-start text-xs font-normal'>
                  {day ? day.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' }) : 'Pick a date'}
                </Button>
              </PopoverTrigger>
              <PopoverContent className='w-auto p-0' align='start'>
                <CalendarPicker mode='single' selected={day} onSelect={setDay} captionLayout='dropdown' defaultMonth={day ?? new Date(2026, 7, 1)} />
              </PopoverContent>
            </Popover>
            <div className='grid grid-cols-2 gap-1.5'>
              <div className='space-y-1'>
                <Label className='text-[0.6875rem] text-muted-foreground'>Time, if known</Label>
                <Input type='time' value={time} onChange={(e) => setTime(e.target.value)} className='h-8 text-xs' />
              </div>
              <div className='space-y-1'>
                <Label className='text-[0.6875rem] text-muted-foreground'>Time zone</Label>
                <Select value={offset} onValueChange={setOffset}>
                  <SelectTrigger size='sm' className='h-8 w-full text-xs'>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {OFFSETS.map((o) => (
                      <SelectItem key={o.value} value={o.value} className='text-xs'>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <p className='text-[0.6875rem] text-muted-foreground'>
              Radar passes from the same orbit track before and after this time are compared. The time decides which same-day pass
              counts as "after".
            </p>
          </Step>

          <Step n={3} title='Evidence' done>
            <div className='divide-y rounded-lg border'>
              <div className='flex items-center justify-between gap-2 px-3 py-2 text-xs'>
                <div>
                  <div className='font-medium'>Sentinel-1 radar</div>
                  <div className='text-[0.6875rem] text-muted-foreground'>Sees through cloud. Always used.</div>
                </div>
                <Switch checked disabled />
              </div>
              <div className='flex items-center justify-between gap-2 px-3 py-2 text-xs'>
                <div>
                  <div className='font-medium'>Sentinel-2 optical</div>
                  <div className='text-[0.6875rem] text-muted-foreground'>Used where the sky is clear</div>
                </div>
                <Switch checked={useS2} onCheckedChange={setUseS2} />
              </div>
              <div className='flex items-center justify-between gap-2 px-3 py-2 text-xs'>
                <div>
                  <div className='font-medium'>Segmentation model</div>
                  <div className='text-[0.6875rem] text-muted-foreground'>
                    {cfg.data?.ml_model_available ? 'Trained on Kuro Siwo' : 'No trained model installed'}
                  </div>
                </div>
                <Switch checked={useMl && !!cfg.data?.ml_model_available} onCheckedChange={setUseMl} disabled={!cfg.data?.ml_model_available} />
              </div>
            </div>
          </Step>

          {preview.error && (
            <Alert variant='destructive'>
              <AlertDescription className='text-xs'>{(preview.error as Error).message}</AlertDescription>
            </Alert>
          )}
          {preview.data && (
            <div className='space-y-1.5 rounded-lg border p-3'>
              <div className='text-xs font-semibold'>
                {preview.data.pairs.length} usable radar pair{preview.data.pairs.length === 1 ? '' : 's'}
              </div>
              {preview.data.pairs.map((p) => (
                <div key={p.relative_orbit} className='flex items-center justify-between text-[0.6875rem]'>
                  <span className='font-medium'>
                    Track {p.relative_orbit} · {p.orbit_state.toLowerCase()}
                  </span>
                  <span className='tabular-nums text-muted-foreground'>
                    {p.pre.start.slice(0, 10)} → {p.post.start.slice(0, 10)} · {Math.round(Math.min(p.pre.aoi_coverage, p.post.aoi_coverage) * 100)}% cover
                  </span>
                </div>
              ))}
              {preview.data.notes.map((n) => (
                <p key={n} className='text-[0.6875rem] text-muted-foreground'>
                  {n}
                </p>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className='grid grid-cols-2 gap-1.5 border-t p-3'>
        <Button variant='outline' size='sm' className='h-9 gap-1.5 text-xs' disabled={!ready || preview.isPending} onClick={() => preview.mutate()}>
          {preview.isPending ? <Loader2 className='size-3.5 animate-spin' /> : <Satellite className='size-3.5' />}
          Check coverage
        </Button>
        <Button size='sm' className='h-9 gap-1.5 text-xs' disabled={!ready || run.isPending} onClick={() => run.mutate()}>
          {run.isPending ? <Loader2 className='size-3.5 animate-spin' /> : <Play className='size-3.5' />}
          Run analysis
        </Button>
      </div>
    </div>
  )
}
