import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { CalendarIcon, Eraser, MousePointer2, Pentagon, Play, Satellite, Square } from 'lucide-react'
import { toast } from 'sonner'
import { useApp } from '@/context/app-state'
import { api, type Polygon } from '@/lib/api'
import { OFFSETS, eventIso } from '@/lib/time'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Calendar } from '@/components/ui/calendar'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'

/** Approximate geodesic area (km²) of a lon/lat polygon ring, for the size check only. */
function areaKm2(p: Polygon): number {
  const R = 6371.0088
  const ring = p.coordinates[0]
  let s = 0
  for (let i = 0; i < ring.length - 1; i++) {
    const [x1, y1] = ring[i]
    const [x2, y2] = ring[i + 1]
    s += ((x2 - x1) * Math.PI) / 180 * (2 + Math.sin((y1 * Math.PI) / 180) + Math.sin((y2 * Math.PI) / 180))
  }
  return Math.abs((s * R * R) / 2)
}

function bboxPolygon(text: string): Polygon | null {
  const v = text.split(/[\s,]+/).map(Number)
  if (v.length !== 4 || v.some((n) => !Number.isFinite(n))) return null
  const [x0, y0, x1, y1] = v
  if (!(x0 < x1 && y0 < y1)) return null
  return { type: 'Polygon', coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]] }
}

export function AnalyzePanel() {
  const app = useApp()
  const cfg = useQuery({ queryKey: ['config'], queryFn: api.config })
  const [day, setDay] = useState<Date | undefined>(undefined)
  const [time, setTime] = useState('')
  const [offset, setOffset] = useState('+00:00')
  const [bbox, setBbox] = useState('')
  const [useS2, setUseS2] = useState(true)
  const [useMl, setUseMl] = useState(true)

  const event = day ? eventIso(day, time, offset) : null
  const area = useMemo(() => (app.aoi ? areaKm2(app.aoi) : 0), [app.aoi])
  const cap = cfg.data?.max_aoi_km2 ?? 0
  const tooBig = !!app.aoi && cap > 0 && area > cap

  const preview = useMutation({ mutationFn: () => api.scenes(app.aoi!, event!) })
  const run = useMutation({
    mutationFn: () => api.createJob({ aoi: app.aoi!, event: event!, use_s2: useS2, use_ml: useMl && !!cfg.data?.ml_model_available }),
    onSuccess: ({ id }) => {
      app.setJobId(id)
      app.setSection('results')
      toast.success(`Job ${id} queued`)
    },
    onError: (e) => toast.error((e as Error).message),
  })

  return (
    <div className='space-y-5 p-4'>
      <section className='space-y-2.5'>
        <h3 className='text-sm font-semibold'>1 · Area of interest</h3>
        <div className='flex flex-wrap gap-2'>
          <Button size='sm' variant={app.drawMode === 'box' ? 'default' : 'outline'} onClick={() => app.setDrawMode(app.drawMode === 'box' ? 'none' : 'box')}>
            <Square className='size-4' /> Box
          </Button>
          <Button size='sm' variant={app.drawMode === 'polygon' ? 'default' : 'outline'} onClick={() => app.setDrawMode(app.drawMode === 'polygon' ? 'none' : 'polygon')}>
            <Pentagon className='size-4' /> Polygon
          </Button>
          <Button size='sm' variant='ghost' onClick={() => app.setAoi(null)} disabled={!app.aoi}>
            <Eraser className='size-4' /> Clear
          </Button>
        </div>
        <div className='flex gap-2'>
          <Input placeholder='minlon, minlat, maxlon, maxlat' value={bbox} onChange={(e) => setBbox(e.target.value)} className='h-8 font-mono text-xs' />
          <Button
            size='sm'
            variant='secondary'
            onClick={() => {
              const p = bboxPolygon(bbox)
              if (p) app.setAoi(p)
              else toast.error('Enter four numbers: minlon, minlat, maxlon, maxlat (EPSG:4326)')
            }}
          >
            <MousePointer2 className='size-4' /> Set
          </Button>
        </div>
        {app.aoi ? (
          <div className='flex items-center gap-2 text-xs text-muted-foreground'>
            <Badge variant={tooBig ? 'destructive' : 'secondary'}>≈ {area.toFixed(0)} km²</Badge>
            <span>limit {cap.toFixed(0)} km²</span>
          </div>
        ) : (
          <p className='text-xs text-muted-foreground'>Draw a box or polygon on the map, or type a bounding box.</p>
        )}
      </section>

      <Separator />

      <section className='space-y-2.5'>
        <h3 className='text-sm font-semibold'>2 · Flood event</h3>
        <div className='grid grid-cols-2 gap-2'>
          <div className='col-span-2 space-y-1'>
            <Label className='text-xs'>Date</Label>
            <Popover>
              <PopoverTrigger asChild>
                <Button variant='outline' size='sm' className='w-full justify-start font-normal'>
                  <CalendarIcon className='size-4' />
                  {day ? eventIso(day, '', '+00:00') : 'Pick the event date'}
                </Button>
              </PopoverTrigger>
              <PopoverContent className='w-auto p-0' align='start'>
                <Calendar mode='single' selected={day} onSelect={setDay} captionLayout='dropdown' defaultMonth={day ?? new Date(2026, 7, 1)} />
              </PopoverContent>
            </Popover>
          </div>
          <div className='space-y-1'>
            <Label className='text-xs'>Time (optional)</Label>
            <Input type='time' value={time} onChange={(e) => setTime(e.target.value)} className='h-8' />
          </div>
          <div className='space-y-1'>
            <Label className='text-xs'>Time zone</Label>
            <Select value={offset} onValueChange={setOffset}>
              <SelectTrigger size='sm' className='w-full'>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {OFFSETS.map((o) => (
                  <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <p className='text-xs text-muted-foreground'>
          Sent as ISO 8601: <code className='rounded bg-muted px-1'>{event ?? '—'}</code>. Without a time, passes on the event day are skipped
          because we can't tell if they were before or after.
        </p>
      </section>

      <Separator />

      <section className='space-y-2.5'>
        <h3 className='text-sm font-semibold'>3 · Evidence</h3>
        <div className='flex items-center justify-between text-sm'>
          <Label htmlFor='s2'>Sentinel-2 optical (when clear)</Label>
          <Switch id='s2' checked={useS2} onCheckedChange={setUseS2} />
        </div>
        <div className='flex items-center justify-between text-sm'>
          <Label htmlFor='ml' className={!cfg.data?.ml_model_available ? 'text-muted-foreground' : ''}>
            ML segmentation (Kuro Siwo)
          </Label>
          <Switch id='ml' checked={useMl && !!cfg.data?.ml_model_available} onCheckedChange={setUseMl} disabled={!cfg.data?.ml_model_available} />
        </div>
        {!cfg.data?.ml_model_available && <p className='text-xs text-muted-foreground'>No trained model is installed yet; runs use the SAR change, debris and optical members.</p>}
      </section>

      <div className='flex gap-2'>
        <Button variant='outline' className='flex-1' disabled={!app.aoi || !event || tooBig || preview.isPending} onClick={() => preview.mutate()}>
          <Satellite className='size-4' /> {preview.isPending ? 'Searching…' : 'Preview scenes'}
        </Button>
        <Button className='flex-1' disabled={!app.aoi || !event || tooBig || run.isPending} onClick={() => run.mutate()}>
          <Play className='size-4' /> Run analysis
        </Button>
      </div>

      {preview.error && (
        <Alert variant='destructive'>
          <AlertDescription className='text-xs'>{(preview.error as Error).message}</AlertDescription>
        </Alert>
      )}
      {preview.data && (
        <div className='space-y-1.5 rounded-md border p-2.5'>
          <div className='text-xs font-medium'>Sentinel-1 same-track pairs</div>
          {preview.data.pairs.map((p) => (
            <div key={p.relative_orbit} className='flex items-center justify-between text-xs'>
              <span className='font-mono'>
                R{String(p.relative_orbit).padStart(3, '0')} · {p.orbit_state}
              </span>
              <span className='text-muted-foreground'>
                {p.pre.start.slice(0, 10)} → {p.post.start.slice(0, 10)} · {Math.round(Math.min(p.pre.aoi_coverage, p.post.aoi_coverage) * 100)}%
              </span>
            </div>
          ))}
          {preview.data.notes.map((n) => (
            <p key={n} className='text-[11px] text-muted-foreground'>{n}</p>
          ))}
        </div>
      )}
    </div>
  )
}
