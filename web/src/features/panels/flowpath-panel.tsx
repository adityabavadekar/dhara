import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Crosshair, Route } from 'lucide-react'
import { toast } from 'sonner'
import { useApp } from '@/context/app-state'
import { api, fileUrl } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

type Meta = { length_km: number; drop_m: number; settlements: number; stopped_because: string; osm_snapshot: string }

export function FlowPathPanel() {
  const app = useApp()
  const [len, setLen] = useState(40)
  const [corr, setCorr] = useState(600)
  const id = app.flowJobId
  const job = useQuery({
    queryKey: ['job', id],
    queryFn: () => api.job(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && ['done', 'failed'].includes(q.state.data.status) ? false : 1500),
  })
  const done = job.data?.status === 'done'
  const sett = useQuery({ queryKey: ['fp', id], queryFn: () => api.geojson(id!, 'settlements.geojson'), enabled: !!id && done })
  const run = useMutation({
    mutationFn: () => api.createFlowPath({ lon: app.flowPoint![0], lat: app.flowPoint![1], length_km: len, corridor_m: corr }),
    onSuccess: ({ id }) => app.setFlowJobId(id),
    onError: (e) => toast.error((e as Error).message),
  })
  const meta = job.data?.summary as Meta | undefined

  return (
    <div className='space-y-4 p-4'>
      <p className='text-sm text-muted-foreground'>
        Click a point upstream. Dhara snaps it to the nearest channel, follows the steepest descent on the Copernicus DEM and lists the settlements along
        the path in downstream order. No arrival times are given: travel speed depends on the flow, which satellites don't measure.
      </p>
      <div className='flex gap-2'>
        <Button size='sm' variant={app.drawMode === 'flowpoint' ? 'default' : 'outline'} onClick={() => app.setDrawMode(app.drawMode === 'flowpoint' ? 'none' : 'flowpoint')}>
          <Crosshair className='size-4' /> Pick point
        </Button>
        <code className='self-center text-xs text-muted-foreground'>{app.flowPoint ? `${app.flowPoint[0]}, ${app.flowPoint[1]}` : 'no point'}</code>
      </div>
      <div className='grid grid-cols-2 gap-2'>
        <div className='space-y-1'>
          <Label className='text-xs'>Max length (km)</Label>
          <Input type='number' min={2} max={120} value={len} onChange={(e) => setLen(Number(e.target.value))} className='h-8' />
        </div>
        <div className='space-y-1'>
          <Label className='text-xs'>Corridor half-width (m)</Label>
          <Input type='number' min={50} max={3000} value={corr} onChange={(e) => setCorr(Number(e.target.value))} className='h-8' />
        </div>
      </div>
      <Button className='w-full' disabled={!app.flowPoint || run.isPending || job.data?.status === 'running'} onClick={() => run.mutate()}>
        <Route className='size-4' /> Trace flow path
      </Button>
      {id && job.data && !done && job.data.status !== 'failed' && <Progress value={job.data.progress * 100} />}
      {job.data?.status === 'failed' && <p className='text-sm text-destructive'>{job.data.error}</p>}
      {done && meta && (
        <div className='space-y-2'>
          <div className='flex flex-wrap gap-1.5'>
            <Badge variant='secondary'>{meta.length_km} km</Badge>
            <Badge variant='secondary'>drop {meta.drop_m} m</Badge>
            <Badge variant='secondary'>{meta.settlements} settlements</Badge>
            <Badge variant='outline'>stopped: {meta.stopped_because}</Badge>
          </div>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>#</TableHead>
                <TableHead>Settlement</TableHead>
                <TableHead className='text-right'>km</TableHead>
                <TableHead className='text-right'>elev.</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {((sett.data?.features ?? []) as GeoJSON.Feature[]).map((f, i) => {
                const p = f.properties as Record<string, unknown>
                const c = (f.geometry as GeoJSON.Point).coordinates
                return (
                  <TableRow key={i} className='cursor-pointer' onClick={() => app.flyTo(c[0], c[1], 14)}>
                    <TableCell>{String(p.order)}</TableCell>
                    <TableCell className='max-w-40 truncate'>{String(p.name || '(unnamed)')} <span className='text-[10px] text-muted-foreground'>{String(p.place)}</span></TableCell>
                    <TableCell className='text-right tabular-nums'>{String(p.distance_km)}</TableCell>
                    <TableCell className='text-right tabular-nums'>{String(p.elevation_m)} m</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
          <Button asChild variant='outline' size='sm' className='w-full'>
            <a href={fileUrl(id!, 'flowpath.geojson')} download>Download path (GeoJSON)</a>
          </Button>
        </div>
      )}
    </div>
  )
}
