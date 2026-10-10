import { useQuery } from '@tanstack/react-query'
import { useApp } from '@/context/app-state'
import { api, type Polygon } from '@/lib/api'
import { fmtUtc } from '@/lib/time'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'

export function JobsPanel() {
  const app = useApp()
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: 5000 })
  return (
    <div className='divide-y'>
      {(jobs.data ?? []).map((j) => (
        <button
          key={j.id}
          className='w-full px-4 py-2.5 text-left hover:bg-muted/60'
          onClick={() => {
            if (j.kind === 'flowpath') {
              app.setFlowJobId(j.id)
              app.setSection('flowpath')
            } else {
              app.setJobId(j.id)
              if (j.request.aoi) app.setAoi(j.request.aoi as Polygon)
              app.setSection('results')
            }
          }}
        >
          <div className='flex items-center justify-between'>
            <span className='font-mono text-xs'>{j.id}</span>
            <Badge variant={j.status === 'failed' ? 'destructive' : j.status === 'done' ? 'secondary' : 'outline'}>{j.status}</Badge>
          </div>
          <div className='mt-0.5 text-xs text-muted-foreground'>
            {j.kind === 'flowpath' ? 'Flow path' : `Event ${String(j.request.event)}`} · {fmtUtc(j.created_utc)}
          </div>
        </button>
      ))}
      {jobs.data?.length === 0 && <p className='p-4 text-sm text-muted-foreground'>No jobs yet.</p>}
    </div>
  )
}

export function AboutPanel() {
  const cfg = useQuery({ queryKey: ['config'], queryFn: api.config })
  return (
    <div className='space-y-4 p-4 text-sm'>
      <div className='rounded-md border border-amber-600/40 bg-amber-50 p-3 text-amber-900 dark:bg-amber-950/30 dark:text-amber-200'>
        {cfg.data?.disclaimer ?? 'Educational prototype, not an operational tool.'}
      </div>
      <section className='space-y-1.5'>
        <h3 className='font-semibold'>What it does</h3>
        <p className='text-muted-foreground'>
          For any area and date: maps new water and debris from before/after Sentinel-1 radar (always) and Sentinel-2 (when clear), overlays pre-event
          OpenStreetMap buildings, roads and bridges, and works out which settlements lost every route to a hospital, a town or the edge of the region.
        </p>
      </section>
      <section className='space-y-1.5'>
        <h3 className='font-semibold'>Inputs, and only these</h3>
        <ul className='list-disc space-y-0.5 pl-4 text-muted-foreground'>
          <li>Sentinel-1 GRD and Sentinel-2 L2A (Copernicus Data Space Ecosystem)</li>
          <li>Copernicus DEM GLO-30</li>
          <li>OpenStreetMap as it was before the event (ohsome, snapshot ≤ {cfg.data?.osm_snapshot_max ?? '…'})</li>
          <li>Kuro Siwo for training the segmentation model</li>
        </ul>
        <p className='text-xs text-muted-foreground'>Basemaps on this page are for display only and are never used in the analysis.</p>
      </section>
      <section className='space-y-1.5'>
        <h3 className='font-semibold'>Limitations</h3>
        <ul className='list-disc space-y-0.5 pl-4 text-muted-foreground'>
          <li>Sentinel-1 revisits a track only every few days, so results are a snapshot after the event, not a warning.</li>
          <li>At 10 m resolution a building is “hit by mapped hazard”, not graded for damage.</li>
          <li>Steep gorges hide ground in radar shadow and layover. That ground is shown as not observable.</li>
          <li>Debris does not look like water on radar, so the debris class comes from change detection with terrain and connectivity priors.</li>
          <li>Isolation can only be as complete as the pre-event OSM road and trail network.</li>
        </ul>
      </section>
      <Separator />
      <section className='space-y-1 text-xs text-muted-foreground'>
        {(cfg.data?.attributions ?? []).map((a) => (
          <p key={a}>{a}</p>
        ))}
      </section>
    </div>
  )
}
