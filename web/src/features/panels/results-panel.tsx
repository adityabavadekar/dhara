import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, Download, FileText, Search } from 'lucide-react'
import { LAYERS, useApp } from '@/context/app-state'
import { api, fileUrl, type Fact } from '@/lib/api'
import { fmtUtc } from '@/lib/time'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Switch } from '@/components/ui/switch'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

const STAGES: Record<string, string> = {
  queued: 'Queued',
  starting: 'Starting',
  scene_selection: 'Finding Sentinel-1 pairs',
  fetch_inputs: 'Downloading S1, S2, DEM and OSM for the area',
  terrain: 'Terrain: flow routing and HAND',
  sentinel1: 'Radar change detection',
  sentinel2: 'Optical evidence',
  fusion: 'Fusing evidence',
  damage: 'Damage overlay and bridges',
  network: 'Cut-off analysis',
  outputs: 'Writing map layers',
  sitrep: 'Writing the situation report',
  done: 'Done',
}

const STATUS_STYLE: Record<string, string> = {
  cut_off: 'bg-red-700 text-white',
  hospital_cut_off: 'bg-orange-600 text-white',
  detour: 'bg-amber-500 text-black',
  connected: 'bg-emerald-600 text-white',
  no_baseline_route: 'bg-slate-400 text-black',
  not_on_network: 'bg-slate-300 text-black',
  likely_lost: 'bg-red-700 text-white',
  at_risk: 'bg-orange-500 text-black',
  likely_standing: 'bg-slate-800 text-white',
  not_observable: 'bg-stone-400 text-black',
}
const STATUS_ORDER = ['cut_off', 'hospital_cut_off', 'detour', 'connected', 'no_baseline_route', 'not_on_network']

function StatusBadge({ s }: { s: string }) {
  return <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium whitespace-nowrap ${STATUS_STYLE[s] ?? 'bg-muted'}`}>{s.replaceAll('_', ' ')}</span>
}

function useFacts(id: string | null, enabled: boolean) {
  const q = useQuery({ queryKey: ['facts', id], queryFn: () => api.facts(id!), enabled: !!id && enabled })
  return useMemo(() => Object.fromEntries((q.data ?? []).map((f: Fact) => [f.key, f])), [q.data])
}

type Props = Record<string, unknown>
type SRow = Props & { geom: GeoJSON.Point }
type BRow = Props & { c: GeoJSON.Position }

export function ResultsPanel() {
  const app = useApp()
  const id = app.jobId
  const job = useQuery({
    queryKey: ['job', id],
    queryFn: () => api.job(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && ['done', 'failed'].includes(q.state.data.status) ? false : 1500),
  })
  const done = job.data?.status === 'done'
  const F = useFacts(id, done)
  const settlements = useQuery({ queryKey: ['gj', id, 'settlements'], queryFn: () => api.geojson(id!, 'web/settlements.geojson'), enabled: !!id && done })
  const bridges = useQuery({ queryKey: ['gj', id, 'bridges'], queryFn: () => api.geojson(id!, 'web/bridges.geojson'), enabled: !!id && done })
  const repairs = useQuery({
    queryKey: ['repairs', id],
    queryFn: () => fetch(fileUrl(id!, 'repairs.json')).then((r) => r.json() as Promise<Props[]>),
    enabled: !!id && done,
  })
  const [q, setQ] = useState('')

  if (!id) return <p className='p-4 text-sm text-muted-foreground'>No analysis selected. Start one from “Analyze”, or open a past job from “Jobs”.</p>
  if (!job.data) return <p className='p-4 text-sm text-muted-foreground'>Loading job…</p>
  const j = job.data
  const t = (k: string) => F[k]?.text ?? '—'

  const sRows = ((settlements.data?.features ?? []) as GeoJSON.Feature[])
    .map((f): SRow => ({ ...(f.properties as Props), geom: f.geometry as GeoJSON.Point }))
    .filter((p) => !q || String(p.name ?? '').toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => STATUS_ORDER.indexOf(String(a.status)) - STATUS_ORDER.indexOf(String(b.status)) || Number(b.buildings_500m) - Number(a.buildings_500m))

  return (
    <div className='flex h-full flex-col'>
      <div className='space-y-2 border-b p-4'>
        <div className='flex items-center justify-between'>
          <div className='font-mono text-xs text-muted-foreground'>job {j.id}</div>
          <Badge variant={j.status === 'failed' ? 'destructive' : j.status === 'done' ? 'secondary' : 'outline'}>{j.status}</Badge>
        </div>
        {(j.status === 'running' || j.status === 'queued') && (
          <>
            <Progress value={j.progress * 100} />
            <div className='text-xs text-muted-foreground'>{STAGES[j.stage] ?? j.stage}…</div>
          </>
        )}
        {j.status === 'failed' && (
          <Alert variant='destructive'>
            <AlertTriangle className='size-4' />
            <AlertTitle>The analysis stopped</AlertTitle>
            <AlertDescription className='text-xs break-words'>{j.error}</AlertDescription>
          </Alert>
        )}
        {done && (
          <div className='grid grid-cols-2 gap-2'>
            <Kpi label='settlements cut off' value={t('settlements_cut_off')} tone='text-red-700 dark:text-red-400' />
            <Kpi label={`bridges likely lost (of ${t('bridges_total')})`} value={t('bridges_lost')} />
            <Kpi label={`buildings hit (of ${t('buildings_total')})`} value={t('buildings_hit')} tone='text-amber-700 dark:text-amber-400' />
            <Kpi label='road / trail network cut' value={`${t('network_failed_km')} km`} />
            <Kpi label='debris / scour' value={`${t('area_debris_km2')} km²`} />
            <Kpi label='not observable' value={`${t('area_not_observable_km2')} km²`} />
          </div>
        )}
      </div>

      {done && (
        <Tabs defaultValue='layers' className='flex min-h-0 flex-1 flex-col'>
          <TabsList className='mx-4 mt-3 grid grid-cols-5'>
            <TabsTrigger value='layers'>Layers</TabsTrigger>
            <TabsTrigger value='settlements'>People</TabsTrigger>
            <TabsTrigger value='bridges'>Bridges</TabsTrigger>
            <TabsTrigger value='repairs'>Repairs</TabsTrigger>
            <TabsTrigger value='report'>Report</TabsTrigger>
          </TabsList>

          <TabsContent value='layers' className='overflow-auto px-4 pb-4'>
            {(['Hazard', 'Damage', 'Isolation', 'Bonus'] as const).map((g) => (
              <div key={g} className='mt-3'>
                <div className='mb-1.5 text-[11px] font-medium tracking-wide text-muted-foreground uppercase'>{g}</div>
                {LAYERS.filter((l) => l.group === g).map((l) => (
                  <label key={l.id} className='flex items-center justify-between py-1 text-sm'>
                    <span className='flex items-center gap-2'>
                      <span className='inline-block size-3 rounded-sm' style={{ background: l.color }} />
                      {l.label}
                    </span>
                    <Switch checked={app.visible[l.id]} onCheckedChange={(v) => app.toggle(l.id, v)} />
                  </label>
                ))}
              </div>
            ))}
            <p className='mt-3 text-xs text-muted-foreground'>
              Grey = not observable (radar shadow/layover, cloud). That means unknown, not undamaged.
            </p>
          </TabsContent>

          <TabsContent value='settlements' className='flex min-h-0 flex-col px-4 pb-4'>
            <div className='relative mt-3'>
              <Search className='absolute top-2 left-2 size-4 text-muted-foreground' />
              <Input className='h-8 pl-8' placeholder='Filter settlements' value={q} onChange={(e) => setQ(e.target.value)} />
            </div>
            <div className='mt-2 min-h-0 flex-1 overflow-auto'>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Settlement</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className='text-right'>Bldgs</TableHead>
                    <TableHead className='text-right'>P(cut)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sRows.slice(0, 400).map((p, i) => (
                    <TableRow key={i} className='cursor-pointer' onClick={() => app.flyTo(p.geom.coordinates[0], p.geom.coordinates[1], 14)}>
                      <TableCell className='max-w-36 truncate'>{String(p.name || '(unnamed)')}</TableCell>
                      <TableCell><StatusBadge s={String(p.status)} /></TableCell>
                      <TableCell className='text-right tabular-nums'>{String(p.buildings_500m)}</TableCell>
                      <TableCell className='text-right tabular-nums'>{p.p_cut_off == null ? '—' : `${Math.round(Number(p.p_cut_off) * 100)}%`}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            <p className='pt-2 text-[11px] text-muted-foreground'>Exposure = OSM buildings within {t('exposure_radius_m')} m (no population data used).</p>
          </TabsContent>

          <TabsContent value='bridges' className='overflow-auto px-4 pb-4'>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Bridge</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className='text-right'>P(lost)</TableHead>
                  <TableHead className='text-right'>Δ dB</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {((bridges.data?.features ?? []) as GeoJSON.Feature[])
                  .map((f): BRow => ({ ...(f.properties as Props), c: (f.geometry as GeoJSON.LineString).coordinates[0] }))
                  .sort((a, b) => Number(b.p_lost ?? 0) - Number(a.p_lost ?? 0))
                  .map((p, i) => (
                    <TableRow key={i} className='cursor-pointer' onClick={() => app.flyTo(p.c[0], p.c[1], 16)}>
                      <TableCell className='max-w-32 truncate'>{String(p.name || p.highway)}</TableCell>
                      <TableCell><StatusBadge s={String(p.status)} /></TableCell>
                      <TableCell className='text-right tabular-nums'>{p.p_lost == null ? '—' : Number(p.p_lost).toFixed(2)}</TableCell>
                      <TableCell className='text-right tabular-nums'>{p.bright_drop_db == null ? '—' : Number(p.bright_drop_db).toFixed(1)}</TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
            <p className='pt-2 text-[11px] text-muted-foreground'>
              “Likely lost” needs the bridge's bright radar return to disappear. Hazard at both ends without that is “at risk”.
            </p>
          </TabsContent>

          <TabsContent value='repairs' className='overflow-auto px-4 pb-4'>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>#</TableHead>
                  <TableHead>Repair</TableHead>
                  <TableHead className='text-right'>Settl.</TableHead>
                  <TableHead className='text-right'>Bldgs</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(repairs.data ?? []).slice(0, 30).map((r, i) => (
                  <TableRow key={i}>
                    <TableCell>{i + 1}</TableCell>
                    <TableCell className='max-w-40'>
                      <div className='truncate'>{r.is_bridge ? 'Bridge' : 'Road'} {String(r.name || '')}</div>
                      <div className='font-mono text-[10px] text-muted-foreground'>{String(r.highway)} · {String(r.osm_id)}</div>
                    </TableCell>
                    <TableCell className='text-right tabular-nums'>{String(r.settlements_reconnected)}</TableCell>
                    <TableCell className='text-right tabular-nums'>{String(r.buildings_reconnected)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {repairs.data?.length === 0 && <p className='pt-3 text-sm text-muted-foreground'>No single repair reconnects a cut-off settlement.</p>}
          </TabsContent>

          <TabsContent value='report' className='space-y-3 overflow-auto px-4 pb-4'>
            <div className='mt-3 grid grid-cols-2 gap-2'>
              <Button asChild variant='outline' size='sm'><a href={fileUrl(j.id, 'sitrep_en.html')} target='_blank' rel='noreferrer'><FileText className='size-4' /> Sitrep (English)</a></Button>
              <Button asChild variant='outline' size='sm'><a href={fileUrl(j.id, 'sitrep_ne.html')} target='_blank' rel='noreferrer'><FileText className='size-4' /> Sitrep (नेपाली)</a></Button>
              <Button asChild size='sm' className='col-span-2'><a href={`/api/jobs/${j.id}/download.zip`}><Download className='size-4' /> GeoJSON + sitrep + provenance (.zip)</a></Button>
              <Button asChild variant='ghost' size='sm' className='col-span-2'><a href={fileUrl(j.id, 'provenance.json')} target='_blank' rel='noreferrer'>Provenance (scene IDs, parameters, timings)</a></Button>
            </div>
            <dl className='grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs'>
              <dt className='text-muted-foreground'>Event</dt><dd>{t('event')}</dd>
              <dt className='text-muted-foreground'>Sentinel-2</dt><dd>{t('s2_pair')}</dd>
              <dt className='text-muted-foreground'>OSM snapshot</dt><dd>{t('osm_snapshot')}</dd>
              <dt className='text-muted-foreground'>Members</dt><dd className='break-words'>{t('members')}</dd>
              <dt className='text-muted-foreground'>Generated</dt><dd>{fmtUtc(F.generated_utc?.text)}</dd>
            </dl>
            <p className='text-[11px] text-muted-foreground'>Every number in the sitrep comes from the job's facts table and is checked before the report is written.</p>
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
}

function Kpi({ label, value, tone = '' }: { label: string; value: string; tone?: string }) {
  return (
    <div className='rounded-md border px-2.5 py-2'>
      <div className={`text-lg leading-tight font-semibold tabular-nums ${tone}`}>{value}</div>
      <div className='text-[11px] text-muted-foreground'>{label}</div>
    </div>
  )
}
