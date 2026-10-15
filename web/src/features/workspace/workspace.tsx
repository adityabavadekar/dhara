import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, Bot, Check, FileText, Loader2, Plus, RotateCcw } from 'lucide-react'
import { useApp, type View } from '@/context/app-state'
import { bounds, jobAoiRing, jobLabel, useJobData } from '@/lib/job'
import { usePrefetchWhenIdle } from '@/lib/prefetch'
import { api, type Polygon } from '@/lib/api'
import { ReportDialog } from '@/features/report/report-dialog'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { CutOff } from './cutoff'
import { Damage } from './damage'
import { FlowPath } from './flowpath'
import { Flooding } from './flooding'
import { Overview } from './overview'

const STAGES: [string, string][] = [
  ['scene_selection', 'Find before and after radar passes'],
  ['fetch_inputs', 'Fetch radar, optical, elevation and map data'],
  ['terrain', 'Model terrain and drainage'],
  ['sentinel1', 'Detect radar change'],
  ['sentinel2', 'Check optical evidence'],
  ['fusion', 'Combine evidence'],
  ['damage', 'Overlay buildings, roads and bridges'],
  ['network', 'Test road access'],
  ['outputs', 'Write map layers'],
  ['sitrep', 'Write situation report'],
]

const TABS: { id: View; label: string; count?: (d: ReturnType<typeof useJobData>) => string | null }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'flooding', label: 'Flooding' },
  { id: 'damage', label: 'Damage' },
  { id: 'cutoff', label: 'Cut off', count: (d) => d.facts.t('settlements_cut_off') },
  { id: 'flowpath', label: 'Flood path' },
]

export function Workspace() {
  const app = useApp()
  const d = useJobData(app.jobId)
  usePrefetchWhenIdle(app.jobId, d.facts.loaded && !d.loading)
  const job = d.job.data

  // Frame the analysed area whenever a different job is opened.
  useEffect(() => {
    const ring = jobAoiRing(job)
    if (ring && job?.id) app.fitTo(bounds(ring))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id])

  if (!app.jobId) return <NoJob />
  if (!job)
    return (
      <div className='space-y-3 p-3'>
        <Skeleton className='h-14' />
        <Skeleton className='h-32' />
        <Skeleton className='h-32' />
      </div>
    )

  const label = jobLabel(job)
  const running = job.status === 'queued' || job.status === 'running'

  return (
    <div className='flex min-h-0 flex-1 flex-col'>
      <div className='border-b px-3 pt-1 pb-0 sm:pt-3'>
        {/* On phones the event picker in the top bar already shows the date, area and place. */}
        <div className='hidden items-start justify-between gap-2 sm:flex'>
          <div className='min-w-0'>
            <div className='text-[0.6875rem] font-medium text-muted-foreground'>Flood event</div>
            <div className='truncate text-sm font-semibold tracking-tight'>{label.title}</div>
            <div className='truncate text-[0.6875rem] text-muted-foreground'>{label.sub}</div>
          </div>
        </div>
        {job.status === 'done' && (
          // On phones the top bar already has Report and Assistant, so this row is hidden there.
          <div className='mt-2.5 hidden grid-cols-2 gap-1.5 sm:grid'>
            <ReportDialog
              jobId={job.id}
              trigger={
                <Button variant='outline' size='sm' className='h-7 justify-start gap-1.5 text-xs'>
                  <FileText className='size-3.5' /> Situation report
                </Button>
              }
            />
            <Button
              variant={app.assistantOpen ? 'secondary' : 'outline'}
              size='sm'
              className='h-7 justify-start gap-1.5 text-xs'
              onClick={() => app.setAssistantOpen(!app.assistantOpen)}
            >
              <Bot className='size-3.5' /> Ask about this event
            </Button>
          </div>
        )}
        {job.status === 'done' && (
          <nav className='-mb-px mt-0 flex sm:mt-2.5' aria-label='Views'>
            {TABS.map((t) => {
              const c = t.count?.(d)
              return (
                <button
                  key={t.id}
                  type='button'
                  onClick={() => app.setView(t.id)}
                  className={cn(
                    'flex h-9 flex-1 items-center justify-center gap-1 border-b-2 px-1 text-xs font-medium whitespace-nowrap transition-colors',
                    app.view === t.id ? 'border-foreground text-foreground' : 'border-transparent text-muted-foreground hover:text-foreground',
                  )}
                >
                  {t.label}
                  {c != null && c !== '0' && <span className='rounded bg-red-600/10 px-1 text-[0.625rem] font-semibold tabular-nums text-red-700 dark:text-red-400'>{c}</span>}
                </button>
              )
            })}
          </nav>
        )}
        {job.status !== 'done' && <div className='h-3' />}
      </div>

      {running && <Running stage={job.stage} progress={job.progress} startedUtc={job.started_utc ?? job.created_utc} queueAhead={job.status === 'queued' ? (job.queue_ahead ?? 0) : null} />}
      {job.status === 'failed' && (
        <div className='space-y-3 p-3'>
          <Alert variant='destructive'>
            <AlertTriangle className='size-4' />
            <AlertTitle>The analysis stopped</AlertTitle>
            <AlertDescription className='text-xs break-words'>{job.error}</AlertDescription>
          </Alert>
          <Button
            size='sm'
            variant='outline'
            className='h-8 gap-1.5 text-xs'
            onClick={() => {
              app.setAoi((job.request.aoi as Polygon) ?? null)
              app.setMode('new')
            }}
          >
            <RotateCcw className='size-3.5' /> Edit and run again
          </Button>
        </div>
      )}
      {job.status === 'done' && (
        // Keyed by tab, so each tab opens scrolled to the top.
        <div key={app.view} className='min-h-0 flex-1 overflow-y-auto'>
          {!d.facts.loaded || d.loading ? (
            <div className='space-y-3 p-3'>
              <Skeleton className='h-28' />
              <Skeleton className='h-28' />
              <Skeleton className='h-28' />
            </div>
          ) : (
            <>
              {app.view === 'overview' && <Overview d={d} />}
              {app.view === 'flooding' && <Flooding d={d} />}
              {app.view === 'damage' && <Damage d={d} />}
              {app.view === 'cutoff' && <CutOff d={d} />}
              {app.view === 'flowpath' && <FlowPath />}
            </>
          )}
        </div>
      )}
    </div>
  )
}

const fmtDuration = (s: number) => (s < 60 ? `${Math.max(1, Math.round(s))} s` : `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, '0')} s`)

/** Typical run length on this server: the median duration of its finished analyses (null when
 *  there are none yet). Used as the starting estimate until the run's own pace takes over. */
function useTypicalRunS(): number | null {
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  return useMemo(() => {
    const d = (jobs.data ?? [])
      .filter((j) => j.kind === 'analysis' && j.status === 'done')
      .map((j) => (Date.parse(j.updated_utc) - Date.parse(j.started_utc ?? j.created_utc)) / 1000)
      .filter((v) => Number.isFinite(v) && v > 0)
      .sort((a, b) => a - b)
    return d.length ? d[Math.floor(d.length / 2)] : null
  }, [jobs.data])
}

/**
 * Time left: starts from this server's typical run length, then blends into the run's own
 * measured pace as stages complete (fully measured by 35%). Without any finished run to learn
 * from, it waits for the measured pace instead of guessing.
 */
function useEta(startedUtc: string, progress: number) {
  const typical = useTypicalRunS()
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(t)
  }, [])
  const elapsed = Math.max(0, (now - Date.parse(startedUtc)) / 1000)
  if (progress >= 1) return { elapsed, left: null as number | null }
  const measured = progress > 0.02 ? (elapsed * (1 - progress)) / progress : null
  const prior = typical != null ? Math.max(10, typical - elapsed) : null
  const alpha = Math.min(1, Math.max(0, (progress - 0.05) / 0.3))
  const blended = prior == null ? (progress >= 0.25 ? measured : null) : measured == null ? prior : (1 - alpha) * prior + alpha * measured
  if (blended == null) return { elapsed, left: null }
  return { elapsed, left: Math.max(5, Math.ceil(blended / 5) * 5) }
}

function Running({ stage, progress, startedUtc, queueAhead }: { stage: string; progress: number; startedUtc: string; queueAhead: number | null }) {
  const idx = STAGES.findIndex(([s]) => s === stage)
  const { elapsed, left } = useEta(startedUtc, progress)
  return (
    <div className='space-y-3 p-3'>
      <div className='space-y-1.5'>
        <div className='flex items-center justify-between text-xs'>
          <span className='font-medium'>{queueAhead != null ? 'Waiting to start' : 'Analysing'}</span>
          <span className='tabular-nums text-muted-foreground'>{Math.round(progress * 100)}%</span>
        </div>
        <Progress value={progress * 100} />
        <div className={cn('flex items-center justify-between text-[0.6875rem] text-muted-foreground tabular-nums', queueAhead != null && 'hidden')}>
          <span>Elapsed {fmtDuration(elapsed)}</span>
          <span className={left != null ? 'font-medium text-foreground' : ''}>{left != null ? `About ${fmtDuration(left)} left` : 'Estimating time left…'}</span>
        </div>
      </div>
      {queueAhead != null && (
        <p className='rounded-md bg-muted/50 px-2.5 py-2 text-[0.6875rem] text-muted-foreground'>
          Analyses run one at a time on this server.{' '}
          {queueAhead > 0 ? `${queueAhead} ${queueAhead === 1 ? 'run is' : 'runs are'} ahead of this one.` : 'This one starts next.'}
        </p>
      )}
      <ol className='space-y-1'>
        {STAGES.map(([s, label], i) => {
          const state = idx < 0 ? 'todo' : i < idx ? 'done' : i === idx ? 'now' : 'todo'
          return (
            <li key={s} className={cn('flex items-center gap-2 text-xs', state === 'todo' && 'text-muted-foreground')}>
              <span className='flex size-4 items-center justify-center'>
                {state === 'done' && <Check className='size-3.5 text-emerald-600' />}
                {state === 'now' && <Loader2 className='size-3.5 animate-spin' />}
                {state === 'todo' && <span className='size-1.5 rounded-full bg-muted-foreground/40' />}
              </span>
              <span className={state === 'now' ? 'font-medium' : ''}>{label}</span>
            </li>
          )
        })}
      </ol>
      <p className='text-[0.6875rem] text-muted-foreground'>Runs on a new area usually take a few minutes. You can browse earlier analyses meanwhile.</p>
    </div>
  )
}

function NoJob() {
  const app = useApp()
  return (
    <div className='flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center'>
      <div className='space-y-1'>
        <h2 className='text-sm font-semibold'>No analysis open</h2>
        <p className='max-w-xs text-xs text-muted-foreground'>
          Choose an area and a flood date. Dhara maps flooded and debris-covered ground, damaged assets and cut-off settlements.
        </p>
      </div>
      <Button size='sm' className='h-8 gap-1.5 text-xs' onClick={() => app.setMode('new')}>
        <Plus className='size-3.5' /> New analysis
      </Button>
    </div>
  )
}
