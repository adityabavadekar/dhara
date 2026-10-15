import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Check, ChevronsUpDown, Loader2, Plus, TriangleAlert } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { api, type Job, type Polygon } from '@/lib/api'
import { jobLabel } from '@/lib/job'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

function ago(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000
  if (s < 90) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`

/** What a run found, or where it is: shown on hover in the analyses list. Every figure is the
 *  run's own display text; nothing is computed here. */
function RunNote({ job }: { job: Job }) {
  const l = jobLabel(job)
  const b = job.brief ?? {}
  const note = typeof job.request.note === 'string' ? job.request.note : null
  let status: string | null = null
  if (job.status === 'queued') status = job.queue_ahead ? `Waiting: ${plural(job.queue_ahead, 'run', 'runs')} ahead. Runs go one at a time.` : 'Starts next.'
  else if (job.status === 'running') status = `Running, ${Math.round(job.progress * 100)}% done.`
  else if (job.status === 'failed') status = `Stopped: ${(job.error ?? 'unknown error').slice(0, 160)}`
  return (
    <div className='space-y-1 text-left'>
      <div>
        <div className='font-semibold'>{l.title}</div>
        <div className='text-white/65'>{l.sub}</div>
      </div>
      {job.status === 'done' && b.verdict_level && (
        <div>
          <span className='font-semibold'>{b.verdict_level}</span>
          {b.buildings_hit != null && ` · ${b.buildings_hit} buildings hit`}
          {b.settlements_cut_off != null && ` · ${b.settlements_cut_off} cut off`}
          {b.bridges_lost != null && ` · ${b.bridges_lost} bridges lost`}
        </div>
      )}
      {job.status === 'done' && (b.s1_post_dates || b.s2_pair) && (
        <div className='text-white/65'>
          {b.s1_post_dates && `Radar after: ${b.s1_post_dates}`}
          {b.s1_post_dates && b.s2_pair && <br />}
          {b.s2_pair && `Optical: ${b.s2_pair}`}
        </div>
      )}
      {status && <div className='text-white/80'>{status}</div>}
      {note && <div className='border-t border-white/15 pt-1 italic'>“{note}”</div>}
    </div>
  )
}

export function EventSwitcher() {
  const app = useApp()
  const [open, setOpen] = useState(false)
  const jobs = useQuery({
    queryKey: ['jobs'],
    queryFn: api.jobs,
    refetchInterval: (q) => (q.state.data?.some((j) => j.status === 'running' || j.status === 'queued') ? 2000 : 15000),
  })
  const analyses = (jobs.data ?? []).filter((j) => j.kind === 'analysis')
  const current = analyses.find((j) => j.id === app.jobId)
  const label = current ? jobLabel(current) : null

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant='outline' size='sm' className='h-8 min-w-0 max-w-[17.5rem] flex-1 justify-between gap-2 px-2.5 text-xs sm:flex-none'>
          <span className='truncate'>
            {app.mode === 'new' ? (
              'New analysis'
            ) : label ? (
              <>
                <span className='font-medium'>{label.title}</span> <span className='text-muted-foreground'>{label.sub}</span>
              </>
            ) : (
              'Choose an analysis'
            )}
          </span>
          <ChevronsUpDown className='size-3.5 shrink-0 opacity-50' />
        </Button>
      </PopoverTrigger>
      <PopoverContent align='start' className='w-80 p-0'>
        <div className='border-b p-1'>
          <button
            type='button'
            className='flex w-full items-center gap-2 rounded px-2 py-1.5 text-xs font-medium hover:bg-accent'
            onClick={() => {
              app.setMode('new')
              setOpen(false)
            }}
          >
            <Plus className='size-3.5' /> New analysis
          </button>
        </div>
        <div className='max-h-80 overflow-y-auto p-1'>
          <div className='px-2 py-1 text-[0.6875rem] font-medium text-muted-foreground'>Analyses</div>
          {analyses.length === 0 && <div className='px-2 py-3 text-xs text-muted-foreground'>None yet.</div>}
          {analyses.map((j) => {
            const l = jobLabel(j)
            const active = j.id === app.jobId && app.mode === 'workspace'
            const note = typeof j.request.note === 'string' ? j.request.note : null
            return (
              <Tooltip key={j.id} delayDuration={0}>
                <TooltipTrigger asChild>
                  <button
                    type='button'
                    onClick={() => {
                      app.openJob(j.id, (j.request.aoi as Polygon) ?? null)
                      setOpen(false)
                    }}
                    className={cn('flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs hover:bg-accent', active && 'bg-accent')}
                  >
                    <span className='flex size-4 items-center justify-center'>
                      {j.status === 'done' && active && <Check className='size-3.5' />}
                      {(j.status === 'running' || j.status === 'queued') && <Loader2 className='size-3.5 animate-spin text-muted-foreground' />}
                      {j.status === 'failed' && <TriangleAlert className='size-3.5 text-destructive' />}
                    </span>
                    <span className='min-w-0 flex-1'>
                      <span className='block truncate font-medium'>{l.title}</span>
                      <span className='block truncate text-[0.6875rem] text-muted-foreground'>{l.sub}</span>
                      {note && <span className='block truncate text-[0.6875rem] text-muted-foreground italic'>“{note}”</span>}
                    </span>
                    <span className='shrink-0 text-[0.6875rem] text-muted-foreground'>
                      {j.status === 'failed' ? 'failed' : j.status === 'queued' ? (j.queue_ahead ? `${j.queue_ahead} ahead` : 'next') : ago(j.created_utc)}
                    </span>
                  </button>
                </TooltipTrigger>
                <TooltipContent
                  side='right'
                  align='start'
                  sideOffset={8}
                  className='max-w-72 animate-none bg-neutral-900/95 px-3 py-2 text-xs leading-snug text-white shadow-lg data-[state=closed]:animate-none'
                  arrowClassName='bg-neutral-900/95 fill-neutral-900/95'
                >
                  <RunNote job={j} />
                </TooltipContent>
              </Tooltip>
            )
          })}
        </div>
      </PopoverContent>
    </Popover>
  )
}
