import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Check, ChevronsUpDown, Loader2, Plus, TriangleAlert } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { api, type Polygon } from '@/lib/api'
import { jobLabel } from '@/lib/job'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { cn } from '@/lib/utils'

function ago(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000
  if (s < 90) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
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
            {app.mode === 'new' ? 'New analysis' : label ? <><span className='font-medium'>{label.title}</span> <span className='text-muted-foreground'>{label.sub}</span></> : 'Choose an analysis'}
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
            return (
              <button
                type='button'
                key={j.id}
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
                </span>
                <span className='shrink-0 text-[0.6875rem] text-muted-foreground'>{j.status === 'failed' ? 'failed' : ago(j.created_utc)}</span>
              </button>
            )
          })}
        </div>
      </PopoverContent>
    </Popover>
  )
}
