import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Bot, Moon, PanelLeftClose, PanelLeftOpen, Plus, Sun } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { useTheme } from '@/context/theme-provider'
import { api, type Polygon } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { NewAnalysis } from '@/features/analyze/new-analysis'
import { EventSwitcher } from '@/features/analyze/event-switcher'
import { AssistantPanel } from '@/features/assistant/assistant-panel'
import { MapView } from '@/features/map/map-view'
import { AboutDialog } from '@/features/report/about-dialog'
import { ReportDialog } from '@/features/report/report-dialog'
import { Workspace } from '@/features/workspace/workspace'
import { cn } from '@/lib/utils'

function Logo() {
  return (
    <svg viewBox='0 0 32 32' className='size-6 shrink-0' aria-hidden>
      <rect width='32' height='32' rx='7' className='fill-primary' />
      <path d='M7 20c4-6 7 2 11-3s4-6 7-5' stroke='#c27803' strokeWidth='3' fill='none' strokeLinecap='round' />
    </svg>
  )
}

export function App() {
  const app = useApp()
  const { resolvedTheme, setTheme } = useTheme()
  const [panelOpen, setPanelOpen] = useState(true)
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  const opened = useRef(false)

  // Open the latest finished analysis on first load, so the page starts with an answer.
  useEffect(() => {
    if (opened.current || app.jobId || !jobs.data) return
    opened.current = true
    const latest = jobs.data.find((j) => j.kind === 'analysis' && j.status === 'done')
    if (latest) app.openJob(latest.id, (latest.request.aoi as Polygon) ?? null)
    else app.setMode('new')
  }, [jobs.data, app])

  return (
    <div className='flex h-svh w-screen flex-col overflow-hidden bg-background'>
      <header className='z-20 flex h-12 shrink-0 items-center gap-2 border-b bg-background px-3'>
        <Button
          variant='ghost'
          size='icon'
          className='size-8'
          onClick={() => setPanelOpen((o) => !o)}
          title={panelOpen ? 'Hide panel' : 'Show panel'}
        >
          {panelOpen ? <PanelLeftClose className='size-4' /> : <PanelLeftOpen className='size-4' />}
        </Button>
        <div className='flex items-center gap-2'>
          <Logo />
          <span className='hidden text-sm font-semibold tracking-tight sm:inline'>Dhara</span>
        </div>
        <Separator orientation='vertical' className='mx-1 hidden h-5 sm:block' />
        <EventSwitcher />
        <Button
          variant={app.mode === 'new' ? 'secondary' : 'ghost'}
          size='sm'
          className='hidden h-8 gap-1.5 px-2.5 text-xs sm:flex'
          onClick={() => {
            app.setMode('new')
            setPanelOpen(true)
          }}
        >
          <Plus className='size-3.5' /> New
        </Button>

        <div className='ml-auto flex shrink-0 items-center gap-0.5'>
          <span className='mr-2 hidden text-[0.6875rem] text-muted-foreground lg:inline'>Educational prototype · not operational</span>
          <ReportDialog />
          <Button
            variant={app.assistantOpen ? 'secondary' : 'ghost'}
            size='sm'
            className='h-8 gap-1.5 px-2.5 text-xs'
            onClick={() => app.setAssistantOpen(!app.assistantOpen)}
            aria-label='Assistant'
          >
            <Bot className='size-3.5' /> <span className='hidden md:inline'>Assistant</span>
          </Button>
          <AboutDialog />
          <Button
            variant='ghost'
            size='icon'
            className='size-8'
            onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
            title='Toggle theme'
          >
            {resolvedTheme === 'dark' ? <Sun className='size-4' /> : <Moon className='size-4' />}
          </Button>
        </div>
      </header>

      <div className='relative flex min-h-0 flex-1 flex-col md:flex-row'>
        <aside
          className={cn(
            'flex min-h-0 w-full flex-col border-b bg-background md:w-[25rem] md:flex-none md:border-r md:border-b-0',
            panelOpen ? 'max-md:h-[55vh]' : 'hidden'
          )}
        >
          {app.mode === 'new' ? <NewAnalysis /> : <Workspace />}
        </aside>
        <main className='relative min-h-[45vh] min-w-0 flex-1 md:min-h-0'>
          <MapView />
        </main>
        <AssistantPanel />
      </div>
    </div>
  )
}
