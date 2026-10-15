import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Bot, FileText, Moon, PanelLeftClose, PanelLeftOpen, Plus, Sun } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { useTheme } from '@/context/theme-provider'
import { api, type Polygon } from '@/lib/api'
import { useJob } from '@/lib/job'
import { navigate } from '@/lib/nav'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { NewAnalysis } from '@/features/analyze/new-analysis'
import { EventSwitcher } from '@/features/analyze/event-switcher'
import { AssistantPanel } from '@/features/assistant/assistant-panel'
import { MapView } from '@/features/map/map-view'
import { AboutDialog } from '@/features/report/about-dialog'
import { ReportDialog } from '@/features/report/report-dialog'
import { ReportPage } from '@/features/report/report-page'
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
  const current = useJob(app.jobId)
  const reportJobId = current.data?.status === 'done' && current.data.kind === 'analysis' ? app.jobId : null

  const [path, setPath] = useState(() => window.location.pathname)

  useEffect(() => {
    const onPop = () => setPath(window.location.pathname)
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])


  // Open the latest finished analysis on first load, so the page starts with an answer.
  useEffect(() => {
    if (opened.current || app.jobId || !jobs.data) return
    opened.current = true
    // ?job=<id> opens that analysis (links from reports and PDFs); otherwise the latest one.
    const wanted = new URLSearchParams(window.location.search).get('job')
    const latest =
      jobs.data.find((j) => j.id === wanted && j.status === 'done') ?? jobs.data.find((j) => j.kind === 'analysis' && j.status === 'done')
    if (latest) app.openJob(latest.id, (latest.request.aoi as Polygon) ?? null)
    else app.setMode('new')
  }, [jobs.data, app])

  // Old links to the report's former address.
  useEffect(() => {
    if (path.startsWith('/labs/reports')) {
      window.history.replaceState({}, '', `/report${window.location.search}`)
      setPath('/report')
    }
  }, [path])

  if (path === '/report' || path.startsWith('/labs/reports')) {
    return <ReportPage onBack={() => navigate('/')} />
  }

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
          <ReportDialog
            jobId={reportJobId}
            trigger={
              <Button variant='ghost' size='sm' className='h-8 gap-1.5 px-2.5 text-xs' disabled={!reportJobId} aria-label='Situation report'>
                <FileText className='size-3.5' /> <span className='hidden md:inline'>Report</span>
              </Button>
            }
          />
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
