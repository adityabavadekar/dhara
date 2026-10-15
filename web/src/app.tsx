import { useEffect, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Bot, FileText, Moon, PanelLeftClose, PanelLeftOpen, Plus, Sun } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { useTheme } from '@/context/theme-provider'
import { api, type Polygon } from '@/lib/api'
import { prefetchJob, useJob } from '@/lib/job'
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
import { Landing } from '@/features/landing/landing'
import { Workspace } from '@/features/workspace/workspace'
import { cn } from '@/lib/utils'
import { Logo } from '@/components/logo'

export function App() {
  const app = useApp()
  const { resolvedTheme, setTheme } = useTheme()
  const [panelOpen, setPanelOpen] = useState(true)
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  const queryClient = useQueryClient()
  // A linked analysis (?job=<id>) starts loading at once, alongside the job list, instead of after it.
  useEffect(() => {
    const wanted = new URLSearchParams(window.location.search).get('job')
    if (wanted) prefetchJob(queryClient, wanted)
  }, [queryClient])
  const opened = useRef(false)
  const current = useJob(app.jobId)
  const reportJobId = current.data?.status === 'done' && current.data.kind === 'analysis' ? app.jobId : null

  const [path, setPath] = useState(() => window.location.pathname)

  useEffect(() => {
    const onPop = () => setPath(window.location.pathname)
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  // ?job=<id> opens that analysis whenever the address changes (links from the landing page,
  // reports and PDFs). Otherwise the dashboard opens the latest finished analysis once, so it
  // starts with an answer. Nothing is opened while the landing page is showing.
  const wantedRef = useRef<string | null>(null)
  useEffect(() => {
    if (!jobs.data) return
    const wanted = new URLSearchParams(window.location.search).get('job')
    if (!wanted && (path === '/' || path === '/landing')) wantedRef.current = null // a later click on the same run opens it again
    if (wanted) {
      if (wanted === wantedRef.current) return
      wantedRef.current = wanted
      opened.current = true
      const j = jobs.data.find((x) => x.id === wanted)
      prefetchJob(queryClient, wanted)
      app.openJob(wanted, (j?.request.aoi as Polygon | undefined) ?? null)
      return
    }
    if (path === '/' || path === '/landing' || opened.current || app.jobId) return
    opened.current = true
    const latest = jobs.data.find((j) => j.kind === 'analysis' && j.status === 'done')
    if (latest) {
      prefetchJob(queryClient, latest.id)
      app.openJob(latest.id, (latest.request.aoi as Polygon) ?? null)
    } else app.setMode('new')
  }, [jobs.data, app, queryClient, path])

  // Old links to the report's former address.
  useEffect(() => {
    if (path.startsWith('/labs/reports')) {
      window.history.replaceState({}, '', `/report${window.location.search}`)
      setPath('/report')
    }
  }, [path])

  // "/" (and /landing) is the landing page; the dashboard is /app. A link with ?job= (from reports,
  // PDFs and shared links) always opens the dashboard at that analysis.
  if ((path === '/' || path === '/landing') && !new URLSearchParams(window.location.search).has('job')) {
    return <Landing />
  }

  if (path === '/report' || path.startsWith('/labs/reports')) {
    return (
      <ReportPage
        onBack={() => {
          // Back to the same analysis in the dashboard.
          const job = new URLSearchParams(window.location.search).get('job')
          navigate(job ? `/app?job=${encodeURIComponent(job)}` : '/app')
        }}
      />
    )
  }

  return (
    <div className='flex h-svh w-screen flex-col overflow-hidden bg-background'>
      <header className='z-20 flex h-12 shrink-0 items-center gap-1.5 border-b bg-background px-2 sm:gap-2 sm:px-3'>
        <Button variant='ghost' size='icon' className='size-8' onClick={() => setPanelOpen((o) => !o)} title={panelOpen ? 'Hide panel' : 'Show panel'}>
          {panelOpen ? <PanelLeftClose className='size-4' /> : <PanelLeftOpen className='size-4' />}
        </Button>
        <a
          href='/'
          onClick={(e) => {
            e.preventDefault()
            navigate('/')
          }}
          className='hidden items-center gap-2 rounded-md focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none sm:flex'
          title='About Dhara'
        >
          <Logo />
          <span className='hidden text-sm font-semibold tracking-tight sm:inline'>Dhara</span>
        </a>
        <Separator orientation='vertical' className='mx-1 hidden h-5 sm:block' />
        <EventSwitcher />
        <Button
          variant={app.mode === 'new' ? 'secondary' : 'ghost'}
          size='sm'
          className='h-8 shrink-0 gap-1.5 px-2 text-xs sm:px-2.5'
          onClick={() => {
            app.setMode('new')
            setPanelOpen(true)
          }}
          aria-label='New analysis'
          title='New analysis'
        >
          <Plus className='size-3.5' /> <span className='hidden sm:inline'>New</span>
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
          <Button variant='ghost' size='icon' className='size-8' onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')} title='Toggle theme'>
            {resolvedTheme === 'dark' ? <Sun className='size-4' /> : <Moon className='size-4' />}
          </Button>
        </div>
      </header>

      <div className='relative flex min-h-0 flex-1 flex-col md:flex-row'>
        <aside className={cn('flex min-h-0 w-full flex-col border-b bg-background md:w-[25rem] md:flex-none md:border-r md:border-b-0', panelOpen ? 'max-md:h-[55vh]' : 'hidden')}>
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
