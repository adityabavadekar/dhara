import { Activity, Info, ListChecks, Map as MapIcon, Moon, Route, Sun } from 'lucide-react'
import { useApp, type Section } from '@/context/app-state'
import { useTheme } from '@/context/theme-provider'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
} from '@/components/ui/sidebar'
import { MapView } from '@/features/map/map-view'
import { AnalyzePanel } from '@/features/panels/analyze-panel'
import { FlowPathPanel } from '@/features/panels/flowpath-panel'
import { AboutPanel, JobsPanel } from '@/features/panels/jobs-about'
import { ResultsPanel } from '@/features/panels/results-panel'

const NAV: { id: Section; label: string; icon: typeof MapIcon; hint: string }[] = [
  { id: 'analyze', label: 'Analyze', icon: MapIcon, hint: 'Area, date, run' },
  { id: 'results', label: 'Results', icon: Activity, hint: 'Hazard, damage, isolation' },
  { id: 'flowpath', label: 'Flow path', icon: Route, hint: 'Trace downstream' },
  { id: 'jobs', label: 'Jobs', icon: ListChecks, hint: 'Recent runs' },
  { id: 'about', label: 'About & data', icon: Info, hint: 'Inputs, limits, credits' },
]

function Logo() {
  return (
    <svg viewBox='0 0 32 32' className='size-7 shrink-0' aria-hidden>
      <rect width='32' height='32' rx='7' className='fill-primary' />
      <path d='M7 20c4-6 7 2 11-3s4-6 7-5' stroke='#c27803' strokeWidth='3' fill='none' strokeLinecap='round' />
    </svg>
  )
}

export function App() {
  const app = useApp()
  const { resolvedTheme, setTheme } = useTheme()
  const current = NAV.find((n) => n.id === app.section)!

  return (
    <SidebarProvider defaultOpen={false}>
      <Sidebar collapsible='icon'>
        <SidebarHeader>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton size='lg' className='pointer-events-none'>
                <Logo />
                <div className='grid flex-1 text-left leading-tight'>
                  <span className='font-semibold'>Dhara</span>
                  <span className='text-xs text-muted-foreground'>Flood damage & isolation</span>
                </div>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarHeader>
        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel>Workspace</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {NAV.map((n) => (
                  <SidebarMenuItem key={n.id}>
                    <SidebarMenuButton tooltip={n.label} isActive={app.section === n.id} onClick={() => app.setSection(n.id)}>
                      <n.icon />
                      <span>{n.label}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton tooltip='Theme' onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}>
                {resolvedTheme === 'dark' ? <Sun /> : <Moon />}
                <span>{resolvedTheme === 'dark' ? 'Light' : 'Dark'} theme</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarFooter>
        <SidebarRail />
      </Sidebar>

      <SidebarInset className='h-svh overflow-hidden'>
        <header className='flex h-12 shrink-0 items-center gap-2 border-b px-3'>
          <SidebarTrigger />
          <Separator orientation='vertical' className='h-5' />
          <span className='text-sm font-medium'>{current.label}</span>
          <span className='hidden text-xs text-muted-foreground sm:inline'>· {current.hint}</span>
          <div className='ml-auto flex items-center gap-2'>
            <Badge variant='outline' className='font-normal'>Educational prototype · not operational</Badge>
            <Button variant='ghost' size='icon' className='size-8 md:hidden' onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}>
              {resolvedTheme === 'dark' ? <Sun className='size-4' /> : <Moon className='size-4' />}
            </Button>
          </div>
        </header>
        <div className='flex min-h-0 flex-1 flex-col md:flex-row'>
          <div className='relative min-h-[45vh] flex-1 md:min-h-0'>
            <MapView />
          </div>
          <aside className='flex min-h-0 w-full flex-1 flex-col border-t bg-background md:w-[400px] md:flex-none md:border-t-0 md:border-l'>
            {app.section === 'results' ? (
              <ResultsPanel />
            ) : (
              <ScrollArea className='min-h-0 flex-1'>
                {app.section === 'analyze' && <AnalyzePanel />}
                {app.section === 'flowpath' && <FlowPathPanel />}
                {app.section === 'jobs' && <JobsPanel />}
                {app.section === 'about' && <AboutPanel />}
              </ScrollArea>
            )}
          </aside>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
