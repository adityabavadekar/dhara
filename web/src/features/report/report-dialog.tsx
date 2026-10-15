// The one-page situation report in a dialog: the same A4 sheet as /report, zoomable (pinch, Ctrl+scroll,
// +/−), with PDF download (rendered by the server), print, and a link to the full page.
import { useRef, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ExternalLink, Maximize, MoveHorizontal, ZoomIn, ZoomOut } from 'lucide-react'
import { api } from '@/lib/api'
import { navigate, reportPath } from '@/lib/nav'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { cn } from '@/lib/utils'
import { PdfButton } from './pdf-button'
import { ReportSheet } from './report-page'
import { ZoomPane, type ZoomHandle } from './zoom-pane'

// A4 in CSS pixels (96 per inch).
const A4_W = (210 / 25.4) * 96
const A4_H = (297 / 25.4) * 96

export function ReportDialog({ jobId, trigger }: { jobId: string | null; trigger: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [lang, setLang] = useState<'en' | 'ne'>('en')
  const [scale, setScale] = useState(1)
  const zoom = useRef<ZoomHandle>(null)
  const report = useQuery({
    queryKey: ['sitrepData', jobId],
    queryFn: () => api.sitrepData(jobId!),
    enabled: !!jobId && open,
    staleTime: 5 * 60_000,
  })

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild disabled={!jobId}>
        {trigger}
      </DialogTrigger>
      <DialogContent className='sr-dialog flex h-[94svh] w-[96vw] max-w-6xl flex-col gap-0 overflow-hidden p-0 sm:max-w-6xl'>
        <DialogHeader className='flex-row flex-wrap items-center gap-2 space-y-0 border-b py-2 pr-12 pl-4'>
          <div className='mr-auto min-w-0'>
            <DialogTitle className='text-sm'>Situation report</DialogTitle>
            <DialogDescription className='text-xs'>One page, every figure from this analysis. Pinch or Ctrl+scroll to zoom.</DialogDescription>
          </div>
          <div className='flex rounded-md border p-0.5' role='group' aria-label='Language'>
            {(['en', 'ne'] as const).map((l) => (
              <button
                key={l}
                type='button'
                aria-pressed={lang === l}
                onClick={() => setLang(l)}
                className={cn('rounded px-2 py-1 text-xs font-medium', lang === l ? 'bg-muted text-foreground' : 'text-muted-foreground')}
              >
                {l === 'en' ? 'English' : 'नेपाली'}
              </button>
            ))}
          </div>
          <div className='flex items-center rounded-md border' role='group' aria-label='Zoom'>
            <Button variant='ghost' size='icon' className='size-8' onClick={() => zoom.current?.zoomBy(0.8)} aria-label='Zoom out'>
              <ZoomOut className='size-4' />
            </Button>
            <span className='w-11 text-center text-xs tabular-nums' aria-live='polite'>
              {Math.round(scale * 100)}%
            </span>
            <Button variant='ghost' size='icon' className='size-8' onClick={() => zoom.current?.zoomBy(1.25)} aria-label='Zoom in'>
              <ZoomIn className='size-4' />
            </Button>
            <Button variant='ghost' size='icon' className='size-8' onClick={() => zoom.current?.fitWidth()} aria-label='Fit width' title='Fit width'>
              <MoveHorizontal className='size-4' />
            </Button>
            <Button variant='ghost' size='icon' className='size-8' onClick={() => zoom.current?.fitPage()} aria-label='Whole page' title='Whole page'>
              <Maximize className='size-4' />
            </Button>
          </div>
          <PdfButton jobId={jobId} lang={lang} />
          <Button
            variant='ghost'
            size='sm'
            className='h-8 gap-1.5 text-xs'
            disabled={!jobId}
            onClick={() => {
              setOpen(false)
              if (jobId) navigate(reportPath(jobId) + (lang === 'ne' ? '&lang=ne' : ''))
            }}
          >
            <ExternalLink className='size-3.5' /> Full page
          </Button>
        </DialogHeader>
        {report.isLoading && (
          <p className='p-6 text-sm text-muted-foreground' role='status'>
            Building the report from the analysis…
          </p>
        )}
        {report.isError && <p className='p-6 text-sm text-destructive'>The report could not be built for this analysis.</p>}
        {report.data && jobId && (
          <ZoomPane ref={zoom} width={A4_W} height={A4_H} onScale={setScale}>
            <ReportSheet data={report.data} jobId={jobId} lang={lang} />
          </ZoomPane>
        )}
      </DialogContent>
    </Dialog>
  )
}
