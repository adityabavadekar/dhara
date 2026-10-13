import { useState } from 'react'
import { Download, ExternalLink, FileText } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { apiUrl, fileUrl } from '@/lib/api'
import { useJob } from '@/lib/job'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { cn } from '@/lib/utils'

/** The situation report viewer. `trigger` replaces the default header button. */
export function ReportDialog({ trigger }: { trigger?: React.ReactNode } = {}) {
  const app = useApp()
  const job = useJob(app.jobId)
  const [lang, setLang] = useState<'en' | 'ne'>('en')
  const ready = job.data?.status === 'done'
  const id = app.jobId

  return (
    <Dialog>
      <DialogTrigger asChild disabled={!ready}>
        {trigger ?? (
          <Button variant='ghost' size='sm' className='h-8 gap-1.5 px-2.5 text-xs' disabled={!ready} aria-label='Situation report'>
            <FileText className='size-3.5' /> <span className='hidden md:inline'>Report</span>
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className='flex h-[90vh] max-w-4xl flex-col gap-0 p-0 sm:max-w-4xl'>
        <DialogHeader className='flex-row items-center gap-3 space-y-0 border-b py-3 pr-12 pl-4'>
          <div className='min-w-0 flex-1'>
            <DialogTitle className='text-sm'>Situation report</DialogTitle>
            <DialogDescription className='text-xs'>Summary on the first page, maps and full tables after. Every figure comes from the results.</DialogDescription>
          </div>
          <div className='flex rounded-md border p-0.5'>
            {(['en', 'ne'] as const).map((l) => (
              <button
                key={l}
                type='button'
                onClick={() => setLang(l)}
                className={cn('rounded px-2 py-1 text-xs font-medium', lang === l ? 'bg-muted text-foreground' : 'text-muted-foreground')}
              >
                {l === 'en' ? 'English' : 'नेपाली'}
              </button>
            ))}
          </div>
        </DialogHeader>
        {id && ready && <iframe key={lang} title='Situation report' src={fileUrl(id, `sitrep_${lang}.html`)} className='min-h-0 flex-1 bg-white' />}
        <div className='flex items-center gap-1.5 border-t px-4 py-2.5'>
          {id && (
            <>
              <Button asChild variant='outline' size='sm' className='h-8 gap-1.5 text-xs'>
                <a href={fileUrl(id, `sitrep_${lang}.pdf`)} target='_blank' rel='noreferrer'>
                  <Download className='size-3.5' /> PDF
                </a>
              </Button>
              <Button asChild variant='outline' size='sm' className='h-8 gap-1.5 text-xs'>
                <a href={apiUrl(`/api/jobs/${id}/download.zip`)}>
                  <Download className='size-3.5' /> GeoJSON, report and provenance (.zip)
                </a>
              </Button>
              <Button asChild variant='ghost' size='sm' className='ml-auto h-8 gap-1.5 text-xs'>
                <a href={fileUrl(id, `sitrep_${lang}.html`)} target='_blank' rel='noreferrer'>
                  <ExternalLink className='size-3.5' /> Open in new tab
                </a>
              </Button>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
