// Download button for the one-page report PDF. The server builds the PDF in the background as soon
// as an analysis finishes; until it is ready the button says so (and starts a build for older jobs).
import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, Download, Loader2 } from 'lucide-react'
import { api, apiUrl } from '@/lib/api'
import { Button } from '@/components/ui/button'

export function PdfButton({ jobId, lang, label = 'PDF' }: { jobId: string | null; lang: 'en' | 'ne'; label?: string }) {
  const status = useQuery({
    queryKey: ['reportStatus', jobId],
    queryFn: () => api.reportStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data && Object.values(q.state.data).every((v) => v === 'ready' || v.startsWith('failed')) ? false : 2000),
  })
  const state = status.data?.[lang]
  // An older analysis has no PDF yet: ask the server to build it (returns at once).
  useEffect(() => {
    if (jobId && state === 'missing') api.reportBuild(jobId).catch(() => undefined)
  }, [jobId, state])

  if (state === 'ready') {
    return (
      <Button asChild variant='outline' size='sm' className='h-8 gap-1.5 text-xs'>
        <a href={apiUrl(`/api/jobs/${jobId}/report.pdf?lang=${lang}`)} download>
          <Download className='size-3.5' aria-hidden /> {label}
        </a>
      </Button>
    )
  }
  if (state?.startsWith('failed')) {
    return (
      <Button variant='outline' size='sm' className='h-8 gap-1.5 text-xs text-destructive' disabled title={state}>
        <AlertTriangle className='size-3.5' aria-hidden /> PDF unavailable
      </Button>
    )
  }
  return (
    <Button variant='outline' size='sm' className='h-8 gap-1.5 text-xs' disabled aria-live='polite'>
      <Loader2 className='size-3.5 animate-spin' aria-hidden /> Preparing PDF…
    </Button>
  )
}
