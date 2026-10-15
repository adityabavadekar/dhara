// Once an analysis is on screen, quietly load what its report, assistant and before/after view need, so opening them
// later is instant instead of waiting on the network. Runs after the map and panels have loaded,
// when the browser is idle, and only once per analysis.
import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { api, fileUrl, type Manifest } from '@/lib/api'

const FIVE_MIN = 5 * 60_000
const done = new Set<string>()

export function usePrefetchWhenIdle(jobId: string | null, ready: boolean, delayMs = 2500) {
  const qc = useQueryClient()
  useEffect(() => {
    if (!jobId || !ready || done.has(jobId)) return
    const run = () => {
      done.add(jobId)
      // Small and most likely to be opened first.
      void qc.prefetchQuery({ queryKey: ['suggested', jobId, 'en'], queryFn: () => api.suggested(jobId, 'en'), staleTime: FIVE_MIN })
      void qc.prefetchQuery({ queryKey: ['reportStatus', jobId], queryFn: () => api.reportStatus(jobId), staleTime: 30_000 })
      // The report data carries its maps (about 2 MB), so it goes last.
      void qc.prefetchQuery({ queryKey: ['sitrepData', jobId], queryFn: () => api.sitrepData(jobId), staleTime: FIVE_MIN })
      void qc.prefetchQuery({ queryKey: ['suggested', jobId, 'ne'], queryFn: () => api.suggested(jobId, 'ne'), staleTime: FIVE_MIN })
      // Before/after images (optical and radar, about 2 MB in all): fetched into the browser cache,
      // so the swipe opens at once. The map later requests the same URLs.
      const m = qc.getQueryData<Manifest>(['manifest', jobId])
      for (const set of [m?.imagery, m?.imagery_radar]) {
        for (const layer of [set?.before, set?.after]) {
          if (layer?.url) void fetch(fileUrl(jobId, layer.url), { priority: 'low' } as RequestInit).catch(() => undefined)
        }
      }
    }
    // Wait for the screen to settle, then for an idle moment (at most 3 s more).
    const hasIdle = typeof window.requestIdleCallback === 'function'
    let idle = 0
    const timer = window.setTimeout(() => {
      if (hasIdle) idle = window.requestIdleCallback(run, { timeout: 3000 })
      else run()
    }, delayMs)
    return () => {
      window.clearTimeout(timer)
      if (idle) window.cancelIdleCallback(idle)
    }
  }, [jobId, ready, delayMs, qc])
}
