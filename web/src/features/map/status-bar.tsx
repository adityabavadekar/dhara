import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'

/** A "nice" scale-bar length (m) and its width in px for the given metres per pixel. */
function scaleBar(mPerPx: number, maxPx = 90): { label: string; px: number } {
  const maxM = mPerPx * maxPx
  const pow = 10 ** Math.floor(Math.log10(maxM))
  const nice = [5, 2, 1].map((f) => f * pow).find((v) => v <= maxM) ?? pow
  return { label: nice >= 1000 ? `${nice / 1000} km` : `${nice} m`, px: nice / mPerPx }
}

/**
 * Thin bar under the map, in the style of Google Earth: credits on the left behind a "Data
 * attribution" link, then the background-map credit, scale, zoom and cursor position (EPSG:4326).
 */
export function StatusBar({
  basemapCredit,
  mPerPx,
  zoom,
  cursor,
}: {
  basemapCredit: string
  mPerPx: number
  zoom: number
  cursor: [number, number] | null
}) {
  const cfg = useQuery({ queryKey: ['config'], queryFn: api.config, staleTime: Infinity })
  const bar = scaleBar(mPerPx)
  return (
    <div className='flex h-7 shrink-0 items-center gap-3 border-t bg-background px-3 text-xs text-foreground/80'>
      <Popover>
        <PopoverTrigger asChild>
          <button type='button' className='shrink-0 font-semibold text-foreground underline-offset-2 hover:underline'>
            Data attribution
          </button>
        </PopoverTrigger>
        <PopoverContent side='top' align='start' className='w-[26.25rem] max-w-[calc(100vw-2rem)] space-y-2.5 p-3 text-xs'>
          <div className='font-semibold'>Data and credits</div>
          <ul className='space-y-1.5 text-muted-foreground'>
            {(cfg.data?.attributions ?? []).map((a) => (
              <li key={a} className='flex gap-2'>
                <span className='mt-1.5 size-1 shrink-0 rounded-full bg-muted-foreground/60' />
                {a}
              </li>
            ))}
          </ul>
          <div className='border-t pt-2 text-muted-foreground'>
            <span className='font-medium text-foreground'>Background map:</span> {basemapCredit}. Shown for orientation only; it is
            never an analysis input.
          </div>
          <div className='border-t pt-2 text-muted-foreground'>{cfg.data?.disclaimer ?? 'Educational prototype, not an operational tool.'}</div>
        </PopoverContent>
      </Popover>
      <span className='hidden min-w-0 flex-1 truncate md:inline' title={basemapCredit}>
        {basemapCredit}
      </span>
      <div className='ml-auto flex shrink-0 items-center gap-3 tabular-nums'>
        <div className='flex items-center gap-1.5'>
          <span className='h-1.5 border-x border-b border-current' style={{ width: bar.px }} />
          <span>{bar.label}</span>
        </div>
        <span className='hidden sm:inline'>Zoom {zoom.toFixed(1)}</span>
        <span className='hidden w-[11rem] text-right sm:inline'>
          {cursor ? `${cursor[1].toFixed(5)}°N ${cursor[0].toFixed(5)}°E` : ''}
        </span>
      </div>
    </div>
  )
}
