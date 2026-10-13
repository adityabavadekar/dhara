import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Activity, ArrowDownRight, ArrowUpRight, Loader2, Mountain } from 'lucide-react'
import { api, type ProfileResponse } from '@/lib/api'
import { cn } from '@/lib/utils'

interface ElevationProfileProps {
  jobId: string
  path: number[][]
  title?: string
  subtitle?: string
  className?: string
  onHoverPoint?: (pt: { lng: number; lat: number } | null) => void
}

export function ElevationProfile({
  jobId,
  path,
  title = 'Longitudinal Elevation Profile',
  subtitle,
  className,
  onHoverPoint,
}: ElevationProfileProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null)

  const { data, isLoading, isError } = useQuery<ProfileResponse>({
    queryKey: ['elevation-profile', jobId, path.length, path[0], path[path.length - 1]],
    queryFn: () => api.profile(jobId, path),
    enabled: !!jobId && (path.length >= 2 || path.length === 0),
    staleTime: 1000 * 60 * 10,
  })

  const points = data?.points ?? []
  const summary = data?.summary

  // SVG Chart Geometry
  const width = 360
  const height = 110
  const padLeft = 38
  const padRight = 14
  const padTop = 16
  const padBottom = 22

  const plotWidth = width - padLeft - padRight
  const plotHeight = height - padTop - padBottom

  const chart = useMemo(() => {
    if (points.length < 2) return null

    const elevs = points.map((p) => p.elevation_m)
    const dists = points.map((p) => p.dist_km)

    const minElev = Math.min(...elevs)
    const maxElev = Math.max(...elevs)
    const elevSpan = Math.max(maxElev - minElev, 10)

    const maxDist = Math.max(...dists, 0.1)

    // Calculate (x, y) coords
    const pts = points.map((p) => {
      const x = padLeft + (p.dist_km / maxDist) * plotWidth
      const y = padTop + plotHeight - ((p.elevation_m - minElev) / elevSpan) * plotHeight
      return { x, y, p }
    })

    const linePath = pts.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x.toFixed(1)} ${pt.y.toFixed(1)}`).join(' ')
    const areaPath = `${linePath} L ${pts[pts.length - 1].x.toFixed(1)} ${(padTop + plotHeight).toFixed(1)} L ${pts[0].x.toFixed(1)} ${(padTop + plotHeight).toFixed(1)} Z`

    return {
      pts,
      linePath,
      areaPath,
      minElev,
      maxElev,
      maxDist,
    }
  }, [points, plotWidth, plotHeight, padLeft, padTop])

  if (isLoading) {
    return (
      <div className={cn('rounded-lg border bg-card p-3 text-xs', className)}>
        <div className='flex items-center gap-2 text-muted-foreground'>
          <Loader2 className='size-3.5 animate-spin' />
          <span>Generating elevation profile from Copernicus DEM…</span>
        </div>
      </div>
    )
  }

  if (isError || !chart || points.length < 2) {
    return null
  }

  const activePoint = hoverIndex !== null && chart.pts[hoverIndex] ? chart.pts[hoverIndex].p : null

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const svgX = (mouseX / rect.width) * width

    // Find nearest point
    let closestIdx = 0
    let closestDist = Infinity
    chart.pts.forEach((pt, idx) => {
      const d = Math.abs(pt.x - svgX)
      if (d < closestDist) {
        closestDist = d
        closestIdx = idx
      }
    })

    setHoverIndex(closestIdx)
    if (onHoverPoint) {
      const p = chart.pts[closestIdx].p
      onHoverPoint({ lng: p.lng, lat: p.lat })
    }
  }

  const handlePointerLeave = () => {
    setHoverIndex(null)
    if (onHoverPoint) onHoverPoint(null)
  }

  return (
    <div className={cn('overflow-hidden rounded-lg border bg-card/80 p-2.5 shadow-xs', className)}>
      {/* Header */}
      <div className='mb-2 flex items-center justify-between gap-2'>
        <div className='min-w-0'>
          <div className='flex items-center gap-1.5 text-xs font-semibold tracking-tight'>
            <Mountain className='size-3.5 text-emerald-600 dark:text-emerald-400' />
            <span>{title}</span>
          </div>
          {subtitle && <div className='text-[0.6875rem] text-muted-foreground'>{subtitle}</div>}
        </div>
        {summary && (
          <span className='rounded bg-muted px-1.5 py-0.5 text-[0.625rem] font-medium tabular-nums text-muted-foreground'>
            {summary.total_distance_km} km
          </span>
        )}
      </div>

      {/* SVG Elevation Cross-Section */}
      <div className='relative select-none'>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className='w-full cursor-crosshair overflow-visible touch-none'
          onPointerMove={handlePointerMove}
          onPointerLeave={handlePointerLeave}
        >
          <defs>
            <linearGradient id='elev-gradient' x1='0' y1='0' x2='0' y2='1'>
              <stop offset='0%' stopColor='var(--color-emerald-500, #10b981)' stopOpacity='0.35' />
              <stop offset='100%' stopColor='var(--color-emerald-500, #10b981)' stopOpacity='0.03' />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          <line
            x1={padLeft}
            y1={padTop}
            x2={width - padRight}
            y2={padTop}
            stroke='currentColor'
            strokeOpacity='0.08'
            strokeDasharray='2 2'
          />
          <line
            x1={padLeft}
            y1={padTop + plotHeight / 2}
            x2={width - padRight}
            y2={padTop + plotHeight / 2}
            stroke='currentColor'
            strokeOpacity='0.08'
            strokeDasharray='2 2'
          />
          <line
            x1={padLeft}
            y1={padTop + plotHeight}
            x2={width - padRight}
            y2={padTop + plotHeight}
            stroke='currentColor'
            strokeOpacity='0.2'
          />

          {/* Elevation Axis Labels */}
          <text
            x={padLeft - 4}
            y={padTop + 3}
            textAnchor='end'
            className='fill-muted-foreground text-[8px] tabular-nums font-mono'
          >
            {Math.round(chart.maxElev)}m
          </text>
          <text
            x={padLeft - 4}
            y={padTop + plotHeight}
            textAnchor='end'
            className='fill-muted-foreground text-[8px] tabular-nums font-mono'
          >
            {Math.round(chart.minElev)}m
          </text>

          {/* Distance Axis Labels */}
          <text
            x={padLeft}
            y={height - 4}
            textAnchor='start'
            className='fill-muted-foreground text-[8px] tabular-nums font-mono'
          >
            0 km
          </text>
          <text
            x={width - padRight}
            y={height - 4}
            textAnchor='end'
            className='fill-muted-foreground text-[8px] tabular-nums font-mono'
          >
            {chart.maxDist.toFixed(1)} km
          </text>

          {/* Terrain Area & Path */}
          <path d={chart.areaPath} fill='url(#elev-gradient)' />
          <path
            d={chart.linePath}
            fill='none'
            stroke='var(--color-emerald-600, #059669)'
            strokeWidth='1.75'
            strokeLinecap='round'
            strokeLinejoin='round'
          />

          {/* Hover Crosshair & Marker */}
          {hoverIndex !== null && chart.pts[hoverIndex] && (
            <g>
              <line
                x1={chart.pts[hoverIndex].x}
                y1={padTop}
                x2={chart.pts[hoverIndex].x}
                y2={padTop + plotHeight}
                stroke='currentColor'
                strokeWidth='1'
                strokeDasharray='2 2'
                className='text-foreground/60'
              />
              <circle
                cx={chart.pts[hoverIndex].x}
                cy={chart.pts[hoverIndex].y}
                r='3.5'
                className='fill-emerald-600 stroke-background stroke-2 dark:fill-emerald-400'
              />
            </g>
          )}
        </svg>

        {/* Floating Tooltip readout on hover */}
        {activePoint && (
          <div className='mt-1 flex items-center justify-between rounded border bg-muted/60 px-2 py-1 text-[0.6875rem] tabular-nums'>
            <div>
              <span className='text-muted-foreground'>Dist: </span>
              <span className='font-medium'>{activePoint.dist_km} km</span>
            </div>
            <div>
              <span className='text-muted-foreground'>Elev: </span>
              <span className='font-medium'>{Math.round(activePoint.elevation_m)} m MSL</span>
            </div>
            <div>
              <span className='text-muted-foreground'>Slope: </span>
              <span className='font-medium'>{activePoint.slope_pct}%</span>
            </div>
          </div>
        )}
      </div>

      {/* Terrain Metrics Summary Grid */}
      {summary && !activePoint && (
        <div className='mt-2 grid grid-cols-4 gap-1 border-t pt-1.5 text-[0.625rem] tabular-nums'>
          <div className='flex flex-col'>
            <span className='text-muted-foreground'>Span</span>
            <span className='font-medium'>
              {summary.min_elevation_m}-{summary.max_elevation_m}m
            </span>
          </div>
          <div className='flex flex-col'>
            <span className='flex items-center gap-0.5 text-muted-foreground'>
              <ArrowUpRight className='size-2.5 text-emerald-600' />
              <span>Gain</span>
            </span>
            <span className='font-medium'>+{Math.round(summary.elevation_gain_m)}m</span>
          </div>
          <div className='flex flex-col'>
            <span className='flex items-center gap-0.5 text-muted-foreground'>
              <ArrowDownRight className='size-2.5 text-amber-600' />
              <span>Loss</span>
            </span>
            <span className='font-medium'>-{Math.round(summary.elevation_loss_m)}m</span>
          </div>
          <div className='flex flex-col'>
            <span className='flex items-center gap-0.5 text-muted-foreground'>
              <Activity className='size-2.5 text-sky-600' />
              <span>Avg Grd</span>
            </span>
            <span className='font-medium'>{summary.avg_slope_pct}%</span>
          </div>
        </div>
      )}
    </div>
  )
}
