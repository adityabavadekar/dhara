import type { ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'
import { STATUS_COLOR, STATUS_LABEL } from '@/lib/job'
import { cn } from '@/lib/utils'
import { GMapsLink } from '@/components/gmaps-link'

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const c = STATUS_COLOR[status] ?? '#94a3b8'
  return (
    <span
      className={cn('inline-flex items-center gap-1.5 rounded-md border px-1.5 py-0.5 text-[0.6875rem] font-medium whitespace-nowrap', className)}
      style={{ borderColor: `${c}40`, color: c, background: `${c}10` }}
    >
      <span className='size-1.5 rounded-full' style={{ background: c }} />
      {STATUS_LABEL[status] ?? status.replaceAll('_', ' ')}
    </span>
  )
}

export function Dot({ color, className }: { color: string; className?: string }) {
  return <span className={cn('inline-block size-2.5 shrink-0 rounded-[0.1875rem]', className)} style={{ background: color }} />
}

export function Section({ title, aside, children, className }: { title: ReactNode; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn('space-y-2', className)}>
      <div className='flex items-baseline justify-between gap-2'>
        <h3 className='text-xs font-semibold tracking-tight text-foreground'>{title}</h3>
        {aside && <div className='text-[0.6875rem] text-muted-foreground'>{aside}</div>}
      </div>
      {children}
    </section>
  )
}

/** A headline figure. `value` null renders an em dash, never a zero. */
export function Figure({
  label,
  value,
  unit,
  of,
  tone,
  hint,
}: {
  label: string
  value: string | null
  unit?: string
  of?: string | null
  tone?: string
  hint?: ReactNode
}) {
  return (
    <div className='min-w-0'>
      <div className='truncate text-[0.6875rem] font-medium text-muted-foreground'>{label}</div>
      <div className='mt-0.5 flex items-baseline gap-1'>
        <span className={cn('text-xl font-semibold tabular-nums tracking-tight', tone)}>{value ?? '—'}</span>
        {unit && value != null && <span className='text-xs text-muted-foreground'>{unit}</span>}
        {of && <span className='text-xs text-muted-foreground'>of {of}</span>}
      </div>
      {hint && <div className='mt-0.5 truncate text-[0.6875rem] text-muted-foreground'>{hint}</div>}
    </div>
  )
}

/** A clickable list row with a trailing chevron, and a Google Maps link when `at` is given. */
export function Row({
  onClick,
  active,
  children,
  className,
  at,
}: {
  onClick?: () => void
  active?: boolean
  children: ReactNode
  className?: string
  at?: [number, number]
}) {
  return (
    <div className={cn('group flex items-center transition-colors hover:bg-muted/60', active && 'bg-muted')}>
      <button
        type='button'
        onClick={onClick}
        className={cn('flex min-w-0 flex-1 items-center gap-2.5 px-3 py-2 text-left text-xs focus-visible:bg-muted/60 focus-visible:outline-none', className)}
      >
        <div className='min-w-0 flex-1'>{children}</div>
        {!at && <ChevronRight className='size-3.5 shrink-0 text-muted-foreground/50 group-hover:text-muted-foreground' />}
      </button>
      {at && <GMapsLink at={at} className='mr-1.5 opacity-60 group-hover:opacity-100' />}
    </div>
  )
}

export function List({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('divide-y overflow-hidden rounded-lg border bg-card', className)}>{children}</div>
}

export function Note({ children, tone = 'muted' }: { children: ReactNode; tone?: 'muted' | 'warn' }) {
  return (
    <div
      className={cn(
        'rounded-lg border px-3 py-2 text-[0.6875rem] leading-relaxed',
        tone === 'warn' ? 'border-amber-500/30 bg-amber-500/5 text-amber-900 dark:text-amber-200' : 'bg-muted/30 text-muted-foreground'
      )}
    >
      {children}
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className='rounded-lg border border-dashed px-3 py-4 text-center text-xs text-muted-foreground'>{children}</div>
}

/** Horizontal stacked bar for parts of a whole (areas, counts). Zero-width parts are dropped. */
export function Stack({ parts }: { parts: { value: number; color: string; label: string }[] }) {
  const total = parts.reduce((s, p) => s + Math.max(0, p.value), 0)
  if (total <= 0) return <div className='h-1.5 rounded-full bg-muted' />
  return (
    <div className='flex h-1.5 gap-px overflow-hidden rounded-full bg-muted'>
      {parts
        .filter((p) => p.value > 0)
        .map((p) => (
          <span key={p.label} title={p.label} style={{ flexGrow: p.value, background: p.color }} />
        ))}
    </div>
  )
}
