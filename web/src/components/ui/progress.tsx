import * as React from 'react'
import { cn } from '@/lib/utils'

function Progress({ className, value = 0, ...props }: React.ComponentProps<'div'> & { value?: number }) {
  const v = Math.max(0, Math.min(100, value))
  return (
    <div
      role='progressbar'
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(v)}
      data-slot='progress'
      className={cn('relative h-1.5 w-full overflow-hidden rounded-full bg-muted', className)}
      {...props}
    >
      <div className='h-full bg-primary transition-[width] duration-300' style={{ width: `${v}%` }} />
    </div>
  )
}

export { Progress }
