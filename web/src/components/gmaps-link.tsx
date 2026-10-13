import { cn } from '@/lib/utils'

/** Small external link that opens a point (EPSG:4326 [lon, lat]) in Google Maps. */
export function GMapsLink({ at, className, label }: { at: [number, number]; className?: string; label?: string }) {
  const [lon, lat] = at
  return (
    <a
      href={`https://www.google.com/maps/search/?api=1&query=${lat.toFixed(6)},${lon.toFixed(6)}`}
      target='_blank'
      rel='noreferrer'
      onClick={(e) => e.stopPropagation()}
      title='Open in Google Maps'
      aria-label='Open in Google Maps'
      className={cn(
        'inline-flex shrink-0 items-center gap-1 rounded-md p-1 text-[0.6875rem] text-muted-foreground transition-colors hover:bg-muted hover:text-foreground',
        className
      )}
    >
      <svg viewBox='0 0 24 24' className='size-3.5' aria-hidden>
        <path d='M12 2a7 7 0 0 0-7 7c0 5.2 7 13 7 13s7-7.8 7-13a7 7 0 0 0-7-7Z' fill='#ea4335' />
        <path d='M5 9a7 7 0 0 1 2.05-4.95L12 9Z' fill='#4285f4' />
        <path d='M7.05 4.05A7 7 0 0 1 12 2v7Z' fill='#1a73e8' />
        <path d='M5.3 11.6C6.7 16.3 12 22 12 22l-4.2-9.2Z' fill='#34a853' />
        <path d='M19 9a7 7 0 0 0-2.05-4.95L12 9Z' fill='#fbbc04' />
        <circle cx='12' cy='9' r='2.6' fill='#fff' />
      </svg>
      {label}
    </a>
  )
}
