export function Logo({ className = 'size-6' }: { className?: string }) {
  return (
    <svg viewBox='0 0 32 32' className={`${className} shrink-0`} aria-hidden>
      <rect width='32' height='32' rx='7' className='fill-primary' />
      <path d='M7 20c4-6 7 2 11-3s4-6 7-5' stroke='#c27803' strokeWidth='3' fill='none' strokeLinecap='round' />
    </svg>
  )
}
