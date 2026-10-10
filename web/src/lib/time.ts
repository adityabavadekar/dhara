// ISO 8601 helpers. The API accepts a date (time unknown) or a datetime with an explicit offset.

export const OFFSETS = [
  { value: '+00:00', label: 'UTC (Z)' },
  { value: '+05:45', label: 'Nepal (+05:45)' },
  { value: '+05:30', label: 'India (+05:30)' },
  { value: '+06:00', label: 'Bhutan / Bangladesh (+06:00)' },
  { value: '+08:00', label: 'China (+08:00)' },
  { value: '+05:00', label: 'Pakistan (+05:00)' },
  { value: '+01:00', label: 'CET (+01:00)' },
  { value: '-05:00', label: 'US Eastern (−05:00)' },
]

const pad = (n: number) => String(n).padStart(2, '0')

/** YYYY-MM-DD from a calendar Date (the date the user picked, no timezone shift). */
export function isoDate(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

/** Event string for the API: date only, or date + HH:MM + offset. */
export function eventIso(day: Date, time: string, offset: string): string {
  const date = isoDate(day)
  if (!time) return date
  const t = time.length === 5 ? `${time}:00` : time
  return `${date}T${t}${offset === '+00:00' ? 'Z' : offset}`
}

/** Display a UTC ISO string compactly and unambiguously. */
export function fmtUtc(iso: string | null | undefined): string {
  if (!iso) return '—'
  return iso.replace('T', ' ').replace(/(\.\d+)?(Z|\+00:00)$/, ' UTC')
}
