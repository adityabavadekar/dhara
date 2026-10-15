// Landing page at "/" (also /landing). One full screen in the style of the project thumbnail: the
// mapped flood-and-debris footprint of the 26 Aug 2026 event as the backdrop, a bold title, the
// featured analysis's own figures, and a panel to open analyses. Validation figures are those in
// validation/results.md (scored against Copernicus EMS, which Dhara never reads).
import '@fontsource/inter-tight/800.css'
import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowRight, Building2, ChevronRight, Code2, FileText, MapPin, Mountain, PlayCircle, Radar, Route, Satellite, ShieldCheck, type LucideIcon } from 'lucide-react'
import { api, type Job } from '@/lib/api'
import { useFacts } from '@/lib/job'
import { navigate } from '@/lib/nav'
import { Logo } from '@/components/logo'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

const REPO = 'https://github.com/adityabavadekar/dhara'
const VIDEO = 'https://youtu.be/dUJkiYxDHBI'
const openJob = (id: string) => navigate(`/app?job=${encodeURIComponent(id)}`)
const DISPLAY = { fontFamily: "'Inter Tight', 'IBM Plex Sans', sans-serif" }

const STEPS: [LucideIcon, string, string][] = [
  [
    Satellite,
    'Sees through cloud',
    'Raw Sentinel-1 radar (GRD) from the Copernicus Data Space, processed by our own chain: noise removal, calibration, speckle filter, terrain geocoding and flattening, with layover and shadow masked. Sentinel-2 is added where the sky is clear.',
  ],
  [
    Radar,
    'Detects change',
    'Radar darkening to water, radar change for debris, optical water and vegetation-loss indices, and a U-Net trained on Kuro Siwo flood events, fused pixel by pixel into flood or debris with a confidence level.',
  ],
  [
    Mountain,
    'Applies physics',
    'Water cannot stand high above the river or on steep slopes; debris must connect to the river. Low valley floor up to the mapped flood level is filled in. Ground the radar could not see is marked unknown, never safe.',
  ],
  [
    Building2,
    'Finds the damage',
    'Buildings, roads, bridges, schools and health facilities from OpenStreetMap as it was before the event are checked against the footprint; each bridge also for loss of its radar echo.',
  ],
  [
    Route,
    'Finds who is cut off',
    'Every settlement is routed over the pre-event road network to a town and a hospital, before and after. Repairs are ranked by the buildings they reconnect, and flood paths can be traced from any upstream point.',
  ],
  [
    FileText,
    'Reports',
    'A one-page situation report in English and Nepali, and an assistant that answers only from the results. Every figure is traced to the analysis, and each run records its scenes and settings.',
  ],
]

export function Landing() {
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  // The page behind the landing is dark too, so overscroll and page edges never flash white.
  useEffect(() => {
    const root = document.documentElement
    const prev = root.style.backgroundColor
    root.style.backgroundColor = '#050505'
    return () => {
      root.style.backgroundColor = prev
    }
  }, [])
  const analyses = (jobs.data ?? []).filter((j) => j.kind === 'analysis' && j.status === 'done')
  // The case-study event (26 Aug 2026) if it is on this server: its run with the largest mapped
  // footprint. Otherwise the latest analysis.
  const mapped = (j: Job) => {
    const a = (j.summary?.area_km2 ?? {}) as Record<string, number>
    return (a.debris ?? 0) + (a.flood ?? 0)
  }
  // The case-study day first (its largest footprint), then every other event, newest first.
  const day = (j: Job) => String(j.request.event ?? '').slice(0, 10)
  const caseStudy = analyses.filter((j) => day(j) === '2026-08-26').sort((a, b) => mapped(b) - mapped(a))
  // Newest event first; within an event, its run with the largest footprint.
  const eventRuns = [...analyses].sort((a, b) => day(b).localeCompare(day(a)) || mapped(b) - mapped(a))
  const featured = caseStudy[0] ?? analyses[0] ?? null
  // One run per flood event (runs within a week of each other count as one), so the list shows
  // different floods rather than several reaches of the same one.
  const sameEvent = (a: Job, b: Job) => Math.abs(Date.parse(day(a)) - Date.parse(day(b))) <= 7 * 86_400_000
  const listed: Job[] = []
  for (const j of [...caseStudy, ...eventRuns]) if (!listed.some((x) => sameEvent(x, j))) listed.push(j)
  const open = () => (featured ? openJob(featured.id) : navigate('/app'))

  return (
    <div className='dark bg-[#050505] text-white'>
      <div className='relative flex min-h-svh flex-col overflow-hidden'>
        {/* The mapped footprint of the event, drawn from the analysis itself. */}
        <img
          src='/landing-hero.webp'
          alt=''
          aria-hidden
          width={2560}
          height={1440}
          className='pointer-events-none absolute inset-0 size-full object-cover object-[62%_center] opacity-90 select-none fade-in-bg'
        />
        {/* Darkens the left third so the text never sits on the bright ribbon. */}
        <div aria-hidden className='pointer-events-none absolute inset-y-0 left-0 w-full bg-gradient-to-r from-[#050505] via-[#050505]/80 to-transparent lg:w-[55%]' />

        <header className='relative z-10'>
          <div className='flex h-16 items-center gap-2.5 px-6 lg:px-12'>
            <Logo className='size-7' />
            <span className='text-base font-semibold tracking-tight'>Dhara</span>
            <nav className='ml-auto flex items-center gap-1'>
              <Button asChild variant='ghost' size='sm' className='hidden text-white/80 hover:bg-white/10 hover:text-white sm:inline-flex'>
                <a href={VIDEO} target='_blank' rel='noreferrer'>
                  <PlayCircle className='size-4' /> Demo
                </a>
              </Button>
              <Button asChild variant='ghost' size='sm' className='hidden text-white/80 hover:bg-white/10 hover:text-white sm:inline-flex'>
                <a href={REPO} target='_blank' rel='noreferrer'>
                  <Code2 className='size-4' /> GitHub
                </a>
              </Button>
              <Button size='sm' className='bg-white text-black hover:bg-white/90' onClick={open}>
                Open dashboard <ArrowRight className='size-4' />
              </Button>
            </nav>
          </div>
        </header>

        <main className='relative z-10 flex flex-1 flex-col gap-8 px-6 pt-4 pb-5 lg:flex-row lg:items-stretch lg:justify-between lg:px-12 lg:pt-8'>
          <section className='max-w-xl'>
            <Badge variant='outline' style={{ animationDelay: '0ms' }} className='rise-in mb-4 border-white/20 bg-black/30 font-normal text-white/70'>
              Multimodal AI Hackathon 2026 · Track B
            </Badge>
            <h1
              style={{ ...DISPLAY, animationDelay: '80ms' }}
              className='rise-in text-7xl leading-[0.9] font-extrabold tracking-[-0.03em] sm:text-8xl xl:text-[8rem] [@media(max-height:800px)]:text-7xl'
            >
              Dhara
            </h1>
            <p style={{ animationDelay: '180ms' }} className='rise-in mt-4 text-xl font-medium text-white/90 sm:text-2xl'>
              Flood damage, mapped from space
            </p>
            <p style={{ animationDelay: '260ms' }} className='rise-in mt-2 max-w-lg text-sm leading-relaxed text-white/60 sm:text-base'>
              Where it flooded, what broke and who is cut off: from satellite radar that sees through monsoon cloud, ending in a one-page report in English and Nepali.
            </p>
            {featured && <Facts job={featured} />}
            <div style={{ animationDelay: '420ms' }} className='rise-in mt-6 flex flex-wrap gap-2'>
              <Button size='lg' className='bg-white text-black hover:bg-white/90' disabled={!featured} onClick={open}>
                Explore the 26 August 2026 flood <ArrowRight className='size-4' />
              </Button>
              <Button asChild size='lg' variant='outline' className='border-white/25 bg-black/30 text-white hover:bg-white/10 hover:text-white'>
                <a href={VIDEO} target='_blank' rel='noreferrer'>
                  <PlayCircle className='size-4' /> Watch the demo
                </a>
              </Button>
            </div>
          </section>
          <aside style={{ animationDelay: '520ms' }} className='rise-in w-full rounded-xl border border-white/10 bg-black/75 p-4 backdrop-blur-sm lg:w-[25rem] lg:self-end'>
            <div className='flex items-center gap-1.5 text-xs font-medium text-white/60'>
              <ShieldCheck className='size-3.5' aria-hidden /> Checked against Copernicus EMS maps ·{' '}
              <a className='underline underline-offset-4 hover:text-white' href={`${REPO}/blob/master/validation/results.md`} target='_blank' rel='noreferrer'>
                results
              </a>
            </div>
            <dl className='mt-2 grid grid-cols-3 gap-3'>
              <Measure value='0.84' label='F1, held-out area' />
              <Measure value='127/127' label='cut-off status match' />
              <Measure value='0.85' label='F1, own radar chain' />
            </dl>
            {listed.length > 0 && (
              <>
                <div className='mt-4 text-xs font-medium text-white/60'>Open an analysis</div>
                <ul className='mt-1.5 divide-y divide-white/10 overflow-hidden rounded-lg border border-white/10'>
                  {listed.slice(0, 4).map((j) => (
                    <AnalysisRow key={j.id} job={j} />
                  ))}
                </ul>
              </>
            )}
          </aside>
        </main>

        <a href='#how' className='relative z-10 mx-auto mb-3 flex items-center gap-1 text-xs text-white/50 hover:text-white'>
          How it works <ChevronRight className='size-3.5 rotate-90' aria-hidden />
        </a>
      </div>

      <section id='how' className='border-t border-white/10 px-6 py-16 lg:px-12'>
        <div className='mx-auto max-w-6xl'>
          <div className='text-xs font-medium tracking-wide text-white/50 uppercase'>How it works</div>
          <h2 style={DISPLAY} className='mt-2 text-3xl font-extrabold tracking-tight sm:text-4xl'>
            From raw radar to who is cut off
          </h2>
          <ol className='mt-10 grid gap-x-10 gap-y-9 sm:grid-cols-2 lg:grid-cols-3'>
            {STEPS.map(([Icon, t, d], i) => (
              <li key={t}>
                <div className='flex items-center gap-3'>
                  <span className='flex size-9 items-center justify-center rounded-lg border border-white/15 bg-white/5'>
                    <Icon className='size-4.5 text-amber-400' aria-hidden />
                  </span>
                  <span className='text-xs text-white/40 tabular-nums'>0{i + 1}</span>
                </div>
                <h3 className='mt-3 text-base font-semibold'>{t}</h3>
                <p className='mt-1.5 text-sm leading-relaxed text-white/60'>{d}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <footer className='border-t border-white/10 px-6 py-4 text-[0.6875rem] text-white/45 lg:px-12'>
        Educational prototype, not for operational decisions · Contains modified Copernicus Sentinel data 2026 · © OpenStreetMap contributors
      </footer>
    </div>
  )
}

function Measure({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <dt style={DISPLAY} className='text-2xl font-extrabold tracking-tight tabular-nums'>
        {value}
      </dt>
      <dd className='text-[0.6875rem] leading-snug text-white/55'>{label}</dd>
    </div>
  )
}

/** Plain English spelling: drops diacritics, so "Ārughāt Bāzār" reads "Arughat Bazar". */
const plain = (s: string) => s.normalize('NFD').replace(/\p{M}/gu, '')

/** The event's name as the analyst gave it, else the nearest mapped place, in plain English. */
const placeName = (job: Job, target: string | null) =>
  plain(job.location_name || (typeof job.request.name === 'string' && job.request.name.trim()) || target?.split(' (')[0] || 'Analysis')

/** "2026-10-11" → "11 Oct 2026". */
const fmtDay = (iso: string | null) => {
  const d = (iso ?? '').slice(0, 10)
  return d ? new Date(`${d}T00:00:00Z`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }) : '—'
}

/** The featured analysis's own figures, as in the thumbnail. */
function Facts({ job }: { job: Job }) {
  const f = useFacts(job.id, true)
  if (!f.loaded) return null
  const event = fmtDay(f.t('event'))
  const place = String(job.request.event ?? '').startsWith('2026-08-26') ? 'Trishuli valley, Nepal' : placeName(job, f.t('target_location'))
  return (
    <ul style={{ animationDelay: '340ms' }} className='rise-in mt-6 space-y-1.5 text-base'>
      <li className='flex items-center gap-2 text-white/75'>
        <MapPin className='size-4 text-amber-400' aria-hidden /> {place} · {event}
      </li>
      <li className='font-semibold'>
        {f.t('settlements_cut_off') ?? '—'} communities cut off · {f.t('buildings_hit') ?? '—'} buildings hit
      </li>
      <li className='text-white/75'>
        {f.t('area_debris_km2') ?? '—'} km² of debris and scour · {f.t('bridges_lost') ?? '—'} bridges likely lost
      </li>
    </ul>
  )
}

function AnalysisRow({ job }: { job: Job }) {
  const f = useFacts(job.id, true)
  const level = f.t('verdict_level')
  return (
    <li>
      <button
        type='button'
        onClick={() => openJob(job.id)}
        className='group flex w-full items-center gap-3 px-3 py-2 text-left transition-colors hover:bg-white/5 focus-visible:bg-white/5 focus-visible:outline-none'
      >
        <div className='min-w-0 flex-1'>
          <div className='truncate text-sm font-medium'>{placeName(job, f.t('target_location'))}</div>
          <div className='truncate text-[0.6875rem] text-white/55'>
            {fmtDay(f.t('event'))} · {f.t('buildings_hit') ?? '—'} buildings · {f.t('settlements_cut_off') ?? '—'} cut off
          </div>
        </div>
        {level && (
          <Badge variant={level === 'SEVERE' ? 'destructive' : 'outline'} className='shrink-0 border-white/20 px-1.5 text-[0.625rem] font-normal'>
            {level}
          </Badge>
        )}
        <ChevronRight className='size-4 shrink-0 text-white/40 group-hover:text-white' aria-hidden />
      </button>
    </li>
  )
}
