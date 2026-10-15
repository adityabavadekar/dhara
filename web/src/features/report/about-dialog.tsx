import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Info } from 'lucide-react'
import { api, type ModelRun } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { cn } from '@/lib/utils'

const TABS = ['How it works', 'Model', 'Limitations', 'Data and credits'] as const
export type AboutTab = (typeof TABS)[number]
type Tab = (typeof TABS)[number]

const STEPS: [string, string][] = [
  ['Find images', 'Sentinel-1 radar passes from the same orbit track before and after the date, and the clearest Sentinel-2 pair when the sky allows.'],
  ['Map the change', 'Radar change detection for new water, a second detector for debris and scour (debris does not look like water to radar), and a segmentation model trained on Kuro Siwo. Terrain from the Copernicus DEM rules out water on steep slopes and high ground.'],
  ['Combine', 'The methods are pooled into flood and debris probability, a confidence tier, and a mask of ground the radar could not see.'],
  ['Overlay assets', 'Pre-event OpenStreetMap buildings, roads and bridges are intersected with the map. Each bridge is checked separately for loss of its strong radar return.'],
  ['Test access', 'A road and trail network from OpenStreetMap is cut where flood or debris crosses it. Each settlement is checked for a route to a town, health facility or the edge of the area, before and after.'],
  ['Report', 'A situation report in English and Nepali: a one-page summary first, then maps and full tables of damage, cut-off settlements, repairs and traced flood paths. Every figure is read from the results and checked; a provenance file lists scenes, parameters and timings.'],
]

const LIMITS: string[] = [
  'Sentinel-1 revisits a place only every few days on the same track, so the map shows the state on the next pass, not the moment of the flood. It cannot warn of a glacier collapse minutes ahead.',
  'Steep valleys hide ground from the radar (shadow and layover). Those areas are marked not observable and are not known to be safe.',
  'At 10 m a house is one or two pixels. A building "hit" means it overlaps mapped flood or debris; it is not a damage grade.',
  'Roads, trails and bridges missing from the pre-event OpenStreetMap cannot be tested. A thin map makes settlements look cut off or connected when they are not.',
  'Population is not an allowed input, so impact is counted in buildings near each settlement.',
  'Travel times use typical speeds per road type. Flood path arrival times use an assumed slope-to-speed rule that has not been calibrated.',
  'Wet sediment and open water look alike to radar on a valley floor.',
  'This is an educational prototype, not an operational tool.',
]

const f = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(2))

function ModelTable({ runs, deployed }: { runs: ModelRun[]; deployed: string | null }) {
  return (
    <Table className='text-xs'>
      <TableHeader>
        <TableRow>
          <TableHead>Variant</TableHead>
          <TableHead className='text-right'>F1 test events</TableHead>
          <TableHead className='text-right'>F1 mountain</TableHead>
          <TableHead className='text-right'>IoU mountain</TableHead>
          <TableHead className='text-right'>Sen1Floods11 IoU (test / Bolivia)</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {[...runs].sort((a, b) => Number(b.mountain_held_out) - Number(a.mountain_held_out)).map((r) => (
          <TableRow key={r.run_name} className={r.run_name === deployed ? 'bg-emerald-600/5 font-medium shadow-[inset_3px_0_0_var(--color-emerald-600)]' : ''}>
            <TableCell>
              {r.run_name}
              {r.run_name === deployed && (
                <span className='ml-2 inline-flex items-center gap-1 rounded-md border border-emerald-600/30 bg-emerald-600/10 px-1.5 py-0.5 text-[0.6875rem] font-semibold text-emerald-700 dark:text-emerald-300'>
                  <span className='size-1.5 rounded-full bg-emerald-600' /> In use
                </span>
              )}
            </TableCell>
            <TableCell className={cn('text-right tabular-nums', !r.mountain_held_out && 'text-muted-foreground')}>
              {f(r.test.f1_flood)}
              {!r.mountain_held_out && <span className='ml-1 text-[0.625rem]'>not held out</span>}
            </TableCell>
            <TableCell className={cn('text-right tabular-nums', !r.mountain_held_out && 'text-muted-foreground')}>
              {f(r.mountain.f1_flood)}
              {!r.mountain_held_out && <span className='ml-1 text-[0.625rem]'>not held out</span>}
            </TableCell>
            <TableCell className={cn('text-right tabular-nums', !r.mountain_held_out && 'text-muted-foreground')}>{f(r.mountain.iou_flood)}</TableCell>
            <TableCell className='text-right tabular-nums'>
              {r.sen1floods11 ? `${f(r.sen1floods11.test)} / ${f(r.sen1floods11.bolivia)}` : '—'}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

export function AboutDialog() {
  const [tab, setTab] = useState<Tab>('How it works')
  const cfg = useQuery({ queryKey: ['config'], queryFn: api.config })
  const model = useQuery({ queryKey: ['model'], queryFn: api.model, staleTime: Infinity })
  const dep = model.data?.deployed

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant='ghost' size='sm' className='h-8 gap-1.5 px-2.5 text-xs' aria-label='About Dhara'>
          <Info className='size-3.5' /> <span className='hidden md:inline'>About</span>
        </Button>
      </DialogTrigger>
      <DialogContent className='flex max-h-[85vh] max-w-2xl flex-col gap-0 p-0 sm:max-w-2xl'>
        <DialogHeader className='space-y-1 border-b px-5 pt-4 pb-0'>
          <DialogTitle className='text-sm'>About Dhara</DialogTitle>
          <DialogDescription className='text-xs'>{cfg.data?.disclaimer ?? 'Educational prototype, not an operational tool.'}</DialogDescription>
          <nav className='-mb-px flex gap-1 pt-2'>
            {TABS.map((t) => (
              <button
                key={t}
                type='button'
                onClick={() => setTab(t)}
                className={cn(
                  'h-8 border-b-2 px-2 text-xs font-medium',
                  tab === t ? 'border-foreground text-foreground' : 'border-transparent text-muted-foreground hover:text-foreground'
                )}
              >
                {t}
              </button>
            ))}
          </nav>
        </DialogHeader>
        <ScrollArea className='min-h-0 flex-1'>
          <div className='space-y-4 px-5 py-4 text-xs leading-relaxed'>
            {tab === 'How it works' && (
              <ol className='space-y-3'>
                {STEPS.map(([h, t], i) => (
                  <li key={h} className='flex gap-3'>
                    <span className='flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[0.6875rem] font-semibold'>{i + 1}</span>
                    <div>
                      <div className='font-medium'>{h}</div>
                      <div className='text-muted-foreground'>{t}</div>
                    </div>
                  </li>
                ))}
              </ol>
            )}
            {tab === 'Model' && (
              <>
                {dep ? (
                  <div className='space-y-1 text-muted-foreground'>
                    <p>
                      <span className='font-medium text-foreground'>{dep.architecture}</span>, {dep.encoder_weights}, trained on{' '}
                      {dep.training_data}.
                    </p>
                    <p>
                      "Mountain" is {dep.splits.mountain_events?.length ?? '—'} high-relief Kuro Siwo events ({dep.splits.counts?.mountain ?? '—'}{' '}
                      tiles), never used in training, as the stand-in for unseen Himalayan scenes. "Test events" are Kuro Siwo's
                      official test floods. Kuro Siwo scores are for the flood class; Sen1Floods11 scores are for water (it has no
                      before-images and no flood/permanent split), where Bolivia is a flood the model never saw.
                    </p>
                    {model.data && model.data.runs.some((r) => !r.mountain_held_out) && (
                      <p className='rounded-md border border-amber-500/30 bg-amber-500/5 px-2.5 py-2 text-amber-900 dark:text-amber-200'>
                        Runs marked "not held out" are earlier runs whose split let test and mountain tiles into training, so their
                        scores overstate performance on unseen floods. They are kept for the record; only the other rows are
                        comparable.
                      </p>
                    )}
                  </div>
                ) : (
                  <p className='text-muted-foreground'>No trained model is installed; mapping uses the radar change detectors only.</p>
                )}
                {model.data && model.data.runs.length > 0 && (
                  <div className='overflow-hidden rounded-lg border'>
                    <ModelTable runs={model.data.runs} deployed={dep?.run_name ?? null} />
                  </div>
                )}
                {dep?.mountain_held_out && dep.mountain.event_f1_flood_ci95 && (
                  <p className='text-muted-foreground'>
                    Model in use, mountain events: 95% interval for per-event F1 {f(dep.mountain.event_f1_flood_ci95[0])} to{' '}
                    {f(dep.mountain.event_f1_flood_ci95[1])}.
                  </p>
                )}
                <p className='text-muted-foreground'>
                  The comparison with the Copernicus EMS EMSR927 maps for the Trishuli case study is in the project report. Those maps
                  are used only to check results, never as input.
                </p>
              </>
            )}
            {tab === 'Limitations' && (
              <ul className='space-y-2'>
                {LIMITS.map((l) => (
                  <li key={l} className='flex gap-2'>
                    <span className='mt-1.5 size-1 shrink-0 rounded-full bg-foreground/60' />
                    <span>{l}</span>
                  </li>
                ))}
              </ul>
            )}
            {tab === 'Data and credits' && (
              <>
                <p className='text-muted-foreground'>
                  Inputs are Sentinel-1, Sentinel-2, the Copernicus DEM, and OpenStreetMap as it was on{' '}
                  {cfg.data?.osm_snapshot_max.slice(0, 10) ?? '—'}. Background maps are for display only and never enter the analysis.
                </p>
                <div className='space-y-1.5 rounded-lg border bg-muted/30 p-3'>
                  {(cfg.data?.attributions ?? []).map((a) => (
                    <p key={a}>{a}</p>
                  ))}
                </div>
              </>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}
