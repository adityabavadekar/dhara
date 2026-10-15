import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Crosshair, HeartPulse, MapPin, Search, Waypoints, X } from 'lucide-react'
import { useApp } from '@/context/app-state'
import { api } from '@/lib/api'
import { useJobData } from '@/lib/job'
import { nameKeys, searchRank } from '@/lib/translit'
import { hasHan, usePinyinReady } from '@/lib/pinyin'
import { Input } from '@/components/ui/input'
import { StatusBadge } from '@/features/workspace/common'

/** "lat, lon" (or "lon, lat" when the first number cannot be a latitude) → [lon, lat]. */
function parseCoords(q: string): [number, number] | null {
  const v = q.split(/[\s,;]+/).filter(Boolean).map(Number)
  if (v.length !== 2 || v.some((n) => !Number.isFinite(n))) return null
  let [lat, lon] = v
  if (Math.abs(lat) > 90) [lat, lon] = [lon, lat]
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null
  return [lon, lat]
}

export function MapSearch() {
  const app = useApp()
  const d = useJobData(app.jobId)
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(false)
  const query = q.trim().toLowerCase()
  const coords = parseCoords(q)
  const places = useQuery({ queryKey: ['places', query], queryFn: () => api.places(query), enabled: open && query.length > 1 && !coords })

  // Empty box: the settlements that most need attention (list is already ranked worst first) and
  // hospitals. Typed text: name matches from the first character, Latin spelling included.
  // Search keys (transliterations included) are built once per analysis, not on every keystroke.
  // Chinese names (border villages) get pinyin keys once that library has loaded.
  const needsPinyin = useMemo(() => d.settlements.some((x) => hasHan(x.name)), [d.settlements])
  const pinyinReady = usePinyinReady(needsPinyin)
  const index = useMemo(
    () => ({
      s: d.settlements.map((x) => [x, nameKeys(x.name)] as const),
      h: d.hospitals.map((x) => [x, nameKeys(x.name)] as const),
      b: d.bridges.filter((x) => x.name && !/^(unnamed|unclassified|path|track|footway)$/i.test(x.name)).map((x) => [x, nameKeys(x.name)] as const),
    }),
    [d.settlements, d.hospitals, d.bridges, pinyinReady],
  )
  const hits = useMemo(() => {
    if (!query)
      return {
        s: d.settlements.filter((s) => ['cut_off', 'hospital_cut_off', 'detour'].includes(s.status)).slice(0, 5),
        h: d.hospitals.slice(0, 3),
        b: [],
      }
    return {
      s: searchRank(index.s, query, 8),
      h: searchRank(index.h, query, 3),
      b: searchRank(index.b, query, 3),
    }
  }, [query, index, d.settlements, d.hospitals])

  const close = () => {
    setOpen(false)
    setQ('')
  }
  const any = coords || hits.s.length || hits.h.length || hits.b.length || (places.data?.length ?? 0)

  return (
    <div className='relative'>
      <Search className='absolute top-2.5 left-2.5 size-3.5 text-muted-foreground' />
      <Input
        value={q}
        onChange={(e) => {
          setQ(e.target.value)
          setOpen(true)
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => e.key === 'Escape' && close()}
        placeholder={app.jobId ? 'Find a settlement, hospital, bridge or lat, lon' : 'Find a place or lat, lon'}
        className='h-9 bg-background pr-8 pl-8 text-xs shadow-sm dark:bg-background'
      />
      {q && (
        <button type='button' className='absolute top-2.5 right-2.5 text-muted-foreground hover:text-foreground' onClick={close} aria-label='Clear'>
          <X className='size-3.5' />
        </button>
      )}
      {open && (query.length > 0 || hits.s.length > 0 || hits.h.length > 0) && (
        <div className='absolute z-20 mt-1 max-h-80 w-full overflow-y-auto rounded-lg border bg-popover p-1 shadow-md' onMouseDown={(e) => e.preventDefault()}>
          {!query && (
            <div className='px-2 pt-1 pb-0.5 text-[0.6875rem] font-medium text-muted-foreground'>
              {hits.s.length ? 'Lost road access' : 'Hospitals and clinics'}
            </div>
          )}
          {query && !any && !places.isFetching && <div className='px-2 py-3 text-xs text-muted-foreground'>Nothing found. Try a name, or “lat, lon”.</div>}
          {coords && (
            <Item
              icon={<Crosshair className='size-3.5' />}
              onClick={() => {
                app.select({ kind: 'point', props: {}, lngLat: coords }, 14)
                close()
              }}
            >
              Go to {coords[1].toFixed(5)}, {coords[0].toFixed(5)}
            </Item>
          )}
          {hits.s.map((s) => (
            <Item
              key={s.id}
              icon={<MapPin className='size-3.5' />}
              onClick={() => {
                app.select({ kind: 'settlement', props: s.raw, lngLat: s.at }, 14)
                close()
              }}
              aside={<StatusBadge status={s.status} />}
            >
              {s.name}
            </Item>
          ))}
          {hits.h.map((h) => (
            <Item
              key={h.id}
              icon={<HeartPulse className='size-3.5 text-red-600' />}
              onClick={() => {
                app.select({ kind: 'hospital', props: h.raw, lngLat: h.at }, 15)
                close()
              }}
            >
              {h.name}
            </Item>
          ))}
          {hits.b.map((b) => (
            <Item
              key={b.id}
              icon={<Waypoints className='size-3.5' />}
              onClick={() => {
                app.select({ kind: 'bridge', props: b.raw, lngLat: b.at }, 15)
                close()
              }}
            >
              {b.name}
            </Item>
          ))}
          {(places.data ?? []).slice(0, 5).map((p) => (
            <Item
              key={p.id}
              icon={<Search className='size-3.5' />}
              onClick={() => {
                app.fitTo(p.bbox)
                close()
              }}
              aside={<span className='text-[0.625rem] text-muted-foreground uppercase'>{p.category}</span>}
            >
              {p.name}
            </Item>
          ))}
        </div>
      )}
    </div>
  )
}

function Item({ icon, children, aside, onClick }: { icon: React.ReactNode; children: React.ReactNode; aside?: React.ReactNode; onClick: () => void }) {
  return (
    <button type='button' onClick={onClick} className='flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs hover:bg-accent'>
      <span className='text-muted-foreground'>{icon}</span>
      <span className='min-w-0 flex-1 truncate'>{children}</span>
      {aside}
    </button>
  )
}
