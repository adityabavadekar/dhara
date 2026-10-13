// Typed access to one finished job's outputs. The exported GeoJSON stores every property as a
// string ("48.41", "None", "[[85.2, 28.3], …]"), so everything is parsed here, once.
// All coordinates are EPSG:4326 [lon, lat].
import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, fileUrl, type Fact, type Job } from '@/lib/api'

export type LngLat = [number, number]
type Raw = Record<string, unknown>

export const num = (v: unknown): number | null => {
  if (v == null || v === '' || v === 'None' || v === 'nan') return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}
export const str = (v: unknown): string => (v == null || v === 'None' || v === 'nan' ? '' : String(v))
const coords = (v: unknown): LngLat[] => {
  if (Array.isArray(v)) return v as LngLat[]
  if (typeof v !== 'string' || !v.startsWith('[')) return []
  try {
    return JSON.parse(v) as LngLat[]
  } catch {
    return []
  }
}

export type SettlementStatus = 'cut_off' | 'hospital_cut_off' | 'detour' | 'connected' | 'no_baseline_route' | 'not_on_network'
export type Settlement = {
  id: string
  name: string
  place: string
  status: SettlementStatus
  at: LngLat
  buildings: number
  pCutOff: number | null
  hospital: string
  kmBefore: number | null
  kmAfter: number | null
  hoursBefore: number | null
  hoursAfter: number | null
  delayHours: number | null
  pathBefore: LngLat[]
  pathAfter: LngLat[]
  breaks: LngLat[]
  raw: Raw
}
export type Bridge = {
  id: string
  name: string
  highway: string
  status: 'likely_lost' | 'at_risk' | 'likely_standing' | 'not_observable' | string
  confidence: string
  pLost: number | null
  dropDb: number | null
  lengthM: number | null
  at: LngLat
  raw: Raw
}
export type Hospital = { id: string; name: string; amenity: string; status: string; isolated: boolean; at: LngLat; raw: Raw }
export type Facility = { id: string; name: string; type: string; hit: boolean; hazard: string; confidence: string; pMax: number | null; at: LngLat; raw: Raw }
export type LandingZone = {
  id: string
  name: string
  settlement: string
  distanceM: number
  elevationM: number
  slopeDeg: number
  areaM2: number
  suitability: 'optimal' | 'viable'
  status: string
  at: LngLat
  raw: Raw
}
export type Repair = {
  osmId: string
  name: string
  highway: string
  isBridge: boolean
  settlements: number
  buildings: number
  settlementNames: string[]
}

/** Order used everywhere a list of settlements is ranked: worst access first. */
export const STATUS_RANK: Record<string, number> = {
  cut_off: 0,
  hospital_cut_off: 1,
  detour: 2,
  connected: 3,
  no_baseline_route: 4,
  not_on_network: 5,
}

export const STATUS_LABEL: Record<string, string> = {
  cut_off: 'Cut off',
  hospital_cut_off: 'No hospital access',
  detour: 'Longer route',
  connected: 'Connected',
  no_baseline_route: 'No route before event',
  not_on_network: 'No mapped road',
  likely_lost: 'Likely lost',
  at_risk: 'At risk',
  likely_standing: 'Likely standing',
  not_observable: 'Not observable',
  accessible: 'Reachable',
  hit: 'In hazard zone',
  not_hit: 'Outside hazard zone',
}

/** Status colours. Red = lost or cut off, orange/amber = degraded, green = fine, grey = unknown. */
export const STATUS_COLOR: Record<string, string> = {
  cut_off: '#b91c1c',
  hospital_cut_off: '#ea580c',
  detour: '#d97706',
  connected: '#16a34a',
  no_baseline_route: '#94a3b8',
  not_on_network: '#cbd5e1',
  likely_lost: '#b91c1c',
  at_risk: '#ea580c',
  likely_standing: '#334155',
  not_observable: '#a8a29e',
  accessible: '#16a34a',
  hit: '#b45309',
  not_hit: '#94a3b8',
}

function first(g: GeoJSON.Geometry): LngLat {
  if (g.type === 'Point') return g.coordinates as LngLat
  if (g.type === 'LineString') {
    const c = g.coordinates
    return c[Math.floor(c.length / 2)] as LngLat
  }
  if (g.type === 'Polygon') {
    const ring = g.coordinates[0]
    const n = ring.length - 1 || 1
    return [ring.slice(0, n).reduce((s, c) => s + c[0], 0) / n, ring.slice(0, n).reduce((s, c) => s + c[1], 0) / n]
  }
  if (g.type === 'MultiPolygon') return first({ type: 'Polygon', coordinates: g.coordinates[0] })
  if (g.type === 'MultiLineString') return first({ type: 'LineString', coordinates: g.coordinates[0] })
  return [0, 0]
}

export function parseSettlement(p: Raw, at: LngLat): Settlement {
  return {
    id: str(p.osm_id) || `${at[0]},${at[1]}`,
    name: str(p.name) || 'Unnamed settlement',
    place: str(p.place),
    status: (str(p.status) || 'not_on_network') as SettlementStatus,
    at,
    buildings: num(p.buildings_500m) ?? 0,
    pCutOff: num(p.p_cut_off),
    hospital: str(p.hospital_name),
    kmBefore: num(p.km_to_hospital_pre),
    kmAfter: num(p.km_to_hospital_post),
    hoursBefore: num(p.hours_to_hospital_pre),
    hoursAfter: num(p.hours_to_hospital_post),
    delayHours: num(p.delay_hours_target),
    pathBefore: coords(p.hospital_path_pre),
    pathAfter: coords(p.hospital_path_post),
    breaks: coords(p.hospital_breaks),
    raw: p,
  }
}

export function parseBridge(p: Raw, at: LngLat): Bridge {
  return {
    id: str(p.osm_id),
    name: str(p.name),
    highway: str(p.highway),
    status: str(p.status),
    confidence: str(p.confidence),
    pLost: num(p.p_lost),
    dropDb: num(p.bright_drop_db),
    lengthM: num(p.length_m),
    at,
    raw: p,
  }
}

export function parseHospital(p: Raw, at: LngLat): Hospital {
  return {
    id: str(p.osm_id),
    name: str(p.name) || 'Unnamed health facility',
    amenity: str(p.amenity) || str(p.kind),
    status: str(p.status) || 'accessible',
    isolated: str(p.is_isolated) === 'True' || p.is_isolated === true,
    at,
    raw: p,
  }
}

function features(fc: GeoJSON.FeatureCollection | undefined): GeoJSON.Feature[] {
  return (fc?.features ?? []).filter((f) => f.geometry)
}

export function useJob(id: string | null) {
  return useQuery({
    queryKey: ['job', id],
    queryFn: () => api.job(id!),
    enabled: !!id,
    refetchInterval: (q) => (q.state.data && ['done', 'failed'].includes(q.state.data.status) ? false : 1500),
  })
}

/** Facts ledger as a lookup. `t(key)` returns the formatted text, or null when the fact is missing. */
export function useFacts(id: string | null, enabled: boolean) {
  const q = useQuery({ queryKey: ['facts', id], queryFn: () => api.facts(id!), enabled: !!id && enabled })
  return useMemo(() => {
    const by = Object.fromEntries((q.data ?? []).map((f: Fact) => [f.key, f])) as Record<string, Fact>
    const t = (k: string): string | null => (by[k] && by[k].text !== '' ? by[k].text : null)
    const n = (k: string): number | null => num(String(by[k]?.value ?? by[k]?.text ?? '').replaceAll(',', ''))
    return { by, t, n, loaded: q.isSuccess }
  }, [q.data, q.isSuccess])
}

function useGeo(id: string | null, done: boolean, name: string) {
  return useQuery({
    queryKey: ['gj', id, name],
    queryFn: () => api.geojson(id!, `web/${name}.geojson`),
    enabled: !!id && done,
    staleTime: Infinity,
  })
}

/** Everything the workspace views need from a finished analysis job. */
export function useJobData(id: string | null) {
  const job = useJob(id)
  const done = job.data?.status === 'done'
  const facts = useFacts(id, done)
  const s = useGeo(id, done, 'settlements')
  const b = useGeo(id, done, 'bridges')
  const h = useGeo(id, done, 'hospitals')
  const bl = useGeo(id, done, 'buildings')
  const lz = useGeo(id, done, 'hlz')
  const repairs = useQuery({
    queryKey: ['repairs', id],
    queryFn: () => fetch(fileUrl(id!, 'repairs.json')).then((r) => (r.ok ? (r.json() as Promise<Raw[]>) : [])),
    enabled: !!id && done,
    staleTime: Infinity,
  })

  const settlements = useMemo(
    () =>
      features(s.data)
        .map((f) => parseSettlement(f.properties as Raw, first(f.geometry)))
        .sort((a, b) => (STATUS_RANK[a.status] ?? 9) - (STATUS_RANK[b.status] ?? 9) || b.buildings - a.buildings),
    [s.data]
  )
  const bridges = useMemo(
    () =>
      features(b.data)
        .map((f) => parseBridge(f.properties as Raw, first(f.geometry)))
        .sort((x, y) => (y.pLost ?? -1) - (x.pLost ?? -1)),
    [b.data]
  )
  const hospitals = useMemo(() => features(h.data).map((f) => parseHospital(f.properties as Raw, first(f.geometry))), [h.data])
  const facilities = useMemo(
    () =>
      features(bl.data)
        .map((f): Facility | null => {
          const p = f.properties as Raw
          const type = str(p.facility_type)
          if (!['school', 'health', 'community'].includes(type)) return null
          return {
            id: str(p.osm_id),
            name: str(p.name) || (type === 'school' ? 'Unnamed school' : type === 'health' ? 'Unnamed health post' : 'Unnamed public building'),
            type,
            hit: str(p.status) === 'hit',
            hazard: str(p.hazard),
            confidence: str(p.confidence),
            pMax: num(p.p_max),
            at: first(f.geometry),
            raw: p,
          }
        })
        .filter((x): x is Facility => x !== null)
        .sort((x, y) => Number(y.hit) - Number(x.hit) || (y.pMax ?? 0) - (x.pMax ?? 0)),
    [bl.data]
  )
  const landingZones = useMemo(
    () =>
      features(lz.data).map(
        (f, i): LandingZone => {
          const p = f.properties as Raw
          return {
            id: str(p.name) || `hlz-${i}`,
            name: str(p.name) || 'Landing Zone',
            settlement: str(p.settlement),
            distanceM: num(p.distance_m) ?? 0,
            elevationM: num(p.elevation_m) ?? 0,
            slopeDeg: num(p.slope_deg) ?? 0,
            areaM2: num(p.area_m2) ?? 0,
            suitability: (str(p.suitability) as 'optimal' | 'viable') || 'viable',
            status: str(p.status) || 'cleared',
            at: first(f.geometry),
            raw: p,
          }
        }
      ),
    [lz.data]
  )
  const repairList = useMemo(
    () =>
      (repairs.data ?? []).map(
        (r): Repair => ({
          osmId: str(r.osm_id),
          name: str(r.name),
          highway: str(r.highway),
          isBridge: r.is_bridge === true || r.is_bridge === 'True',
          settlements: num(r.settlements_reconnected) ?? 0,
          buildings: num(r.buildings_reconnected) ?? 0,
          settlementNames: Array.isArray(r.settlement_names) ? (r.settlement_names as string[]) : [],
        })
      ),
    [repairs.data]
  )

  return {
    job,
    done,
    facts,
    settlements,
    bridges,
    hospitals,
    facilities,
    landingZones,
    repairs: repairList,
    loading: done && (s.isLoading || b.isLoading || h.isLoading),
  }
}

/** [minLon, minLat, maxLon, maxLat] of a polygon ring. */
export function bounds(ring: number[][]): [number, number, number, number] {
  const xs = ring.map((c) => c[0])
  const ys = ring.map((c) => c[1])
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
}

export function jobAoiRing(job: Job | undefined): number[][] | null {
  const aoi = job?.request?.aoi as { coordinates?: number[][][] } | undefined
  return aoi?.coordinates?.[0] ?? null
}

/** Short, human label for an analysis job: event date + AOI centre. */
export function jobLabel(job: Job): { title: string; sub: string } {
  const ev = String(job.request.event ?? '').slice(0, 10)
  const ring = jobAoiRing(job)
  let where = ''
  if (ring) {
    const [x0, y0, x1, y1] = bounds(ring)
    where = `${((y0 + y1) / 2).toFixed(3)}°N ${((x0 + x1) / 2).toFixed(3)}°E`
  }
  // summary.area_km2 holds the AOI split by class; their sum is the AOI area.
  const parts = (job.summary as { area_km2?: Record<string, number> } | null)?.area_km2
  const km2 = parts && typeof parts === 'object' ? Object.values(parts).reduce((a, b) => a + (Number(b) || 0), 0) : null
  if (km2 != null) where = `${km2 >= 100 ? Math.round(km2).toLocaleString('en') : km2.toFixed(1)} km² · ${where}`
  const d = ev ? new Date(`${ev}T00:00:00Z`) : null
  const title = d ? d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }) : 'Analysis'
  return { title, sub: where }
}

export const fmtKm = (v: number | null, d = 1) => (v == null ? '—' : `${v.toFixed(d)} km`)
export const fmtHours = (v: number | null) => {
  if (v == null) return '—'
  if (v < 1) return `${Math.round(v * 60)} min`
  return `${v.toFixed(1)} h`
}
export const pct = (v: number | null) => (v == null ? '—' : `${Math.round(v * 100)}%`)

/** Great-circle distance in km between two EPSG:4326 points. */
export function distanceKm(a: LngLat, b: LngLat): number {
  const r = Math.PI / 180
  const dLat = (b[1] - a[1]) * r
  const dLon = (b[0] - a[0]) * r
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[1] * r) * Math.cos(b[1] * r) * Math.sin(dLon / 2) ** 2
  return 12742 * Math.asin(Math.sqrt(h))
}

/** Ray-casting point-in-polygon on a lon/lat ring. */
export function inRing(p: LngLat, ring: number[][]): boolean {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i]
    const [xj, yj] = ring[j]
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}
