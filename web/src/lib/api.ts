// Typed client for the Dhara API. All geometry is EPSG:4326 GeoJSON; all times are ISO 8601.

export type Polygon = { type: 'Polygon'; coordinates: number[][][] }

export type Config = {
  max_aoi_km2: number
  osm_snapshot_max: string
  attributions: string[]
  disclaimer: string
  ml_model_available: boolean
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed'

export type Job = {
  id: string
  kind: 'analysis' | 'flowpath'
  status: JobStatus
  stage: string
  progress: number
  created_utc: string
  updated_utc: string
  request: Record<string, unknown>
  summary: Record<string, unknown> | null
  error: string | null
  location_name?: string | null
  /** When the run left the queue (absent for runs made before queueing existed). */
  started_utc?: string | null
  /** Headline facts of a finished analysis, as display text (job list only). */
  brief?: Record<string, string> | null
  /** Runs ahead of this one in the queue (queued jobs only). */
  queue_ahead?: number | null
}

export type RasterLayer = { url: string; coordinates: [number, number][] }
export type ImageryLayer = RasterLayer & { date: string }
export type ImagerySet = { kind: 'radar' | 'optical'; sensor: string; track: string; before: ImageryLayer; after: ImageryLayer }
export type Manifest = {
  rasters: Record<string, RasterLayer>
  vectors: Record<string, { url: string; count: number }>
  /** Before/after images for the sweep: Sentinel-2 true colour when both scenes are clear enough,
   *  otherwise Sentinel-1 radar. `imagery_radar` is always the radar pair when available. */
  imagery?: ImagerySet
  imagery_radar?: ImagerySet
  /** Measured before/after change from the Sentinel-2 pair (clear sky on both dates only). */
  change?: ChangeSummary
}

export type ChangeSummary = {
  before_day: string
  after_day: string
  clear_both_km2: number
  clear_both_pct: number
  water_before_km2: number
  water_after_km2: number
  channel_before_km2: number
  channel_after_km2: number
  stations: number
  width_before_median_m: number | null
  width_after_median_m: number | null
  widened_stations: number
  width_gain_max_m: number | null
  shift_median_m: number | null
  shift_max_m: number | null
}

export type S1PassInfo = {
  platform: string
  relative_orbit: number
  orbit_state: string
  start: string
  scene_ids: string[]
  aoi_coverage: number
}
export type ScenePreview = {
  pairs: { relative_orbit: number; orbit_state: string; pre: S1PassInfo; post: S1PassInfo }[]
  notes: string[]
  area_km2: number
}

export type Fact = { key: string; value: string; unit: string; text: string; source: string }
export type ChatCard = {
  key: string
  title: string
  value: string
  unit: string
  detail: string
  source: string
  kind: 'metric' | 'limit'
}

export type Place = {
  id: string
  name: string
  name_ne?: string
  category: string
  center: [number, number]
  bbox: [number, number, number, number]
  polygon: Polygon
  description: string
}

export const API_BASE = (() => {
  if (typeof window !== 'undefined' && (window as unknown as { __DHARA_API_URL?: string }).__DHARA_API_URL) {
    return (window as unknown as { __DHARA_API_URL?: string }).__DHARA_API_URL!
  }
  if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL
  if (typeof window !== 'undefined') {
    const { hostname, port, protocol } = window.location
    // Only in local development on non-API dev ports (e.g. port 3000, 5000) redirect to :8000.
    // When served by FastAPI (port 8000, 8010, etc.) or behind a reverse proxy, use relative origin "".
    if (import.meta.env.DEV && port && port !== '8000' && port !== '8010' && port !== '5173' && port !== '4173') {
      const apiHost = hostname === 'localhost' ? '127.0.0.1' : hostname
      return `${protocol}//${apiHost}:8000`
    }
  }
  return ''
})()

export const apiUrl = (path: string) => (path.startsWith('http') ? path : `${API_BASE}${path}`)

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const url = apiUrl(path)
  const r = await fetch(url, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const b = await r.json()
      msg = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail ?? b)
    } catch {
      /* keep status text */
    }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

export const fileUrl = (jobId: string, name: string) => apiUrl(`/api/jobs/${jobId}/files/${name}`)

/** URL of a map layer. The settlements layer is served without per-settlement routes (~95 % of its
 *  size); `useSettlementRoute` fetches the route of the selected settlement. */
export const layerUrl = (jobId: string, name: string, path: string) =>
  fileUrl(jobId, path) + (name === 'settlements' ? '?slim=1' : '')

export type SplitMetrics = {
  f1_flood: number | null
  iou_flood: number | null
  precision_flood: number | null
  recall_flood: number | null
  event_f1_flood_ci95: [number, number] | null
}
export type ModelRun = {
  run_name: string
  /** False when the run's split let mountain tiles into training, so the mountain score is not held out. */
  mountain_held_out: boolean
  architecture: string | null
  training_data: string | null
  encoder_weights: string | null
  splits: { mountain_events?: number[]; counts?: Record<string, number> }
  test: SplitMetrics
  mountain: SplitMetrics
  /** Water IoU on Sen1Floods11 (test split, held-out Bolivia event); null when not scored. */
  sen1floods11: { test: number | null; bolivia: number | null } | null
}
export type ModelInfo = { deployed: ModelRun | null; runs: ModelRun[] }

export type ProfilePoint = {
  dist_km: number
  elevation_m: number
  lng: number
  lat: number
  slope_pct: number
}

export type ProfileSummary = {
  total_distance_km: number
  min_elevation_m: number
  max_elevation_m: number
  elevation_gain_m: number
  elevation_loss_m: number
  max_slope_pct: number
  avg_slope_pct: number
}

export type ProfileResponse = {
  points: ProfilePoint[]
  summary: ProfileSummary
}

export const api = {
  config: () => req<Config>('/api/config'),
  model: () => req<ModelInfo>('/api/model'),
  places: (q: string = '') => req<Place[]>(`/api/places?q=${encodeURIComponent(q)}`),
  scenes: (aoi: Polygon, event: string) =>
    req<ScenePreview>('/api/scenes', { method: 'POST', body: JSON.stringify({ aoi, event }) }),
  createJob: (body: { aoi: Polygon; event: string; use_s2: boolean; use_ml: boolean; relative_orbit?: number | null; name?: string; note?: string }) =>
    req<{ id: string; reused?: boolean }>('/api/jobs', { method: 'POST', body: JSON.stringify(body) }),
  createFlowPath: (body: { lon: number; lat: number; length_km: number; corridor_m: number; name?: string }) =>
    req<{ id: string }>('/api/flowpath', { method: 'POST', body: JSON.stringify(body) }),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  jobs: () => req<Job[]>('/api/jobs'),
  facts: (id: string) => req<Fact[]>(`/api/jobs/${id}/facts`),
  manifest: (id: string) => req<Manifest>(fileUrl(id, 'layers.json')),
  geojson: <T = GeoJSON.FeatureCollection>(id: string, name: string) => req<T>(fileUrl(id, name)),
  chat: (id: string, message: string, lang: string = 'en', entity?: { id: string; kind: string; coordinates: [number, number] }) =>
    req<{ answer: string; lang: string; provider?: string; cards?: ChatCard[] }>(`/api/jobs/${id}/chat`, {
      method: 'POST',
      body: JSON.stringify({ message, lang, entity }),
    }),
  explainAsset: (
    id: string,
    asset_type: 'bridge' | 'settlement' | 'road',
    asset_name: string,
    lang: string = 'en',
  ) =>
    req<{ explanation: string }>(`/api/jobs/${id}/explain-asset`, {
      method: 'POST',
      body: JSON.stringify({ asset_type, asset_name, lang }),
    }),
  profile: (id: string, path: number[][], step_m: number = 30.0) =>
    req<ProfileResponse>(`/api/jobs/${id}/profile`, {
      method: 'POST',
      body: JSON.stringify({ path, step_m }),
    }),
  // The one-page report shows two map images; the others are not sent.
  sitrepData: (id: string) => req<SitrepData>(`/api/jobs/${id}/sitrep/data?maps=overview,satellite_pair`),
  /** Report PDF state per language: ready, building, missing or "failed: …". */
  /** Suggested questions with answers computed from the job's facts (no language model). */
  suggested: (id: string, lang: string) =>
    req<{ id: string; question: string; answer: string; cards: ChatCard[]; provider: string }[]>(
      `/api/jobs/${id}/assistant/suggested?lang=${lang}`,
    ),
  reportStatus: (id: string) => req<Record<'en' | 'ne', string>>(`/api/jobs/${id}/report/status`),
  reportBuild: (id: string) => req<Record<'en' | 'ne', string>>(`/api/jobs/${id}/report/build`, { method: 'POST' }),
  /** One settlement's hospital route (JSON-encoded [lon, lat] lists), fetched when it is selected. */
  settlementRoute: (id: string, osmId: string, at: [number, number]) =>
    req<Record<string, string | null>>(
      `/api/jobs/${id}/settlements/route?osm_id=${encodeURIComponent(osmId)}&lon=${at[0]}&lat=${at[1]}`,
    ),
}

export type SitrepData = {
  job_id: string
  /** The run's own name, when it was given one. */
  name?: string | null
  facts: Record<string, string>
  values: Record<string, unknown>
  tables: {
    cut_off?: Array<Record<string, any>>
    bridges?: Array<Record<string, any>>
    detours?: Array<Record<string, any>>
    hospitals?: Array<Record<string, any>>
    repairs?: Array<Record<string, any>>
    s1?: Array<Record<string, any>>
    flowpaths?: Array<Record<string, any>>
    facilities_hit?: Array<Record<string, any>>
    [key: string]: Array<Record<string, any>> | undefined
  }
  maps: {
    overview?: string | null
    zoom?: string | null
    access?: string | null
    change?: string | null
    satellite_pair?: string | null
    flow?: string[]
    [key: string]: unknown
  }
  /** The images behind the satellite-pair figure (optical when clear enough, else radar). */
  imagery: { kind: 'optical' | 'radar'; sensor: string; before: string; after: string } | null
}


