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
}

export type RasterLayer = { url: string; coordinates: [number, number][] }
export type Manifest = {
  rasters: Record<string, RasterLayer>
  vectors: Record<string, { url: string; count: number }>
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

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, {
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

export const fileUrl = (jobId: string, name: string) => `/api/jobs/${jobId}/files/${name}`

export const api = {
  config: () => req<Config>('/api/config'),
  scenes: (aoi: Polygon, event: string) =>
    req<ScenePreview>('/api/scenes', { method: 'POST', body: JSON.stringify({ aoi, event }) }),
  createJob: (body: { aoi: Polygon; event: string; use_s2: boolean; use_ml: boolean; relative_orbit?: number | null }) =>
    req<{ id: string }>('/api/jobs', { method: 'POST', body: JSON.stringify(body) }),
  createFlowPath: (body: { lon: number; lat: number; length_km: number; corridor_m: number }) =>
    req<{ id: string }>('/api/flowpath', { method: 'POST', body: JSON.stringify(body) }),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  jobs: () => req<Job[]>('/api/jobs'),
  facts: (id: string) => req<Fact[]>(`/api/jobs/${id}/facts`),
  manifest: (id: string) => req<Manifest>(fileUrl(id, 'layers.json')),
  geojson: <T = GeoJSON.FeatureCollection>(id: string, name: string) => req<T>(fileUrl(id, name)),
}
