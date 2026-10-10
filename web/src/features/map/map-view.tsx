import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import type { GeoJSONSource, LngLatLike, Map as MLMap } from 'maplibre-gl'
import { useQuery } from '@tanstack/react-query'
import { Layers3 } from 'lucide-react'
import { useApp, type LayerId } from '@/context/app-state'
import { useTheme } from '@/context/theme-provider'
import { api, fileUrl, type Polygon } from '@/lib/api'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

type Basemap = 'light' | 'dark' | 'imagery' | 'osm'
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services'
const BASEMAPS: Record<Basemap, { tiles: string[]; attribution: string; label: string }> = {
  light: {
    label: 'Light',
    tiles: [`${ESRI}/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}`],
    attribution: 'Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors (display only)',
  },
  dark: {
    label: 'Dark',
    tiles: [`${ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`],
    attribution: 'Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors (display only)',
  },
  imagery: {
    label: 'Imagery',
    tiles: [`${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`],
    attribution: 'Imagery © Esri, Maxar, Earthstar Geographics (display only)',
  },
  osm: {
    label: 'OpenStreetMap',
    tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
    attribution: '© OpenStreetMap contributors (current map, display only)',
  },
}

const EMPTY: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] }
const RASTERS = ['unobs', 'water', 'flood', 'debris', 'confidence'] as const
// Which toggles control which map layers
const LAYER_MAP: Record<LayerId, string[]> = {
  debris: ['r-debris'],
  flood: ['r-flood'],
  water: ['r-water'],
  unobs: ['r-unobs'],
  confidence: ['r-confidence'],
  buildings: ['v-buildings'],
  roads: ['v-roads'],
  bridges: ['v-bridges'],
  failed: ['v-failed'],
  settlements: ['v-settlements', 'v-settlements-label'],
  flowpath: ['f-corridor', 'f-path', 'f-settlements', 'f-settlements-label'],
}

const STATUS_COLOR = [
  'match',
  ['get', 'status'],
  'cut_off', '#b91c1c',
  'hospital_cut_off', '#ea580c',
  'detour', '#d97706',
  'connected', '#16a34a',
  'no_baseline_route', '#94a3b8',
  '#94a3b8',
] as unknown as maplibregl.ExpressionSpecification

function rectangle(a: maplibregl.LngLat, b: maplibregl.LngLat): Polygon {
  const [x0, x1] = [Math.min(a.lng, b.lng), Math.max(a.lng, b.lng)]
  const [y0, y1] = [Math.min(a.lat, b.lat), Math.max(a.lat, b.lat)]
  return { type: 'Polygon', coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]] }
}

export function MapView() {
  const ref = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MLMap | null>(null)
  const [ready, setReady] = useState(false)
  const { resolvedTheme } = useTheme()
  const [basemap, setBasemap] = useState<Basemap>(resolvedTheme === 'dark' ? 'dark' : 'light')
  const app = useApp()
  const appRef = useRef(app)
  appRef.current = app

  // ---------------------------------------------------------------- init
  useEffect(() => {
    if (!ref.current || mapRef.current) return
    const map = new maplibregl.Map({
      container: ref.current,
      style: {
        version: 8,
        glyphs: 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',
        sources: {},
        layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#e2e8f0' } }],
      },
      center: [85.32, 28.05],
      zoom: 9,
      attributionControl: { compact: true },
    })
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), 'top-right')
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left')
    map.on('load', () => {
      for (const id of ['aoi', 'draft', 'buildings', 'roads', 'bridges', 'failed', 'settlements', 'f-path', 'f-corridor', 'f-settlements', 'f-start'])
        map.addSource(id, { type: 'geojson', data: EMPTY })
      setReady(true)
    })
    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  // ---------------------------------------------------------------- basemap
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    if (map.getLayer('basemap')) map.removeLayer('basemap')
    if (map.getSource('basemap')) map.removeSource('basemap')
    const b = BASEMAPS[basemap]
    map.addSource('basemap', { type: 'raster', tiles: b.tiles, tileSize: 256, attribution: b.attribution, maxzoom: basemap === 'osm' ? 19 : 16 })
    const first = map.getStyle().layers.find((l) => l.id !== 'bg')?.id
    map.addLayer({ id: 'basemap', type: 'raster', source: 'basemap', paint: { 'raster-saturation': basemap === 'imagery' ? -0.2 : 0 } }, first)
  }, [basemap, ready])

  // ---------------------------------------------------------------- AOI + flow start display
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const src = map.getSource('aoi') as GeoJSONSource
    src.setData(app.aoi ? { type: 'Feature', geometry: app.aoi, properties: {} } : EMPTY)
    if (!map.getLayer('aoi-line')) {
      map.addLayer({ id: 'aoi-fill', type: 'fill', source: 'aoi', paint: { 'fill-color': '#0f172a', 'fill-opacity': 0.04 } })
      map.addLayer({ id: 'aoi-line', type: 'line', source: 'aoi', paint: { 'line-color': '#0f172a', 'line-width': 1.5, 'line-dasharray': [3, 2] } })
      map.addLayer({ id: 'draft-line', type: 'line', source: 'draft', paint: { 'line-color': '#2563eb', 'line-width': 1.5 } })
      map.addLayer({ id: 'draft-pts', type: 'circle', source: 'draft', paint: { 'circle-radius': 3, 'circle-color': '#2563eb' }, filter: ['==', '$type', 'Point'] })
      map.addLayer({ id: 'f-start', type: 'circle', source: 'f-start', paint: { 'circle-radius': 6, 'circle-color': '#7c3aed', 'circle-stroke-color': '#fff', 'circle-stroke-width': 2 } })
    }
  }, [app.aoi, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    ;(map.getSource('f-start') as GeoJSONSource).setData(
      app.flowPoint ? { type: 'Feature', geometry: { type: 'Point', coordinates: app.flowPoint }, properties: {} } : EMPTY
    )
  }, [app.flowPoint, ready])

  // ---------------------------------------------------------------- drawing
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const mode = app.drawMode
    const canvas = map.getCanvasContainer()
    canvas.style.cursor = mode === 'none' ? '' : 'crosshair'
    const draft = map.getSource('draft') as GeoJSONSource
    if (mode === 'none') {
      draft.setData(EMPTY)
      return
    }
    let start: maplibregl.LngLat | null = null
    const verts: [number, number][] = []

    const onDown = (e: maplibregl.MapMouseEvent) => {
      if (mode !== 'box') return
      e.preventDefault()
      map.dragPan.disable()
      start = e.lngLat
    }
    const onMove = (e: maplibregl.MapMouseEvent) => {
      if (mode === 'box' && start) draft.setData({ type: 'Feature', geometry: rectangle(start, e.lngLat), properties: {} })
      if (mode === 'polygon' && verts.length)
        draft.setData({
          type: 'FeatureCollection',
          features: [
            { type: 'Feature', geometry: { type: 'LineString', coordinates: [...verts, [e.lngLat.lng, e.lngLat.lat]] }, properties: {} },
            ...verts.map((v) => ({ type: 'Feature' as const, geometry: { type: 'Point' as const, coordinates: v }, properties: {} })),
          ],
        })
    }
    const onUp = (e: maplibregl.MapMouseEvent) => {
      if (mode !== 'box' || !start) return
      map.dragPan.enable()
      const poly = rectangle(start, e.lngLat)
      start = null
      draft.setData(EMPTY)
      appRef.current.setAoi(poly)
      appRef.current.setDrawMode('none')
    }
    const onClick = (e: maplibregl.MapMouseEvent) => {
      if (mode === 'polygon') verts.push([e.lngLat.lng, e.lngLat.lat])
      if (mode === 'flowpoint') {
        appRef.current.setFlowPoint([+e.lngLat.lng.toFixed(5), +e.lngLat.lat.toFixed(5)])
        appRef.current.setDrawMode('none')
      }
    }
    const onDbl = (e: maplibregl.MapMouseEvent) => {
      if (mode !== 'polygon') return
      e.preventDefault()
      if (verts.length >= 3) {
        appRef.current.setAoi({ type: 'Polygon', coordinates: [[...verts, verts[0]]] })
        appRef.current.setDrawMode('none')
      }
    }
    if (mode === 'polygon') map.doubleClickZoom.disable()
    map.on('mousedown', onDown)
    map.on('mousemove', onMove)
    map.on('mouseup', onUp)
    map.on('click', onClick)
    map.on('dblclick', onDbl)
    return () => {
      map.off('mousedown', onDown)
      map.off('mousemove', onMove)
      map.off('mouseup', onUp)
      map.off('click', onClick)
      map.off('dblclick', onDbl)
      map.dragPan.enable()
      map.doubleClickZoom.enable()
      canvas.style.cursor = ''
    }
  }, [app.drawMode, ready])

  // ---------------------------------------------------------------- analysis results
  const job = useQuery({ queryKey: ['job', app.jobId], queryFn: () => api.job(app.jobId!), enabled: !!app.jobId })
  const done = job.data?.status === 'done'
  const manifest = useQuery({
    queryKey: ['manifest', app.jobId],
    queryFn: () => api.manifest(app.jobId!),
    enabled: !!app.jobId && done,
  })

  useEffect(() => {
    const map = mapRef.current
    const m = manifest.data
    const id = app.jobId
    if (!map || !ready) return
    for (const r of RASTERS) {
      if (map.getLayer(`r-${r}`)) map.removeLayer(`r-${r}`)
      if (map.getSource(`r-${r}`)) map.removeSource(`r-${r}`)
    }
    if (!m || !id) {
      for (const s of ['buildings', 'roads', 'bridges', 'failed', 'settlements']) (map.getSource(s) as GeoJSONSource).setData(EMPTY)
      return
    }
    const below = map.getLayer('aoi-fill') ? 'aoi-fill' : undefined
    for (const r of RASTERS) {
      const layer = m.rasters[r]
      if (!layer) continue
      map.addSource(`r-${r}`, { type: 'image', url: `${fileUrl(id, layer.url)}?v=${id}`, coordinates: layer.coordinates as never })
      map.addLayer({ id: `r-${r}`, type: 'raster', source: `r-${r}`, paint: { 'raster-resampling': 'nearest', 'raster-opacity': 0.9, 'raster-fade-duration': 0 } }, below)
    }
    const vec: [string, string][] = [['buildings', 'buildings'], ['roads', 'roads'], ['bridges', 'bridges'], ['failed', 'failed_edges'], ['settlements', 'settlements']]
    for (const [src, name] of vec) {
      const v = m.vectors[name]
      ;(map.getSource(src) as GeoJSONSource).setData(v ? fileUrl(id, v.url) : EMPTY)
    }
    if (!map.getLayer('v-roads')) {
      map.addLayer({ id: 'v-roads', type: 'line', source: 'roads', filter: ['==', ['get', 'status'], 'hit'], paint: { 'line-color': '#7c2d12', 'line-width': ['interpolate', ['linear'], ['zoom'], 9, 1, 14, 3] } })
      map.addLayer({ id: 'v-failed', type: 'line', source: 'failed', paint: { 'line-color': '#dc2626', 'line-width': ['interpolate', ['linear'], ['zoom'], 9, 1.5, 14, 4] } })
      map.addLayer({ id: 'v-buildings', type: 'fill', source: 'buildings', paint: { 'fill-color': ['match', ['get', 'status'], 'hit', '#b45309', '#78716c'], 'fill-outline-color': '#451a03', 'fill-opacity': 0.85 } })
      map.addLayer({
        id: 'v-bridges', type: 'line', source: 'bridges',
        paint: {
          'line-color': ['match', ['get', 'status'], 'likely_lost', '#b91c1c', 'at_risk', '#ea580c', 'likely_standing', '#0f172a', '#94a3b8'],
          'line-width': ['interpolate', ['linear'], ['zoom'], 9, 3, 15, 8],
        },
      })
      map.addLayer({
        id: 'v-settlements', type: 'circle', source: 'settlements',
        paint: {
          'circle-color': STATUS_COLOR,
          'circle-radius': ['interpolate', ['linear'], ['get', 'buildings_500m'], 0, 3, 200, 6, 1000, 10],
          'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1.2,
        },
      })
      map.addLayer({
        id: 'v-settlements-label', type: 'symbol', source: 'settlements',
        filter: ['in', ['get', 'status'], ['literal', ['cut_off', 'hospital_cut_off']]],
        layout: { 'text-field': ['get', 'name'], 'text-font': ['Open Sans Semibold'], 'text-size': 11, 'text-offset': [0, 1.1], 'text-anchor': 'top' },
        paint: { 'text-color': '#7f1d1d', 'text-halo-color': '#ffffff', 'text-halo-width': 1.2 },
      })
      for (const l of ['v-settlements', 'v-bridges', 'v-buildings']) {
        map.on('click', l, (e) => popup(map, e))
        map.on('mouseenter', l, () => (map.getCanvas().style.cursor = 'pointer'))
        map.on('mouseleave', l, () => (map.getCanvas().style.cursor = ''))
      }
    }
    // fit to AOI on first load
    const all = (m.rasters.debris ?? m.rasters.classes)?.coordinates
    if (all) {
      const xs = all.map((c) => c[0])
      const ys = all.map((c) => c[1])
      map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 40, duration: 600 })
    }
  }, [manifest.data, app.jobId, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const v = app.visible
    for (const [toggle, layers] of Object.entries(LAYER_MAP)) {
      for (const l of layers) if (map.getLayer(l)) map.setLayoutProperty(l, 'visibility', v[toggle as LayerId] ? 'visible' : 'none')
    }
  }, [app.visible, ready, manifest.data, app.flowJobId])

  // ---------------------------------------------------------------- flow path
  const flow = useQuery({ queryKey: ['job', app.flowJobId], queryFn: () => api.job(app.flowJobId!), enabled: !!app.flowJobId })
  useEffect(() => {
    const map = mapRef.current
    const id = app.flowJobId
    if (!map || !ready) return
    const ok = id && flow.data?.status === 'done'
    ;(map.getSource('f-path') as GeoJSONSource).setData(ok ? fileUrl(id, 'flowpath.geojson') : EMPTY)
    ;(map.getSource('f-corridor') as GeoJSONSource).setData(ok ? fileUrl(id, 'corridor.geojson') : EMPTY)
    ;(map.getSource('f-settlements') as GeoJSONSource).setData(ok ? fileUrl(id, 'settlements.geojson') : EMPTY)
    if (!map.getLayer('f-path')) {
      map.addLayer({ id: 'f-corridor', type: 'fill', source: 'f-corridor', paint: { 'fill-color': '#7c3aed', 'fill-opacity': 0.08 } })
      map.addLayer({ id: 'f-path', type: 'line', source: 'f-path', paint: { 'line-color': '#7c3aed', 'line-width': 2.5 } })
      map.addLayer({ id: 'f-settlements', type: 'circle', source: 'f-settlements', paint: { 'circle-radius': 5, 'circle-color': '#ffffff', 'circle-stroke-color': '#7c3aed', 'circle-stroke-width': 2 } })
      map.addLayer({
        id: 'f-settlements-label', type: 'symbol', source: 'f-settlements',
        layout: { 'text-field': ['concat', ['to-string', ['get', 'order']], '. ', ['get', 'name']], 'text-font': ['Open Sans Semibold'], 'text-size': 11, 'text-offset': [0.8, 0], 'text-anchor': 'left' },
        paint: { 'text-color': '#4c1d95', 'text-halo-color': '#fff', 'text-halo-width': 1.2 },
      })
    }
  }, [flow.data?.status, app.flowJobId, ready])

  // ---------------------------------------------------------------- fly to
  useEffect(() => {
    const map = mapRef.current
    if (!map || !app.focus) return
    map.flyTo({ center: [app.focus.lon, app.focus.lat] as LngLatLike, zoom: app.focus.zoom ?? 13, duration: 700 })
  }, [app.focus])

  return (
    <div className='absolute inset-0'>
      <div ref={ref} className='h-full w-full' />
      <div className='absolute top-3 left-3 flex gap-2'>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size='sm' variant='secondary' className='shadow-none border'>
              <Layers3 className='size-4' /> {BASEMAPS[basemap].label}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align='start'>
            <DropdownMenuRadioGroup value={basemap} onValueChange={(v) => setBasemap(v as Basemap)}>
              {(Object.keys(BASEMAPS) as Basemap[]).map((k) => (
                <DropdownMenuRadioItem key={k} value={k}>{BASEMAPS[k].label}</DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      {app.drawMode !== 'none' && (
        <div className='absolute top-3 left-1/2 -translate-x-1/2 rounded-md border bg-background px-3 py-1.5 text-xs'>
          {app.drawMode === 'box' && 'Drag on the map to draw a box'}
          {app.drawMode === 'polygon' && 'Click to add vertices · double-click to finish'}
          {app.drawMode === 'flowpoint' && 'Click an upstream point'}
        </div>
      )}
    </div>
  )
}

function esc(s: unknown): string {
  return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
}

function popup(map: MLMap, e: maplibregl.MapLayerMouseEvent) {
  const f = e.features?.[0]
  if (!f) return
  const p = f.properties as Record<string, unknown>
  const rows: [string, unknown][] =
    f.layer.id === 'v-settlements'
      ? [
          ['Status', String(p.status).replaceAll('_', ' ')],
          ['Buildings within 500 m', p.buildings_500m],
          ['P(cut off)', p.p_cut_off == null ? 'n/a' : `${Math.round(Number(p.p_cut_off) * 100)}%`],
          ['km to hospital before', p.km_to_hospital_pre ?? 'n/a'],
          ['km to hospital after', p.km_to_hospital_post ?? 'no route'],
        ]
      : f.layer.id === 'v-bridges'
        ? [
            ['Status', String(p.status).replaceAll('_', ' ')],
            ['Confidence', p.confidence],
            ['P(lost)', p.p_lost == null ? 'n/a' : Number(p.p_lost).toFixed(2)],
            ['Radar return drop (dB)', p.bright_drop_db == null ? 'n/a' : Number(p.bright_drop_db).toFixed(1)],
            ['Length (m)', p.length_m],
          ]
        : [
            ['Status', p.status],
            ['Hazard', p.hazard || '—'],
            ['Confidence', p.confidence],
          ]
  const title = esc(p.name || p.osm_id || 'Feature')
  new maplibregl.Popup({ closeButton: true, maxWidth: '280px' })
    .setLngLat(e.lngLat)
    .setHTML(
      `<div class="text-xs"><div class="font-semibold mb-1">${title}</div>${rows
        .map(([k, v]) => `<div class="flex justify-between gap-4"><span class="text-muted-foreground">${esc(k)}</span><span>${esc(v)}</span></div>`)
        .join('')}<div class="mt-1 text-[10px] text-muted-foreground">${esc(p.osm_id)}</div></div>`
    )
    .addTo(map)
}
