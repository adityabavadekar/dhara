// The map. Everything in EPSG:4326. Layers are created once on load with empty sources and then
// fed from the open job, the selection and the drawing state; clicks become `app.select(...)`.
import { useEffect, useMemo, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import type { GeoJSONSource, Map as MLMap } from 'maplibre-gl'
import { useQuery } from '@tanstack/react-query'
import { LAYERS, useApp, type LayerId, type Selection } from '@/context/app-state'
import { useTheme } from '@/context/theme-provider'
import { api, fileUrl, type Polygon } from '@/lib/api'
import { bounds, parseSettlement, STATUS_COLOR, str, useJob } from '@/lib/job'
import { formatArea, formatDistance, measurePathDistanceKm, measurePolygonAreaKm2 } from '@/lib/measure'
import { Legend } from './legend'
import { MapSearch } from './map-search'
import { MapToolbar, type Basemap, type MeasureMode } from './map-toolbar'
import { SelectionCard } from './selection-card'
import { StatusBar } from './status-bar'

const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services'
export const BASEMAPS: Record<Basemap, { tiles: string[]; attribution: string; label: string; maxzoom: number }> = {
  light: {
    label: 'Light',
    tiles: [`${ESRI}/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}`],
    attribution: 'Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors (display only)',
    maxzoom: 16,
  },
  dark: {
    label: 'Dark',
    tiles: [`${ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`],
    attribution: 'Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors (display only)',
    maxzoom: 16,
  },
  imagery: {
    label: 'Imagery',
    tiles: [`${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`],
    attribution: 'Imagery © Esri, Maxar, Earthstar Geographics (display only, not an analysis input)',
    maxzoom: 18,
  },
  osm: {
    label: 'Streets',
    tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
    attribution: '© OpenStreetMap contributors (current map, display only)',
    maxzoom: 19,
  },
}

const EMPTY: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] }
const RASTERS = ['unobs', 'water', 'flood', 'debris', 'confidence'] as const
const SOURCES = [
  'aoi', 'draft', 'measure', 'buildings', 'roads', 'bridges', 'failed', 'settlements', 'hospitals',
  'route', 'route-breaks', 'f-path', 'f-corridor', 'f-settlements', 'f-start', 'sel',
]

const LAYER_MAP: Record<LayerId, string[]> = {
  debris: ['r-debris'],
  flood: ['r-flood'],
  water: ['r-water'],
  unobs: ['r-unobs'],
  confidence: ['r-confidence'],
  buildings: ['v-buildings'],
  roads: ['v-roads'],
  bridges: ['v-bridges-casing', 'v-bridges'],
  failed: ['v-failed'],
  settlements: ['v-settlements', 'v-settlements-label'],
  hospitals: ['v-hospitals', 'v-hospitals-symbol', 'v-hospitals-label'],
  flowpath: ['f-corridor', 'f-path', 'f-pulse', 'f-settlements', 'f-settlements-label', 'f-start'],
}

const CLICKABLE: Record<string, Selection['kind']> = {
  'v-settlements': 'settlement',
  'v-hospitals': 'hospital',
  'v-bridges': 'bridge',
  'v-buildings': 'building',
  'v-roads': 'road',
  'f-settlements': 'flow-settlement',
}

const statusColor = (fallback: string) =>
  ['match', ['get', 'status'], ...Object.entries(STATUS_COLOR).flat(), fallback] as unknown as maplibregl.ExpressionSpecification

function rectangle(a: maplibregl.LngLat, b: maplibregl.LngLat): Polygon {
  const [x0, x1] = [Math.min(a.lng, b.lng), Math.max(a.lng, b.lng)]
  const [y0, y1] = [Math.min(a.lat, b.lat), Math.max(a.lat, b.lat)]
  return { type: 'Polygon', coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]] }
}

function anchor(f: maplibregl.MapGeoJSONFeature, at: maplibregl.LngLat): [number, number] {
  const g = f.geometry
  if (g.type === 'Point') return g.coordinates as [number, number]
  return [at.lng, at.lat]
}

function setData(map: MLMap, id: string, data: GeoJSON.GeoJSON | string) {
  ;(map.getSource(id) as GeoJSONSource | undefined)?.setData(data as GeoJSON.GeoJSON)
}

function addLayers(map: MLMap) {
  const w = (a: number, b: number) => ['interpolate', ['linear'], ['zoom'], 9, a, 15, b] as unknown as maplibregl.ExpressionSpecification
  const halo = { 'text-halo-color': '#ffffff', 'text-halo-width': 1.4 }
  map.addLayer({ id: 'aoi-fill', type: 'fill', source: 'aoi', paint: { 'fill-color': '#0284c7', 'fill-opacity': 0.03 } })
  map.addLayer({ id: 'v-roads', type: 'line', source: 'roads', filter: ['==', ['get', 'status'], 'hit'], paint: { 'line-color': '#7c2d12', 'line-width': w(1, 3) } })
  map.addLayer({ id: 'v-failed', type: 'line', source: 'failed', paint: { 'line-color': '#dc2626', 'line-width': w(1.5, 4) } })
  map.addLayer({
    id: 'v-buildings',
    type: 'fill',
    source: 'buildings',
    paint: { 'fill-color': ['match', ['get', 'status'], 'hit', '#b45309', '#94a3b8'], 'fill-outline-color': '#451a03', 'fill-opacity': 0.85 },
  })
  map.addLayer({ id: 'f-corridor', type: 'fill', source: 'f-corridor', paint: { 'fill-color': '#7c3aed', 'fill-opacity': 0.07 } })
  map.addLayer({ id: 'f-path', type: 'line', source: 'f-path', paint: { 'line-color': '#7c3aed', 'line-width': 3 } })
  map.addLayer({ id: 'f-pulse', type: 'line', source: 'f-path', paint: { 'line-color': '#ffffff', 'line-width': 1.5, 'line-dasharray': [0, 4, 3] } })
  map.addLayer({
    id: 'route-pre',
    type: 'line',
    source: 'route',
    filter: ['==', ['get', 'kind'], 'pre'],
    paint: { 'line-color': '#64748b', 'line-width': 3, 'line-dasharray': [1.5, 1.5] },
  })
  map.addLayer({
    id: 'route-post',
    type: 'line',
    source: 'route',
    filter: ['==', ['get', 'kind'], 'post'],
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': '#2563eb', 'line-width': 4 },
  })
  map.addLayer({
    id: 'v-bridges-casing',
    type: 'line',
    source: 'bridges',
    layout: { 'line-cap': 'round' },
    paint: { 'line-color': '#ffffff', 'line-width': w(5, 11) },
  })
  map.addLayer({
    id: 'v-bridges',
    type: 'line',
    source: 'bridges',
    layout: { 'line-cap': 'round' },
    paint: { 'line-color': statusColor('#94a3b8'), 'line-width': w(3, 8) },
  })
  map.addLayer({
    id: 'route-breaks',
    type: 'circle',
    source: 'route-breaks',
    paint: { 'circle-radius': 4, 'circle-color': '#dc2626', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1.5 },
  })
  map.addLayer({
    id: 'v-settlements',
    type: 'circle',
    source: 'settlements',
    paint: {
      'circle-color': statusColor('#94a3b8'),
      'circle-radius': ['interpolate', ['linear'], ['to-number', ['get', 'buildings_500m'], 0], 0, 3.5, 200, 6, 1000, 10],
      'circle-stroke-color': '#ffffff',
      'circle-stroke-width': 1.2,
    },
  })
  map.addLayer({
    id: 'v-hospitals',
    type: 'circle',
    source: 'hospitals',
    paint: { 'circle-radius': 8, 'circle-color': '#ffffff', 'circle-stroke-color': '#dc2626', 'circle-stroke-width': 2.5 },
  })
  map.addLayer({
    id: 'v-hospitals-symbol',
    type: 'symbol',
    source: 'hospitals',
    layout: { 'text-field': '+', 'text-font': ['Open Sans Bold'], 'text-size': 13, 'text-allow-overlap': true, 'text-ignore-placement': true },
    paint: { 'text-color': '#dc2626' },
  })
  map.addLayer({ id: 'f-settlements', type: 'circle', source: 'f-settlements', paint: { 'circle-radius': 5, 'circle-color': '#7c3aed', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1.5 } })
  map.addLayer({ id: 'f-start', type: 'circle', source: 'f-start', paint: { 'circle-radius': 7, 'circle-color': '#7c3aed', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2.5 } })
  map.addLayer({
    id: 'v-settlements-label',
    type: 'symbol',
    source: 'settlements',
    filter: ['any', ['in', ['get', 'status'], ['literal', ['cut_off', 'hospital_cut_off']]], ['>=', ['zoom'], 13]],
    layout: { 'text-field': ['get', 'name'], 'text-font': ['Open Sans Semibold'], 'text-size': 11, 'text-offset': [0, 1.1], 'text-anchor': 'top', 'text-optional': true },
    paint: { 'text-color': ['match', ['get', 'status'], 'cut_off', '#991b1b', 'hospital_cut_off', '#9a3412', '#334155'], ...halo },
  })
  map.addLayer({
    id: 'v-hospitals-label',
    type: 'symbol',
    source: 'hospitals',
    minzoom: 11,
    layout: { 'text-field': ['get', 'name'], 'text-font': ['Open Sans Semibold'], 'text-size': 10.5, 'text-offset': [0, 1.3], 'text-anchor': 'top', 'text-optional': true },
    paint: { 'text-color': '#991b1b', ...halo },
  })
  map.addLayer({
    id: 'f-settlements-label',
    type: 'symbol',
    source: 'f-settlements',
    layout: {
      'text-field': ['concat', ['to-string', ['get', 'order']], '. ', ['get', 'name']],
      'text-font': ['Open Sans Semibold'],
      'text-size': 11,
      'text-offset': [0.9, 0],
      'text-anchor': 'left',
      'text-optional': true,
    },
    paint: { 'text-color': '#4c1d95', ...halo },
  })
  map.addLayer({ id: 'aoi-casing', type: 'line', source: 'aoi', paint: { 'line-color': '#ffffff', 'line-width': 4, 'line-opacity': 0.7 } })
  map.addLayer({ id: 'aoi-line', type: 'line', source: 'aoi', paint: { 'line-color': '#0284c7', 'line-width': 2, 'line-dasharray': [3, 2] } })
  map.addLayer({ id: 'draft-line', type: 'line', source: 'draft', paint: { 'line-color': '#0284c7', 'line-width': 2 } })
  map.addLayer({ id: 'draft-points', type: 'circle', source: 'draft', filter: ['==', ['geometry-type'], 'Point'], paint: { 'circle-radius': 4, 'circle-color': '#ffffff', 'circle-stroke-color': '#0284c7', 'circle-stroke-width': 2 } })
  map.addLayer({ id: 'measure-fill', type: 'fill', source: 'measure', filter: ['==', ['geometry-type'], 'Polygon'], paint: { 'fill-color': '#0f172a', 'fill-opacity': 0.08 } })
  map.addLayer({ id: 'measure-line', type: 'line', source: 'measure', paint: { 'line-color': '#0f172a', 'line-width': 2, 'line-dasharray': [2, 1] } })
  map.addLayer({ id: 'measure-points', type: 'circle', source: 'measure', filter: ['==', ['geometry-type'], 'Point'], paint: { 'circle-radius': 4, 'circle-color': '#ffffff', 'circle-stroke-color': '#0f172a', 'circle-stroke-width': 2 } })
  map.addLayer({ id: 'sel-ring', type: 'circle', source: 'sel', paint: { 'circle-radius': 13, 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': '#0f172a', 'circle-stroke-width': 2.5 } })
}

/** A MapLibre control button that brings the analysed area back into view, like Google Maps' recentre. */
class RecenterControl implements maplibregl.IControl {
  private el?: HTMLDivElement
  constructor(private onClick: () => void) {}
  onAdd() {
    this.el = document.createElement('div')
    this.el.className = 'maplibregl-ctrl maplibregl-ctrl-group'
    const b = document.createElement('button')
    b.type = 'button'
    b.style.color = '#333'
    b.title = 'Back to the analysed area'
    b.setAttribute('aria-label', 'Back to the analysed area')
    b.innerHTML =
      '<svg viewBox="0 0 24 24" width="18" height="18" style="margin:auto;display:block" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3.5"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/><circle cx="12" cy="12" r="7.5"/></svg>'
    b.onclick = () => this.onClick()
    this.el.appendChild(b)
    return this.el
  }
  onRemove() {
    this.el?.remove()
  }
}

export function MapView() {
  const ref = useRef<HTMLDivElement>(null)
  const mapRef = useRef<MLMap | null>(null)
  const [ready, setReady] = useState(false)
  const { resolvedTheme } = useTheme()
  const [basemap, setBasemap] = useState<Basemap>(resolvedTheme === 'dark' ? 'dark' : 'light')
  const [measure, setMeasure] = useState<MeasureMode>('none')
  const [measurePts, setMeasurePts] = useState<[number, number][]>([])
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null)
  const [cursor, setCursor] = useState<[number, number] | null>(null)
  const [view, setView] = useState({ zoom: 8, mPerPx: 600 })
  const app = useApp()
  const appRef = useRef(app)
  appRef.current = app
  // Frames the analysed area (or the drawn area) again; used by the recentre button.
  const recenterRef = useRef<() => void>(() => {})
  recenterRef.current = () => {
    const ring = app.aoi?.coordinates[0]
    if (ring) app.fitTo(bounds(ring))
  }
  const measureRef = useRef(measure)
  measureRef.current = measure

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
      zoom: 8,
      attributionControl: false,
    })
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), 'bottom-right')
    map.addControl(new RecenterControl(() => recenterRef.current()), 'bottom-right')
    map.on('load', () => {
      for (const id of SOURCES) map.addSource(id, { type: 'geojson', data: EMPTY })
      addLayers(map)

      const interactive = Object.keys(CLICKABLE)
      const onView = () => {
        const lat = map.getCenter().lat
        setView({ zoom: map.getZoom(), mPerPx: (40075016.686 * Math.cos((lat * Math.PI) / 180)) / (512 * 2 ** map.getZoom()) })
      }
      onView()
      // Update on every frame while zooming or panning, like a normal web map, not just at the end.
      let pending = 0
      map.on('move', () => {
        if (!pending) pending = requestAnimationFrame(() => ((pending = 0), onView()))
      })
      map.on('mousemove', (e) => setCursor([e.lngLat.lng, e.lngLat.lat]))
      map.on('mousemove', (e) => {
        if (appRef.current.drawMode !== 'none' || measureRef.current !== 'none') return
        const f = map.queryRenderedFeatures(e.point, { layers: interactive.filter((l) => map.getLayer(l)) })[0]
        map.getCanvas().style.cursor = f ? 'pointer' : ''
        const name = f ? str(f.properties?.name) : ''
        setHover(f && name ? { x: e.point.x, y: e.point.y, text: name } : null)
      })
      map.on('mouseout', () => {
        setHover(null)
        setCursor(null)
      })
      map.on('click', (e) => {
        const a = appRef.current
        if (a.drawMode !== 'none') return
        if (measureRef.current !== 'none') {
          setMeasurePts((p) => [...p, [e.lngLat.lng, e.lngLat.lat]])
          return
        }
        const f = map.queryRenderedFeatures(e.point, { layers: interactive.filter((l) => map.getLayer(l)) })[0]
        if (f) {
          a.select({ kind: CLICKABLE[f.layer.id], props: f.properties as Record<string, unknown>, lngLat: anchor(f, e.lngLat) } as Selection)
          return
        }
        a.select(a.selection ? null : { kind: 'point', props: {}, lngLat: [e.lngLat.lng, e.lngLat.lat] })
      })
      setReady(true)
    })
    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
      setReady(false)
    }
  }, [])

  // ---------------------------------------------------------------- basemap
  useEffect(() => {
    setBasemap((b) => (b === 'light' || b === 'dark' ? (resolvedTheme === 'dark' ? 'dark' : 'light') : b))
  }, [resolvedTheme])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    if (map.getLayer('basemap')) map.removeLayer('basemap')
    if (map.getSource('basemap')) map.removeSource('basemap')
    const b = BASEMAPS[basemap]
    map.addSource('basemap', { type: 'raster', tiles: b.tiles, tileSize: 256, attribution: b.attribution, maxzoom: b.maxzoom })
    map.addLayer({ id: 'basemap', type: 'raster', source: 'basemap', paint: { 'raster-saturation': basemap === 'imagery' ? -0.3 : 0 } }, 'aoi-fill')
  }, [basemap, ready])

  // ---------------------------------------------------------------- AOI, drawing
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    setData(map, 'aoi', app.aoi ? { type: 'Feature', geometry: app.aoi, properties: {} } : EMPTY)
  }, [app.aoi, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const mode = app.drawMode
    const canvas = map.getCanvasContainer()
    canvas.style.cursor = mode === 'none' ? '' : 'crosshair'
    if (mode === 'none') {
      setData(map, 'draft', EMPTY)
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
      if (mode === 'box' && start) setData(map, 'draft', { type: 'Feature', geometry: rectangle(start, e.lngLat), properties: {} })
      if (mode === 'polygon' && verts.length)
        setData(map, 'draft', {
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
      setData(map, 'draft', EMPTY)
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
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && appRef.current.setDrawMode('none')
    if (mode === 'polygon') map.doubleClickZoom.disable()
    map.on('mousedown', onDown)
    map.on('mousemove', onMove)
    map.on('mouseup', onUp)
    map.on('click', onClick)
    map.on('dblclick', onDbl)
    window.addEventListener('keydown', onKey)
    return () => {
      map.off('mousedown', onDown)
      map.off('mousemove', onMove)
      map.off('mouseup', onUp)
      map.off('click', onClick)
      map.off('dblclick', onDbl)
      window.removeEventListener('keydown', onKey)
      map.dragPan.enable()
      map.doubleClickZoom.enable()
      canvas.style.cursor = ''
    }
  }, [app.drawMode, ready])

  // ---------------------------------------------------------------- analysis layers
  const job = useJob(app.jobId)
  const done = job.data?.status === 'done'
  const manifest = useQuery({ queryKey: ['manifest', app.jobId], queryFn: () => api.manifest(app.jobId!), enabled: !!app.jobId && done })

  useEffect(() => {
    const map = mapRef.current
    const m = manifest.data
    const id = app.jobId
    if (!map || !ready) return
    for (const r of RASTERS) {
      if (map.getLayer(`r-${r}`)) map.removeLayer(`r-${r}`)
      if (map.getSource(`r-${r}`)) map.removeSource(`r-${r}`)
    }
    const vec: [string, string][] = [
      ['buildings', 'buildings'],
      ['roads', 'roads'],
      ['bridges', 'bridges'],
      ['failed', 'failed_edges'],
      ['settlements', 'settlements'],
      ['hospitals', 'hospitals'],
    ]
    if (!m || !id || !done) {
      for (const [src] of vec) setData(map, src, EMPTY)
      return
    }
    for (const r of RASTERS) {
      const layer = m.rasters[r]
      if (!layer) continue
      map.addSource(`r-${r}`, { type: 'image', url: `${fileUrl(id, layer.url)}?v=${id}`, coordinates: layer.coordinates as never })
      map.addLayer({ id: `r-${r}`, type: 'raster', source: `r-${r}`, paint: { 'raster-resampling': 'nearest', 'raster-opacity': 0.85, 'raster-fade-duration': 0 } }, 'aoi-fill')
    }
    for (const [src, name] of vec) {
      const v = m.vectors[name]
      setData(map, src, v ? fileUrl(id, v.url) : EMPTY)
    }
  }, [manifest.data, app.jobId, done, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    // While setting up a new analysis the previous results would only get in the way.
    const show = (id: LayerId) => app.visible[id] && (app.mode === 'workspace' || id === 'flowpath')
    for (const [toggle, layers] of Object.entries(LAYER_MAP))
      for (const l of layers) if (map.getLayer(l)) map.setLayoutProperty(l, 'visibility', show(toggle as LayerId) ? 'visible' : 'none')
  }, [app.visible, app.mode, ready, manifest.data])

  // ---------------------------------------------------------------- flow path
  const flow = useJob(app.flowJobId)
  const flowOk = !!app.flowJobId && flow.data?.status === 'done'
  useEffect(() => {
    const map = mapRef.current
    const id = app.flowJobId
    if (!map || !ready) return
    setData(map, 'f-path', flowOk && id ? fileUrl(id, 'flowpath.geojson') : EMPTY)
    setData(map, 'f-corridor', flowOk && id ? fileUrl(id, 'corridor.geojson') : EMPTY)
    setData(map, 'f-settlements', flowOk && id ? fileUrl(id, 'settlements.geojson') : EMPTY)
    if (!flowOk || !id) return
    // Frame the whole traced path.
    fetch(fileUrl(id, 'corridor.geojson'))
      .then((r) => r.json() as Promise<GeoJSON.FeatureCollection>)
      .then((fc) => {
        const ring = (fc.features[0]?.geometry as GeoJSON.Polygon | undefined)?.coordinates?.[0]
        if (!ring?.length) return
        const b = new maplibregl.LngLatBounds(ring[0] as [number, number], ring[0] as [number, number])
        for (const c of ring) b.extend(c as [number, number])
        map.fitBounds(b, { padding: { top: 70, bottom: 50, left: 60, right: 200 }, duration: 700 })
      })
      .catch(() => {})
    let raf = 0
    let step = 0
    const tick = () => {
      step = (step + 0.08) % 7
      const s = Math.floor(step)
      if (map.getLayer('f-pulse')) map.setPaintProperty('f-pulse', 'line-dasharray', [s, 4, 7 - s])
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [flowOk, app.flowJobId, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    setData(map, 'f-start', app.flowPoint ? { type: 'Feature', geometry: { type: 'Point', coordinates: app.flowPoint }, properties: {} } : EMPTY)
  }, [app.flowPoint, ready])

  // ---------------------------------------------------------------- selection and route
  const routeSettlement = useMemo(() => {
    const s = app.selection
    if (!s || s.kind !== 'settlement') return null
    return parseSettlement(s.props, s.lngLat)
  }, [app.selection])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const s = app.selection
    setData(map, 'sel', s ? { type: 'Feature', geometry: { type: 'Point', coordinates: s.lngLat }, properties: {} } : EMPTY)
    const r = routeSettlement
    const feats: GeoJSON.Feature[] = []
    if (r?.pathBefore.length && r.pathBefore.length > 1) feats.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: r.pathBefore }, properties: { kind: 'pre' } })
    if (r?.pathAfter.length && r.pathAfter.length > 1) feats.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: r.pathAfter }, properties: { kind: 'post' } })
    setData(map, 'route', { type: 'FeatureCollection', features: feats })
    setData(map, 'route-breaks', {
      type: 'FeatureCollection',
      features: (r?.breaks ?? []).map((c) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: c }, properties: {} })),
    })
    // Frame the settlement and its whole hospital route (the fly-to below is skipped for these).
    const pts = [...(r?.pathBefore ?? []), ...(r?.pathAfter ?? [])]
    if (r && pts.length > 1) {
      const b = new maplibregl.LngLatBounds(r.at, r.at)
      for (const c of pts) b.extend(c)
      map.fitBounds(b, { padding: { top: 70, bottom: 50, left: 350, right: 90 }, duration: 700, maxZoom: 14 })
    }
  }, [app.selection, routeSettlement, ready])

  // ---------------------------------------------------------------- camera
  useEffect(() => {
    const map = mapRef.current
    if (!map || !app.focus) return
    if (routeSettlement && routeSettlement.pathBefore.length > 1) return
    map.flyTo({ center: [app.focus.lon, app.focus.lat], zoom: Math.max(map.getZoom(), app.focus.zoom ?? 13), duration: 700 })
  }, [app.focus])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !app.fit) return
    const [x0, y0, x1, y1] = app.fit.bounds
    map.fitBounds([[x0, y0], [x1, y1]], { padding: 48, duration: 700 })
  }, [app.fit])

  // ---------------------------------------------------------------- measuring
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    if (measure === 'none' || measurePts.length === 0) {
      setData(map, 'measure', EMPTY)
      return
    }
    const feats: GeoJSON.Feature[] = measurePts.map((c) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: c }, properties: {} }))
    if (measure === 'area' && measurePts.length >= 3)
      feats.push({ type: 'Feature', geometry: { type: 'Polygon', coordinates: [[...measurePts, measurePts[0]]] }, properties: {} })
    else if (measurePts.length >= 2) feats.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: measurePts }, properties: {} })
    setData(map, 'measure', { type: 'FeatureCollection', features: feats })
    map.getCanvasContainer().style.cursor = 'crosshair'
  }, [measure, measurePts, ready])

  useEffect(() => {
    const map = mapRef.current
    if (map && measure === 'none') map.getCanvasContainer().style.cursor = ''
    setMeasurePts([])
  }, [measure])

  const measured =
    measure === 'distance' ? formatDistance(measurePathDistanceKm(measurePts)) : measure === 'area' ? formatArea(measurePolygonAreaKm2(measurePts)) : ''

  const drawHint =
    app.drawMode === 'box'
      ? 'Drag to draw the area. Esc cancels.'
      : app.drawMode === 'polygon'
        ? 'Click to add corners, double-click to finish. Esc cancels.'
        : app.drawMode === 'flowpoint'
          ? 'Click the upstream start point. Esc cancels.'
          : null

  return (
    <div className='flex h-full w-full flex-col'>
    <div className='@container relative min-h-0 w-full flex-1'>
      <div ref={ref} className='h-full w-full' />

      <div className='pointer-events-none absolute inset-x-3 top-3 z-10 flex items-start gap-2'>
        <div className='pointer-events-auto min-w-0 max-w-sm flex-1'>
          <MapSearch />
        </div>
        <div className='pointer-events-auto ml-auto shrink-0'>
          <MapToolbar basemap={basemap} setBasemap={setBasemap} measure={measure} setMeasure={setMeasure} />
        </div>
      </div>

      {drawHint && (
        <div className='absolute top-16 left-1/2 z-10 -translate-x-1/2 rounded-md border bg-background/95 px-3 py-1.5 text-xs font-medium shadow-sm'>
          {drawHint}
        </div>
      )}

      {measure !== 'none' && (
        <div className='absolute top-16 left-1/2 z-10 flex -translate-x-1/2 items-center gap-3 rounded-md border bg-background/95 px-3 py-1.5 text-xs shadow-sm'>
          <span className='text-muted-foreground'>{measure === 'distance' ? 'Distance' : 'Area'}</span>
          <span className='font-semibold tabular-nums'>{measurePts.length ? measured : 'Click the map'}</span>
          {measurePts.length > 0 && (
            <button type='button' className='text-muted-foreground hover:text-foreground' onClick={() => setMeasurePts((p) => p.slice(0, -1))}>
              Undo
            </button>
          )}
          <button type='button' className='text-muted-foreground hover:text-foreground' onClick={() => setMeasure('none')}>
            Done
          </button>
        </div>
      )}

      {hover && (
        <div
          className='pointer-events-none absolute z-10 rounded border bg-background/95 px-1.5 py-0.5 text-[0.6875rem] font-medium shadow-sm'
          style={{ left: hover.x + 12, top: hover.y - 10 }}
        >
          {hover.text}
        </div>
      )}

      {app.selection && (
        <div className='absolute bottom-3 left-3 z-10 w-[20rem] max-w-[calc(100%-1.5rem)]'>
          <SelectionCard />
        </div>
      )}

      <div className={app.mode === 'new' ? 'hidden' : 'absolute right-12 bottom-3 z-10 hidden @xl:block'}>
        <Legend visible={app.visible} route={!!routeSettlement?.pathBefore.length} layers={LAYERS} />
      </div>
    </div>
    <StatusBar basemapCredit={BASEMAPS[basemap].attribution} mPerPx={view.mPerPx} zoom={view.zoom} cursor={cursor} />
    </div>
  )
}
