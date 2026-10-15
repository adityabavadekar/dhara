// The map. Everything in EPSG:4326. Layers are created once on load with empty sources and then
// fed from the open job, the selection and the drawing state; clicks become `app.select(...)`.
import { useEffect, useMemo, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import type { GeoJSONSource, Map as MLMap } from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url'
import { useQuery } from '@tanstack/react-query'
import { ChevronsLeftRight } from 'lucide-react'
import { LAYERS, useApp, type LayerId, type Selection } from '@/context/app-state'
import { useTheme } from '@/context/theme-provider'
import { api, fileUrl, layerUrl, type Polygon } from '@/lib/api'
import { bounds, parseSettlement, STATUS_COLOR, str, useGeo, useJob, useSettlementRoute } from '@/lib/job'
import { formatArea, formatDistance, measurePathDistanceKm, measurePolygonAreaKm2 } from '@/lib/measure'
import { Legend } from './legend'
import { MapSearch } from './map-search'
import { MapToolbar, type Basemap, type MeasureMode } from './map-toolbar'
import { SelectionCard } from './selection-card'
import { StatusBar } from './status-bar'
import { SWEEP_DEFAULT, SweepPanel, sweepLayers } from './sweep-panel'

// MapLibre loads its worker from a separate file; import its URL so the build ships it.
maplibregl.setWorkerUrl(workerUrl)

// Esri serves tiles over HTTP/1.1, where a browser opens only 6 connections per host; spreading
// tiles over its two equivalent hosts doubles the parallel downloads when zooming out.
const ESRI_HOSTS = ['server', 'services'].map((h) => `https://${h}.arcgisonline.com/ArcGIS/rest/services`)
const esri = (path: string) => ESRI_HOSTS.map((base) => `${base}/${path}`)
// Label fonts are served with the app (web/public/assets/glyphs), not from a third-party font server.
const GLYPHS = `${window.location.origin}/assets/glyphs/{fontstack}/{range}.pbf`
export const BASEMAPS: Record<Basemap, { tiles: string[]; attribution: string; label: string; maxzoom: number }> = {
  // Esri has no gray-base tiles past z16 in rural Nepal and no imagery past z18 (it returns "Map data
  // not yet available" images), so those stop there and the map enlarges the last real level.
  light: {
    label: 'Light',
    tiles: esri('Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'),
    attribution: 'Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors (display only)',
    maxzoom: 16,
  },
  dark: {
    label: 'Dark',
    tiles: esri('Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'),
    attribution: 'Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors (display only)',
    maxzoom: 16,
  },
  imagery: {
    label: 'Imagery',
    tiles: esri('World_Imagery/MapServer/tile/{z}/{y}/{x}'),
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
  'aoi', 'draft', 'measure', 'buildings', 'roads', 'bridges', 'failed', 'settlements', 'hospitals', 'hlz',
  'route', 'route-breaks', 'hz-outline', 'change', 'f-path', 'f-corridor', 'f-settlements', 'f-start', 'sel', 'origin',
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
  helipads: ['v-hlz', 'v-hlz-symbol'],
  flowpath: ['f-corridor', 'f-path', 'f-pulse', 'f-settlements', 'f-settlements-label', 'f-start'],
}

const CLICKABLE: Record<string, Selection['kind']> = {
  'v-settlements': 'settlement',
  'v-hospitals': 'hospital',
  'v-bridges': 'bridge',
  'v-buildings': 'building',
  'v-roads': 'road',
  'v-hlz': 'point',
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
  const placeLabelFont = ['Open Sans Semibold']
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
  // Outline of mapped flood and debris, shown on both halves of the before/after sweep.
  map.addLayer({ id: 'hz-outline-casing', type: 'line', source: 'hz-outline', layout: { visibility: 'none' }, paint: { 'line-color': '#0f172a', 'line-width': 2.6, 'line-opacity': 0.7 } })
  map.addLayer({ id: 'hz-outline', type: 'line', source: 'hz-outline', layout: { visibility: 'none' }, paint: { 'line-color': ['match', ['get', 'hazard'], 'flood', '#38bdf8', '#fbbf24'], 'line-width': 1.2 } })
  // Measured before/after change for the sweep: water edges and river-channel width at stations.
  // Before-date features are drawn white on the left map, after-date ones cyan on the right map.
  const kind = (k: string) => ['==', ['get', 'kind'], k] as maplibregl.FilterSpecification
  const hidden = { visibility: 'none' } as const
  map.addLayer({ id: 'c-water-before', type: 'line', source: 'change', filter: kind('water_before'), layout: hidden, paint: { 'line-color': '#f8fafc', 'line-width': 1.2 } })
  map.addLayer({ id: 'c-water-after', type: 'line', source: 'change', filter: kind('water_after'), layout: hidden, paint: { 'line-color': '#22d3ee', 'line-width': 1.2 } })
  for (const [k, c] of [['before', '#f8fafc'], ['after', '#22d3ee']] as const) {
    map.addLayer({ id: `c-width-${k}-casing`, type: 'line', source: 'change', filter: kind(`width_${k}`), layout: { ...hidden, 'line-cap': 'round' }, paint: { 'line-color': '#0f172a', 'line-width': 6 } })
    map.addLayer({ id: `c-width-${k}`, type: 'line', source: 'change', filter: kind(`width_${k}`), layout: { ...hidden, 'line-cap': 'round' }, paint: { 'line-color': c, 'line-width': 3 } })
  }
  const label: maplibregl.SymbolLayerSpecification['layout'] = { 'text-font': ['Open Sans Semibold'], 'text-size': 11, 'text-anchor': 'left', 'text-offset': [0.4, 0], 'text-optional': true }
  const dark = { 'text-color': '#ffffff', 'text-halo-color': '#0f172a', 'text-halo-width': 1.6 }
  map.addLayer({
    id: 'c-label-before',
    type: 'symbol',
    source: 'change',
    filter: kind('station'),
    layout: { ...hidden, ...label, 'text-field': ['concat', ['to-string', ['get', 'width_before_m']], ' m'] },
    paint: dark,
  })
  map.addLayer({
    id: 'c-label-after',
    type: 'symbol',
    source: 'change',
    filter: kind('station'),
    layout: {
      ...hidden,
      ...label,
      'text-field': [
        'concat',
        ['to-string', ['get', 'width_after_m']],
        ' m',
        ['case', ['>=', ['get', 'width_change_m'], 0], ' (+', ' ('],
        ['to-string', ['get', 'width_change_m']],
        ')',
        ['case', ['>=', ['get', 'shift_m'], 50], ['concat', '\nmoved ', ['to-string', ['get', 'shift_m']], ' m'], ''],
      ],
    },
    paint: { ...dark, 'text-color': '#a5f3fc' },
  })
  // Routes carry a dark casing so they stay visible on satellite imagery as well as on plain maps.
  map.addLayer({
    id: 'route-casing',
    type: 'line',
    source: 'route',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': '#0f172a', 'line-width': 7, 'line-opacity': 0.75 },
  })
  map.addLayer({
    id: 'route-pre',
    type: 'line',
    source: 'route',
    filter: ['==', ['get', 'kind'], 'pre'],
    paint: { 'line-color': '#f8fafc', 'line-width': 3, 'line-dasharray': [1.5, 1.2] },
  })
  map.addLayer({
    id: 'route-post',
    type: 'line',
    source: 'route',
    filter: ['==', ['get', 'kind'], 'post'],
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': '#38bdf8', 'line-width': 4 },
  })
  map.addLayer({
    id: 'route-direct',
    type: 'line',
    source: 'route',
    filter: ['==', ['get', 'kind'], 'direct'],
    paint: { 'line-color': '#dc2626', 'line-width': 2.5, 'line-dasharray': [3, 2] },
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
  map.addLayer({
    id: 'v-hlz',
    type: 'circle',
    source: 'hlz',
    paint: { 'circle-radius': 8, 'circle-color': '#059669', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2 },
  })
  map.addLayer({
    id: 'v-hlz-symbol',
    type: 'symbol',
    source: 'hlz',
    layout: { 'text-field': 'H', 'text-font': ['Open Sans Bold'], 'text-size': 10, 'text-allow-overlap': true, 'text-ignore-placement': true },
    paint: { 'text-color': '#ffffff' },
  })
  map.addLayer({ id: 'f-settlements', type: 'circle', source: 'f-settlements', paint: { 'circle-radius': 5, 'circle-color': '#7c3aed', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1.5 } })
  map.addLayer({ id: 'f-start', type: 'circle', source: 'f-start', paint: { 'circle-radius': 7, 'circle-color': '#7c3aed', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2.5 } })
  map.addLayer({
    id: 'v-settlements-label',
    type: 'symbol',
    source: 'settlements',
    filter: ['any', ['in', ['get', 'status'], ['literal', ['cut_off', 'hospital_cut_off']]], ['>=', ['zoom'], 13]],
    layout: { 'text-field': ['get', 'name'], 'text-font': placeLabelFont, 'text-size': 11, 'text-offset': [0, 1.1], 'text-anchor': 'top', 'text-optional': true },
    paint: { 'text-color': ['match', ['get', 'status'], 'cut_off', '#991b1b', 'hospital_cut_off', '#9a3412', '#334155'], ...halo },
  })
  map.addLayer({
    id: 'v-hospitals-label',
    type: 'symbol',
    source: 'hospitals',
    minzoom: 11,
    layout: { 'text-field': ['get', 'name'], 'text-font': placeLabelFont, 'text-size': 10.5, 'text-offset': [0, 1.3], 'text-anchor': 'top', 'text-optional': true },
    paint: { 'text-color': '#991b1b', ...halo },
  })
  map.addLayer({
    id: 'f-settlements-label',
    type: 'symbol',
    source: 'f-settlements',
    layout: {
      'text-field': ['concat', ['to-string', ['get', 'order']], '. ', ['get', 'name']],
      'text-font': placeLabelFont,
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
  map.addLayer({
    id: 'sel-hospital-beacon',
    type: 'circle',
    source: 'sel',
    filter: ['==', ['get', 'isHospital'], true],
    paint: {
      'circle-radius': 20,
      'circle-color': 'rgba(220, 38, 38, 0.15)',
      'circle-stroke-color': '#dc2626',
      'circle-stroke-width': 2.5,
    },
  })
  map.addLayer({
    id: 'origin-marker',
    type: 'circle',
    source: 'origin',
    paint: {
      'circle-radius': 7,
      'circle-color': '#0284c7',
      'circle-stroke-color': '#ffffff',
      'circle-stroke-width': 2,
    },
  })
  map.addLayer({
    id: 'origin-label',
    type: 'symbol',
    source: 'origin',
    layout: {
      'text-field': 'Start',
      'text-font': ['Open Sans Semibold'],
      'text-size': 11,
      'text-offset': [0, -1.3],
      'text-anchor': 'bottom',
    },
    paint: { 'text-color': '#0369a1', ...halo },
  })
}

function applySweep(map: MLMap, keep: Set<string>) {
  for (const l of map.getStyle().layers) if (l.id !== 'bg') map.setLayoutProperty(l.id, 'visibility', keep.has(l.id) ? 'visible' : 'none')
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
  const secondaryRef = useRef<HTMLDivElement>(null)
  const secondaryMapRef = useRef<MLMap | null>(null)
  const isSyncingRef = useRef(false)
  const [ready, setReady] = useState(false)
  // Set when the browser cannot draw the map (no WebGL2); the panels keep working without it.
  const [mapError, setMapError] = useState<string | null>(null)
  const { resolvedTheme } = useTheme()
  const [basemap, setBasemap] = useState<Basemap>('imagery')
  const [measure, setMeasure] = useState<MeasureMode>('none')
  const [measurePts, setMeasurePts] = useState<[number, number][]>([])
  const [sweep, setSweep] = useState(false)
  const [sweepPos, setSweepPos] = useState(50)
  const [sweepKind, setSweepKind] = useState<'optical' | 'radar'>('optical')
  const [sweepOpts, setSweepOpts] = useState(SWEEP_DEFAULT)
  const sweepOptsRef = useRef(sweepOpts)
  sweepOptsRef.current = sweepOpts
  const flowRef = useRef<string | null>(null)
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
    let map: maplibregl.Map
    try {
      map = new maplibregl.Map({
      container: ref.current,
      style: {
        version: 8,
        glyphs: GLYPHS,
        sources: {},
        layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#e2e8f0' } }],
      },
      center: [85.32, 28.05],
      zoom: 8,
      attributionControl: false,
    })
    } catch (e) {
      setMapError(e instanceof Error && /webgl/i.test(e.message) ? 'webgl' : 'other')
      return
    }
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
  const manifest = useQuery({
    queryKey: ['manifest', app.jobId],
    queryFn: () => api.manifest(app.jobId!),
    enabled: !!app.jobId && done,
    staleTime: Infinity,
  })
  // Layers the side panel also lists are downloaded once and shared (same query keys as useJobData).
  const vectors = manifest.data?.vectors ?? {}
  const gBuildings = useGeo(app.jobId, done, 'buildings', vectors.buildings?.url)
  const gRoads = useGeo(app.jobId, done, 'roads', vectors.roads?.url)
  const gBridges = useGeo(app.jobId, done, 'bridges', vectors.bridges?.url)
  const gSettlements = useGeo(app.jobId, done, 'settlements', vectors.settlements?.url ? `${vectors.settlements.url}?slim=1` : undefined)
  const gHospitals = useGeo(app.jobId, done, 'hospitals', vectors.hospitals?.url)
  const gHlz = useGeo(app.jobId, done, 'hlz', vectors.hlz?.url)
  const shared = useMemo<Record<string, GeoJSON.FeatureCollection | undefined>>(
    () => ({
      buildings: gBuildings.data,
      roads: gRoads.data,
      bridges: gBridges.data,
      settlements: gSettlements.data,
      hospitals: gHospitals.data,
      hlz: gHlz.data,
    }),
    [gBuildings.data, gRoads.data, gBridges.data, gSettlements.data, gHospitals.data, gHlz.data],
  )
  const sharedRef = useRef(shared)
  sharedRef.current = shared
  const imgSet =
    sweepKind === 'radar' && manifest.data?.imagery_radar ? manifest.data.imagery_radar : manifest.data?.imagery

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
      ['hlz', 'hlz'],
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
      if (name in sharedRef.current) continue // drawn from the shared download, below
      const v = m.vectors[name]
      const url = v ? layerUrl(id, name, v.url) : null
      setData(map, src, url ?? EMPTY)
    }
    const hz = m.vectors.hazard_outline
    setData(map, 'hz-outline', hz ? fileUrl(id, hz.url) : EMPTY)
    setData(map, 'change', m.vectors.change ? fileUrl(id, m.vectors.change.url) : EMPTY)
  }, [manifest.data, app.jobId, done, ready])

  // Shared layers, set as each download arrives.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    for (const [name, data] of Object.entries(shared)) setData(map, name, data ?? EMPTY)
  }, [shared, ready])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return

    const img = imgSet
    if (map.getLayer('img-before')) map.removeLayer('img-before')
    if (map.getSource('img-before')) map.removeSource('img-before')
    if (sweep && img && app.jobId) {
      map.addSource('img-before', { type: 'image', url: fileUrl(app.jobId, img.before.url), coordinates: img.before.coordinates as never })
      map.addLayer({ id: 'img-before', type: 'raster', source: 'img-before', paint: { 'raster-fade-duration': 0 } }, 'aoi-fill')
    }

    if (sweep) {
      // During the sweep only imagery, the area boundary and the overlays chosen in the sweep bar
      // are shown, so the two images can be compared without clutter.
      applySweep(map, sweepLayers(sweepOpts, 'before'))
      const map2 = secondaryMapRef.current
      if (map2?.isStyleLoaded()) applySweep(map2, sweepLayers(sweepOpts, 'after'))
    }
    if (!sweep) {
      for (const l of map.getStyle().layers) if (l.id.startsWith('c-') || l.id.startsWith('hz-outline')) map.setLayoutProperty(l.id, 'visibility', 'none')
      // Layers the sweep hid that are not user toggles (routes, selection, drawing, measuring).
      const toggled = new Set(Object.values(LAYER_MAP).flat())
      for (const l of map.getStyle().layers)
        if (!toggled.has(l.id) && !l.id.startsWith('hz-outline') && !l.id.startsWith('c-') && !l.id.startsWith('r-') && l.id !== 'route-direct')
          map.setLayoutProperty(l.id, 'visibility', 'visible')
      // Normal interactive mode on Map 1
      const show = (id: LayerId) => app.visible[id] && (app.mode === 'workspace' || id === 'flowpath')
      for (const [toggle, layers] of Object.entries(LAYER_MAP)) {
        for (const l of layers) {
          if (map.getLayer(l)) map.setLayoutProperty(l, 'visibility', show(toggle as LayerId) ? 'visible' : 'none')
        }
      }
      if (map.getLayer('route-breaks')) map.setLayoutProperty('route-breaks', 'visibility', 'visible')
      if (map.getLayer('route-post')) map.setLayoutProperty('route-post', 'visibility', 'visible')
      if (map.getLayer('v-roads')) {
        map.setLayoutProperty('v-roads', 'visibility', show('roads') ? 'visible' : 'none')
        map.setPaintProperty('v-roads', 'line-color', '#7c2d12')
      }
      if (map.getLayer('v-bridges')) map.setPaintProperty('v-bridges', 'line-color', statusColor('#94a3b8'))
      if (map.getLayer('v-settlements')) map.setPaintProperty('v-settlements', 'circle-color', statusColor('#94a3b8'))
      if (map.getLayer('v-buildings')) {
        map.setPaintProperty('v-buildings', 'fill-color', ['match', ['get', 'status'], 'hit', '#b45309', '#94a3b8'])
      }
    }
  }, [sweep, app.visible, app.mode, ready, manifest.data, app.jobId, imgSet, sweepOpts])

  // ---------------------------------------------------------------- secondary map for sweep
  useEffect(() => {
    if (!sweep || !secondaryRef.current || !ready) {
      if (secondaryMapRef.current) {
        secondaryMapRef.current.remove()
        secondaryMapRef.current = null
      }
      return
    }
    const map1 = mapRef.current
    if (!map1) return

    const map2 = new maplibregl.Map({
      container: secondaryRef.current,
      style: {
        version: 8,
        glyphs: GLYPHS,
        sources: {},
        layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#e2e8f0' } }],
      },
      center: map1.getCenter(),
      zoom: map1.getZoom(),
      bearing: map1.getBearing(),
      pitch: map1.getPitch(),
      attributionControl: false,
      interactive: false,
    })

    let cleanSync: (() => void) | null = null

    map2.on('load', () => {
      const b = BASEMAPS[basemap]
      map2.addSource('basemap', { type: 'raster', tiles: b.tiles, tileSize: 256, attribution: b.attribution, maxzoom: b.maxzoom })
      map2.addLayer({ id: 'basemap', type: 'raster', source: 'basemap', paint: { 'raster-saturation': basemap === 'imagery' ? -0.3 : 0 } })

      for (const id of SOURCES) map2.addSource(id, { type: 'geojson', data: EMPTY })
      addLayers(map2)
      const img = imgSet
      if (img && app.jobId) {
        map2.addSource('img-after', { type: 'image', url: fileUrl(app.jobId, img.after.url), coordinates: img.after.coordinates as never })
        map2.addLayer({ id: 'img-after', type: 'raster', source: 'img-after', paint: { 'raster-fade-duration': 0 } }, 'aoi-fill')
        const hz = manifest.data?.vectors.hazard_outline
        if (hz) setData(map2, 'hz-outline', fileUrl(app.jobId, hz.url))
        const ch = manifest.data?.vectors.change
        if (ch) setData(map2, 'change', fileUrl(app.jobId, ch.url))
      }

      setData(map2, 'aoi', app.aoi ? { type: 'Feature', geometry: app.aoi, properties: {} } : EMPTY)

      const m = manifest.data
      const id = app.jobId
      if (m && id && done) {
        for (const r of RASTERS) {
          const layer = m.rasters[r]
          if (!layer) continue
          map2.addSource(`r-${r}`, { type: 'image', url: `${fileUrl(id, layer.url)}?v=${id}`, coordinates: layer.coordinates as never })
          map2.addLayer({ id: `r-${r}`, type: 'raster', source: `r-${r}`, paint: { 'raster-resampling': 'nearest', 'raster-opacity': 0.85, 'raster-fade-duration': 0 } }, 'aoi-fill')
        }
        const vec: [string, string][] = [
          ['buildings', 'buildings'],
          ['roads', 'roads'],
          ['bridges', 'bridges'],
          ['failed', 'failed_edges'],
          ['settlements', 'settlements'],
          ['hospitals', 'hospitals'],
          ['hlz', 'hlz'],
        ]
        for (const [src, name] of vec) {
          const own = sharedRef.current[name]
          if (own) {
            setData(map2, src, own)
            continue
          }
          const v = m.vectors[name]
          const url = v ? layerUrl(id, name, v.url) : null
          setData(map2, src, url ?? EMPTY)
        }
      }

      if (flowRef.current) {
        setData(map2, 'f-path', fileUrl(flowRef.current, 'flowpath.geojson'))
        setData(map2, 'f-corridor', fileUrl(flowRef.current, 'corridor.geojson'))
        setData(map2, 'f-settlements', fileUrl(flowRef.current, 'settlements.geojson'))
      }
      applySweep(map2, sweepLayers(sweepOptsRef.current, 'after'))
      map2.resize()
    })

    // Sync camera from map1 to map2
    const onMove = () => {
      if (!secondaryMapRef.current || isSyncingRef.current) return
      isSyncingRef.current = true
      secondaryMapRef.current.jumpTo({
        center: map1.getCenter(),
        zoom: map1.getZoom(),
        bearing: map1.getBearing(),
        pitch: map1.getPitch(),
      })
      isSyncingRef.current = false
    }
    map1.on('move', onMove)
    cleanSync = () => map1.off('move', onMove)

    const resizeTimer = setTimeout(() => {
      secondaryMapRef.current?.resize()
    }, 80)

    secondaryMapRef.current = map2
    return () => {
      clearTimeout(resizeTimer)
      cleanSync?.()
      map2.remove()
      secondaryMapRef.current = null
    }
  }, [sweep, ready, basemap, manifest.data, app.jobId, done, app.aoi, imgSet])

  // ---------------------------------------------------------------- flow path
  const flow = useJob(app.flowJobId)
  const flowOk = !!app.flowJobId && flow.data?.status === 'done'
  flowRef.current = flowOk ? app.flowJobId : null
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
  const selProps = useSettlementRoute(app.jobId, app.selection)
  const routeSettlement = useMemo(() => {
    const s = app.selection
    if (!s || s.kind !== 'settlement' || !selProps) return null
    return parseSettlement(selProps, s.lngLat)
  }, [app.selection, selProps])

  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    const s = app.selection
    const isHosp = s?.kind === 'hospital'
    const selFeat = s ? { type: 'Feature' as const, geometry: { type: 'Point' as const, coordinates: s.lngLat }, properties: { isHospital: isHosp } } : EMPTY
    const originFeat = s?.originLngLat ? { type: 'Feature' as const, geometry: { type: 'Point' as const, coordinates: s.originLngLat }, properties: { name: 'Origin' } } : EMPTY

    setData(map, 'sel', selFeat)
    setData(map, 'origin', originFeat)
    if (secondaryMapRef.current) {
      setData(secondaryMapRef.current, 'sel', selFeat)
      setData(secondaryMapRef.current, 'origin', originFeat)
    }

    const r = routeSettlement
    const feats: GeoJSON.Feature[] = []
    // Only real road routes are drawn; a straight line between two points would look like a route.
    if (s?.routePath && s.routePath.length > 1) {
      feats.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: s.routePath }, properties: { kind: 'post' } })
    } else if (r) {
      if (r.pathBefore.length > 1) feats.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: r.pathBefore }, properties: { kind: 'pre' } })
      if (r.pathAfter.length > 1) feats.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: r.pathAfter }, properties: { kind: 'post' } })
    }

    const routeData = { type: 'FeatureCollection' as const, features: feats }
    const breaksData = {
      type: 'FeatureCollection' as const,
      features: (r?.breaks ?? []).map((c) => ({ type: 'Feature' as const, geometry: { type: 'Point' as const, coordinates: c }, properties: {} })),
    }

    setData(map, 'route', routeData)
    setData(map, 'route-breaks', breaksData)
    if (secondaryMapRef.current) {
      setData(secondaryMapRef.current, 'route', routeData)
      setData(secondaryMapRef.current, 'route-breaks', breaksData)
    }

    // Frame the settlement, hospital, origin, and entire evacuation route
    const pts = [
      ...(s?.routePath?.length ? [s.lngLat] : []),
      ...(s?.routePath ?? []),
      ...(r?.pathBefore ?? []),
      ...(r?.pathAfter ?? []),
    ]
    if (pts.length > 1) {
      const b = new maplibregl.LngLatBounds(pts[0], pts[0])
      for (const c of pts) b.extend(c)
      map.fitBounds(b, { padding: { top: 80, bottom: 60, left: 360, right: 100 }, duration: 700, maxZoom: 15 })
    } else if (s && !r) {
      map.flyTo({ center: s.lngLat, zoom: Math.max(map.getZoom(), 13), duration: 700 })
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

  // Say what the map now shows whenever the view switches the layers, then fade out.
  const VIEW_NAME: Record<string, string> = { overview: 'Overview', flooding: 'Flooding', damage: 'Damage', cutoff: 'Cut off', flowpath: 'Flood path' }
  const [caption, setCaption] = useState<string | null>(null)
  const firstView = useRef(true)
  useEffect(() => {
    if (firstView.current) {
      firstView.current = false
      return
    }
    const names = LAYERS.filter((l) => app.visible[l.id]).map((l) => l.label.toLowerCase())
    setCaption(`${VIEW_NAME[app.view] ?? ''}: map shows ${names.join(', ')}`)
    const t = setTimeout(() => setCaption(null), 4000)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [app.view])

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
      {mapError && (
        <div className='absolute inset-0 z-30 flex items-center justify-center bg-muted p-6 text-center'>
          <div className='max-w-xs space-y-1'>
            <div className='text-sm font-semibold'>The map cannot be shown in this browser</div>
            <p className='text-xs text-muted-foreground'>
              {mapError === 'webgl'
                ? 'It needs WebGL 2, which is turned off or unsupported here. Try another browser or turn on hardware acceleration.'
                : 'The map failed to start. Reload the page or try another browser.'}{' '}
              The results, report and assistant still work.
            </p>
          </div>
        </div>
      )}

      {sweep && (
        <div
          className='pointer-events-none absolute inset-0 z-10 overflow-hidden'
          style={{
            clipPath: `inset(0 0 0 ${sweepPos}%)`,
            WebkitClipPath: `inset(0 0 0 ${sweepPos}%)`,
          }}
        >
          {/* MapLibre forces position: relative on its container, so it needs a sized wrapper. */}
          <div ref={secondaryRef} className='h-full w-full' />
        </div>
      )}

      {sweep && (
        <>
          <div
            // touch-none: a finger drag moves the divider instead of scrolling or panning the page.
            className='absolute inset-y-0 z-30 -ml-5 flex w-10 cursor-ew-resize touch-none justify-center'
            style={{ left: `${sweepPos}%` }}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              const box = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect()
              const move = (ev: PointerEvent) => setSweepPos(Math.min(99, Math.max(1, ((ev.clientX - box.left) / box.width) * 100)))
              const up = () => {
                window.removeEventListener('pointermove', move)
                window.removeEventListener('pointerup', up)
                window.removeEventListener('pointercancel', up)
              }
              window.addEventListener('pointermove', move)
              window.addEventListener('pointerup', up)
              window.addEventListener('pointercancel', up)
              e.preventDefault()
            }}
            role='slider'
            aria-label='Before and after divider'
            aria-valuenow={Math.round(sweepPos)}
          >
            <div className='h-full w-0.5 bg-white shadow-[0_0_8px_rgba(0,0,0,0.6)]' />
            <div className='absolute top-1/2 flex size-10 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white bg-neutral-900 text-white shadow-lg'>
              <ChevronsLeftRight className='size-4' />
            </div>
          </div>
          <div className='pointer-events-none absolute top-14 inset-x-0 z-30 flex justify-between gap-2 px-3'>
            <span className='rounded-md bg-neutral-900/85 px-2.5 py-1.5 text-xs text-white shadow'>
              <b>Before</b> · {imgSet?.before.date ?? '—'}
              <span className='block text-[0.6875rem] text-white/70'>{imgSet?.sensor}</span>
            </span>
            <span className='rounded-md bg-neutral-900/85 px-2.5 py-1.5 text-right text-xs text-white shadow'>
              <b>After</b> · {imgSet?.after.date ?? '—'}
              <span className='block text-[0.6875rem] text-white/70'>{imgSet?.sensor}</span>
            </span>
          </div>
          <SweepPanel jobId={app.jobId} change={manifest.data?.change} opts={sweepOpts} setOpts={setSweepOpts} hasFlow={!!flowRef.current} />
          <div className='absolute right-3 bottom-3 z-30 flex items-center gap-2 rounded-md bg-neutral-900/85 px-2.5 py-1.5 text-[0.6875rem] text-white/85 shadow @3xl:max-w-[calc(100%-17rem)]'>
            {manifest.data?.imagery?.kind === 'optical' && manifest.data?.imagery_radar && (
              <div className='flex shrink-0 rounded border border-white/20 p-0.5'>
                {(['optical', 'radar'] as const).map((k) => (
                  <button
                    key={k}
                    type='button'
                    onClick={() => setSweepKind(k)}
                    className={`rounded px-1.5 py-0.5 font-medium ${sweepKind === k ? 'bg-white text-neutral-900' : 'text-white/80'}`}
                  >
                    {k === 'optical' ? 'Optical' : 'Radar'}
                  </button>
                ))}
              </div>
            )}
            {/* The explanation needs room; narrow maps show just the Optical/Radar switch. */}
            <span className='hidden @3xl:inline'>
              {imgSet?.kind === 'optical'
                ? 'Natural colour. White patches are cloud; switch to radar to see through them.'
                : 'Radar: dark = calm water or smooth ground, bright = rough ground or slopes facing the satellite.'}{' '}
              Drag the divider; most ground looks the same because it did not change.
            </span>
          </div>
        </>
      )}


      <div className='pointer-events-none absolute inset-x-3 top-3 z-40 flex items-start gap-2'>
        <div className='pointer-events-auto min-w-0 max-w-sm flex-1'>
          <MapSearch />
        </div>
        <div className='pointer-events-auto ml-auto shrink-0'>
          <MapToolbar basemap={basemap} setBasemap={setBasemap} measure={measure} setMeasure={setMeasure} sweep={sweep} setSweep={setSweep} sweepAvailable={!!manifest.data?.imagery} />
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

      {caption && !drawHint && !sweep && (
        <div className='pointer-events-none absolute top-16 left-1/2 z-20 max-w-[80%] -translate-x-1/2 rounded-md bg-neutral-900/90 px-3 py-1.5 text-center text-xs text-white shadow-lg'>
          {caption}
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
        // On phones the card is a short sheet at the bottom (its body scrolls) so the map stays visible.
        <div className='absolute bottom-3 left-3 z-10 flex max-h-[55%] w-[20rem] max-w-[calc(100%-1.5rem)] sm:max-h-[calc(100%-5rem)]'>
          <SelectionCard />
        </div>
      )}

      <div className={app.mode === 'new' || sweep ? 'hidden' : 'absolute right-12 bottom-3 z-10 hidden @xl:block'}>
        <Legend visible={app.visible} route={!!routeSettlement?.pathBefore.length} layers={LAYERS} />
      </div>
    </div>
    <StatusBar basemapCredit={BASEMAPS[basemap].attribution} mPerPx={view.mPerPx} zoom={view.zoom} cursor={cursor} />
    </div>
  )
}
