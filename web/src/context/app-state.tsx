import { createContext, useContext, useMemo, useState } from 'react'
import type { Polygon } from '@/lib/api'

export type DrawMode = 'none' | 'box' | 'polygon' | 'flowpoint'
/** Workspace views. Each one answers one question and drives which map layers are shown. */
export type View = 'overview' | 'flooding' | 'damage' | 'cutoff' | 'flowpath'
export type Mode = 'workspace' | 'new'

export const LAYERS = [
  { id: 'flood', label: 'New water', group: 'Hazard', color: '#2563eb' },
  { id: 'debris', label: 'Debris and scour', group: 'Hazard', color: '#c27803' },
  { id: 'water', label: 'Water before the event', group: 'Hazard', color: '#7dd3fc' },
  { id: 'unobs', label: 'Not observable', group: 'Hazard', color: '#a8a29e' },
  { id: 'confidence', label: 'Confidence tiers', group: 'Hazard', color: '#b91c1c' },
  { id: 'buildings', label: 'Buildings hit', group: 'Damage', color: '#b45309' },
  { id: 'roads', label: 'Roads and trails hit', group: 'Damage', color: '#7c2d12' },
  { id: 'bridges', label: 'Bridges', group: 'Damage', color: '#0f172a' },
  { id: 'failed', label: 'Impassable road links', group: 'Access', color: '#dc2626' },
  { id: 'settlements', label: 'Settlements', group: 'Access', color: '#475569' },
  { id: 'hospitals', label: 'Hospitals and clinics', group: 'Access', color: '#dc2626' },
  { id: 'helipads', label: 'Possible landing areas', group: 'Access', color: '#059669' },
  { id: 'flowpath', label: 'Flood path', group: 'Flood path', color: '#7c3aed' },
] as const
export type LayerId = (typeof LAYERS)[number]['id']

type Visible = Record<LayerId, boolean>
const on = (...ids: LayerId[]): Visible =>
  Object.fromEntries(LAYERS.map((l) => [l.id, ids.includes(l.id)])) as Visible

/** Layers each view turns on. The user can still change them from the layer menu. */
export const VIEW_LAYERS: Record<View, Visible> = {
  overview: on('flood', 'debris', 'unobs', 'bridges', 'failed', 'settlements', 'hospitals', 'helipads'),
  flooding: on('flood', 'debris', 'water', 'unobs'),
  damage: on('flood', 'debris', 'buildings', 'roads', 'bridges'),
  cutoff: on('debris', 'flood', 'failed', 'bridges', 'settlements', 'hospitals', 'helipads'),
  flowpath: on('flowpath', 'debris', 'flood', 'settlements'),
}

/** What the user clicked, on the map or in a list. Coordinates are EPSG:4326. */
export type Selection = {
  kind: 'settlement' | 'bridge' | 'hospital' | 'building' | 'road' | 'flow-settlement' | 'point'
  props: Record<string, unknown>
  lngLat: [number, number]
  originLngLat?: [number, number]
  routePath?: [number, number][]
}
export type AssistantEntityContext = { id: string; kind: Selection['kind']; coordinates: [number, number] }
export type AssistantDraft = { question: string; entity?: AssistantEntityContext }

type State = {
  mode: Mode
  setMode: (m: Mode) => void
  view: View
  setView: (v: View) => void
  aoi: Polygon | null
  setAoi: (p: Polygon | null) => void
  drawMode: DrawMode
  setDrawMode: (m: DrawMode) => void
  jobId: string | null
  openJob: (id: string | null, aoi?: Polygon | null) => void
  flowJobId: string | null
  setFlowJobId: (id: string | null) => void
  flowPoint: [number, number] | null
  setFlowPoint: (p: [number, number] | null) => void
  visible: Visible
  toggle: (id: LayerId, on: boolean) => void
  selection: Selection | null
  select: (s: Selection | null, fly?: number) => void
  focus: { lon: number; lat: number; zoom?: number; key: number } | null
  flyTo: (lon: number, lat: number, zoom?: number) => void
  fit: { bounds: [number, number, number, number]; key: number } | null
  fitTo: (bounds: [number, number, number, number]) => void
  assistantOpen: boolean
  setAssistantOpen: (o: boolean) => void
  assistantDraft: AssistantDraft | null
  askAssistant: (q: string, entity?: AssistantEntityContext) => void
  clearAssistantDraft: () => void
}

const Ctx = createContext<State | null>(null)

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<Mode>('workspace')
  const [view, setViewState] = useState<View>('overview')
  const [aoi, setAoi] = useState<Polygon | null>(null)
  const [drawMode, setDrawMode] = useState<DrawMode>('none')
  const [jobId, setJobId] = useState<string | null>(null)
  const [flowJobId, setFlowJobId] = useState<string | null>(null)
  const [flowPoint, setFlowPoint] = useState<[number, number] | null>(null)
  const [visible, setVisible] = useState<Visible>(VIEW_LAYERS.overview)
  const [selection, setSelection] = useState<Selection | null>(null)
  const [focus, setFocus] = useState<State['focus']>(null)
  const [fit, setFit] = useState<State['fit']>(null)
  const [assistantOpen, setAssistantOpen] = useState(false)
  const [assistantDraft, setAssistantDraft] = useState<AssistantDraft | null>(null)

  const value = useMemo<State>(
    () => ({
      mode,
      setMode: (m) => {
        setMode(m)
        setDrawMode('none')
      },
      view,
      setView: (v) => {
        setViewState(v)
        setVisible(VIEW_LAYERS[v])
        setDrawMode('none')
        setSelection(null) // a highlight from another view would look like a random selection
      },
      aoi,
      setAoi,
      drawMode,
      setDrawMode,
      jobId,
      openJob: (id, a) => {
        setJobId(id)
        setSelection(null)
        setMode('workspace')
        if (a !== undefined) setAoi(a)
      },
      flowJobId,
      setFlowJobId,
      flowPoint,
      setFlowPoint,
      visible,
      toggle: (id, v) => setVisible((s) => ({ ...s, [id]: v })),
      selection,
      select: (s, zoom) => {
        setSelection(s)
        if (s && zoom) setFocus({ lon: s.lngLat[0], lat: s.lngLat[1], zoom, key: Date.now() })
      },
      focus,
      flyTo: (lon, lat, zoom) => setFocus({ lon, lat, zoom, key: Date.now() }),
      fit,
      fitTo: (bounds) => setFit({ bounds, key: Date.now() }),
      assistantOpen,
      setAssistantOpen,
      assistantDraft,
      askAssistant: (q, entity) => {
        setAssistantDraft({ question: q, entity })
        setAssistantOpen(true)
      },
      clearAssistantDraft: () => setAssistantDraft(null),
    }),
    [mode, view, aoi, drawMode, jobId, flowJobId, flowPoint, visible, selection, focus, fit, assistantOpen, assistantDraft]
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp(): State {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp outside AppStateProvider')
  return v
}
