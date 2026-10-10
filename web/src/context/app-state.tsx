import { createContext, useContext, useMemo, useState } from 'react'
import type { Polygon } from '@/lib/api'

export type DrawMode = 'none' | 'box' | 'polygon' | 'flowpoint'
export type Section = 'analyze' | 'results' | 'flowpath' | 'jobs' | 'about'

export const LAYERS = [
  { id: 'debris', label: 'Debris / scour', group: 'Hazard', default: true, color: '#c27803' },
  { id: 'flood', label: 'New water', group: 'Hazard', default: true, color: '#1d4ed8' },
  { id: 'water', label: 'Water before event', group: 'Hazard', default: false, color: '#38bdf8' },
  { id: 'unobs', label: 'Not observable', group: 'Hazard', default: true, color: '#78716c' },
  { id: 'confidence', label: 'Confidence tiers', group: 'Hazard', default: false, color: '#b91c1c' },
  { id: 'buildings', label: 'Buildings hit', group: 'Damage', default: true, color: '#b45309' },
  { id: 'roads', label: 'Roads & trails hit', group: 'Damage', default: true, color: '#7c2d12' },
  { id: 'bridges', label: 'Bridges', group: 'Damage', default: true, color: '#0f172a' },
  { id: 'failed', label: 'Failed network edges', group: 'Isolation', default: true, color: '#dc2626' },
  { id: 'settlements', label: 'Settlements', group: 'Isolation', default: true, color: '#475569' },
  { id: 'flowpath', label: 'Flow path', group: 'Bonus', default: true, color: '#7c3aed' },
] as const
export type LayerId = (typeof LAYERS)[number]['id']

type State = {
  section: Section
  setSection: (s: Section) => void
  aoi: Polygon | null
  setAoi: (p: Polygon | null) => void
  drawMode: DrawMode
  setDrawMode: (m: DrawMode) => void
  jobId: string | null
  setJobId: (id: string | null) => void
  flowJobId: string | null
  setFlowJobId: (id: string | null) => void
  flowPoint: [number, number] | null
  setFlowPoint: (p: [number, number] | null) => void
  visible: Record<LayerId, boolean>
  toggle: (id: LayerId, on: boolean) => void
  focus: { lon: number; lat: number; zoom?: number; key: number } | null
  flyTo: (lon: number, lat: number, zoom?: number) => void
}

const Ctx = createContext<State | null>(null)

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [section, setSection] = useState<Section>('analyze')
  const [aoi, setAoi] = useState<Polygon | null>(null)
  const [drawMode, setDrawMode] = useState<DrawMode>('none')
  const [jobId, setJobId] = useState<string | null>(null)
  const [flowJobId, setFlowJobId] = useState<string | null>(null)
  const [flowPoint, setFlowPoint] = useState<[number, number] | null>(null)
  const [visible, setVisible] = useState(
    () => Object.fromEntries(LAYERS.map((l) => [l.id, l.default])) as Record<LayerId, boolean>
  )
  const [focus, setFocus] = useState<State['focus']>(null)
  const value = useMemo<State>(
    () => ({
      section,
      setSection,
      aoi,
      setAoi,
      drawMode,
      setDrawMode,
      jobId,
      setJobId,
      flowJobId,
      setFlowJobId,
      flowPoint,
      setFlowPoint,
      visible,
      toggle: (id, on) => setVisible((v) => ({ ...v, [id]: on })),
      focus,
      flyTo: (lon, lat, zoom) => setFocus({ lon, lat, zoom, key: Date.now() }),
    }),
    [section, aoi, drawMode, jobId, flowJobId, flowPoint, visible, focus]
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp(): State {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp outside AppStateProvider')
  return v
}
