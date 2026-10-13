import type { LAYERS as L, LayerId } from '@/context/app-state'
import { STATUS_COLOR, STATUS_LABEL } from '@/lib/job'

/** Shows only what is on the map right now. */
export function Legend({ visible, route, layers }: { visible: Record<LayerId, boolean>; route: boolean; layers: typeof L }) {
  const areas = layers.filter((l) => l.group === 'Hazard' && visible[l.id] && l.id !== 'confidence')
  const showSettlements = visible.settlements
  const showBridges = visible.bridges
  if (!areas.length && !showSettlements && !showBridges && !route && !visible.failed && !visible.flowpath) return null
  return (
    <div className='max-w-56 space-y-2 rounded-lg border bg-background/95 p-2.5 text-[0.6875rem] shadow-sm'>
      {areas.length > 0 && (
        <div className='space-y-1'>
          {areas.map((l) => (
            <Item key={l.id} swatch={<span className='size-2.5 rounded-[0.1875rem]' style={{ background: l.color }} />}>
              {l.label}
            </Item>
          ))}
        </div>
      )}
      {(visible.failed || showBridges) && (
        <div className='space-y-1'>
          {visible.failed && <Item swatch={<span className='h-0.5 w-3 rounded bg-red-600' />}>Impassable road</Item>}
          {showBridges && (
            <Item swatch={<span className='h-1 w-3 rounded' style={{ background: STATUS_COLOR.likely_lost }} />}>Bridge likely lost</Item>
          )}
        </div>
      )}
      {showSettlements && (
        <div className='space-y-1'>
          {(['cut_off', 'hospital_cut_off', 'detour', 'connected'] as const).map((s) => (
            <Item key={s} swatch={<span className='size-2.5 rounded-full' style={{ background: STATUS_COLOR[s] }} />}>
              {STATUS_LABEL[s]}
            </Item>
          ))}
        </div>
      )}
      {route && (
        <div className='space-y-1'>
          <Item swatch={<span className='h-0 w-3 border-t-2 border-dashed border-slate-500' />}>Route before</Item>
          <Item swatch={<span className='h-1 w-3 rounded bg-blue-600' />}>Route after</Item>
          <Item swatch={<span className='size-2.5 rounded-full border-2 border-white bg-red-600' />}>Route broken</Item>
        </div>
      )}
      {visible.flowpath && <Item swatch={<span className='h-1 w-3 rounded bg-violet-600' />}>Flood path</Item>}
    </div>
  )
}

function Item({ swatch, children }: { swatch: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className='flex items-center gap-2'>
      <span className='flex w-3 items-center justify-center'>{swatch}</span>
      <span>{children}</span>
    </div>
  )
}
