// Pan and zoom for a fixed-size page (the A4 report): touch pinch, trackpad pinch (ctrl+wheel in
// Chrome/Firefox, gesture events in Safari), drag or wheel to pan, double-click to zoom in or back to
// fit, and +/−/0 keys. The page is real HTML, so text stays sharp at every zoom level.
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState, type ReactNode } from 'react'

type View = { s: number; x: number; y: number }
export type ZoomHandle = { zoomBy: (f: number) => void; fitWidth: () => void; fitPage: () => void }

const MIN = 0.2
const MAX = 6
const PAD = 16

export const ZoomPane = forwardRef<ZoomHandle, { width: number; height: number; children: ReactNode; onScale?: (s: number) => void }>(
  function ZoomPane({ width, height, children, onScale }, ref) {
    const box = useRef<HTMLDivElement>(null)
    const view = useRef<View>({ s: 1, x: 0, y: 0 })
    const fitted = useRef<'width' | 'page' | null>('width')
    const [, render] = useState(0)

    const set = useCallback(
      (v: View) => {
        const el = box.current
        if (el) {
          // Keep at least a strip of the page on screen.
          const keep = 80
          v.x = Math.min(el.clientWidth - keep, Math.max(keep - width * v.s, v.x))
          v.y = Math.min(el.clientHeight - keep, Math.max(keep - height * v.s, v.y))
        }
        view.current = v
        onScale?.(v.s)
        render((n) => n + 1)
      },
      [width, height, onScale],
    )

    const fit = useCallback(
      (mode: 'width' | 'page') => {
        const el = box.current
        if (!el) return
        const W = el.clientWidth
        const H = el.clientHeight
        const s = mode === 'width' ? (W - 2 * PAD) / width : Math.min((W - 2 * PAD) / width, (H - 2 * PAD) / height)
        fitted.current = mode
        set({ s, x: (W - width * s) / 2, y: mode === 'width' ? PAD : (H - height * s) / 2 })
      },
      [width, height, set],
    )

    const zoomAt = useCallback(
      (px: number, py: number, f: number) => {
        const { s, x, y } = view.current
        const ns = Math.min(MAX, Math.max(MIN, s * f))
        fitted.current = null
        set({ s: ns, x: px - (px - x) * (ns / s), y: py - (py - y) * (ns / s) })
      },
      [set],
    )

    const center = () => {
      const el = box.current
      return el ? [el.clientWidth / 2, el.clientHeight / 2] : [0, 0]
    }
    useImperativeHandle(ref, () => ({
      zoomBy: (f) => {
        const [cx, cy] = center()
        zoomAt(cx, cy, f)
      },
      fitWidth: () => fit('width'),
      fitPage: () => fit('page'),
    }))

    // Fit on open, and again on resize while the user has not zoomed by hand.
    useEffect(() => {
      const el = box.current
      if (!el) return
      fit('width')
      const ro = new ResizeObserver(() => fitted.current && fit(fitted.current))
      ro.observe(el)
      return () => ro.disconnect()
    }, [fit])

    // Wheel and Safari gesture listeners must be non-passive to stop the browser zooming the app.
    useEffect(() => {
      const el = box.current
      if (!el) return
      const local = (e: { clientX: number; clientY: number }) => {
        const r = el.getBoundingClientRect()
        return [e.clientX - r.left, e.clientY - r.top]
      }
      const onWheel = (e: WheelEvent) => {
        e.preventDefault()
        if (e.ctrlKey || e.metaKey) {
          const [px, py] = local(e)
          // Trackpad pinch sends small deltas; a mouse wheel tick (~100) is capped to one gentle step.
          zoomAt(px, py, Math.exp(-Math.max(-40, Math.min(40, e.deltaY)) * 0.0075))
        } else {
          const { s, x, y } = view.current
          fitted.current = null
          set({ s, x: x - e.deltaX, y: y - e.deltaY })
        }
      }
      let gestureScale = 1
      const onGestureStart = (e: Event) => {
        e.preventDefault()
        gestureScale = 1
      }
      const onGestureChange = (e: Event) => {
        e.preventDefault()
        const g = e as Event & { scale: number; clientX: number; clientY: number }
        const [px, py] = local(g)
        zoomAt(px, py, g.scale / gestureScale)
        gestureScale = g.scale
      }
      el.addEventListener('wheel', onWheel, { passive: false })
      el.addEventListener('gesturestart', onGestureStart)
      el.addEventListener('gesturechange', onGestureChange)
      return () => {
        el.removeEventListener('wheel', onWheel)
        el.removeEventListener('gesturestart', onGestureStart)
        el.removeEventListener('gesturechange', onGestureChange)
      }
    }, [zoomAt, set])

    // Pointers: one drags, two pinch. A drag suppresses the click that would follow it (links).
    const pts = useRef(new Map<number, { x: number; y: number }>())
    const moved = useRef(0)
    const onPointerDown = (e: React.PointerEvent) => {
      if (e.pointerType === 'mouse' && e.button !== 0) return
      try {
        box.current?.setPointerCapture(e.pointerId)
      } catch {
        /* pointer already released */
      }
      pts.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
      if (pts.current.size === 1) moved.current = 0
    }
    const onPointerMove = (e: React.PointerEvent) => {
      const prev = pts.current.get(e.pointerId)
      if (!prev) return
      const el = box.current!
      const r = el.getBoundingClientRect()
      if (pts.current.size === 1) {
        const dx = e.clientX - prev.x
        const dy = e.clientY - prev.y
        moved.current += Math.abs(dx) + Math.abs(dy)
        const { s, x, y } = view.current
        fitted.current = null
        set({ s, x: x + dx, y: y + dy })
      } else if (pts.current.size === 2) {
        const other = [...pts.current.entries()].find(([id]) => id !== e.pointerId)![1]
        const d0 = Math.hypot(prev.x - other.x, prev.y - other.y)
        const d1 = Math.hypot(e.clientX - other.x, e.clientY - other.y)
        const mx0 = (prev.x + other.x) / 2 - r.left
        const my0 = (prev.y + other.y) / 2 - r.top
        const mx1 = (e.clientX + other.x) / 2 - r.left
        const my1 = (e.clientY + other.y) / 2 - r.top
        moved.current += 10
        if (d0 > 0) zoomAt(mx0, my0, d1 / d0)
        const { s, x, y } = view.current
        set({ s, x: x + (mx1 - mx0), y: y + (my1 - my0) })
      }
      pts.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    }
    const onPointerUp = (e: React.PointerEvent) => {
      pts.current.delete(e.pointerId)
    }
    const onDoubleClick = (e: React.MouseEvent) => {
      const r = box.current!.getBoundingClientRect()
      if (fitted.current) zoomAt(e.clientX - r.left, e.clientY - r.top, 2)
      else fit('width')
    }
    const onKeyDown = (e: React.KeyboardEvent) => {
      const [cx, cy] = center()
      if (e.key === '+' || e.key === '=') zoomAt(cx, cy, 1.25)
      else if (e.key === '-') zoomAt(cx, cy, 0.8)
      else if (e.key === '0') fit('width')
      else return
      e.preventDefault()
    }

    const { s, x, y } = view.current
    return (
      <div
        ref={box}
        tabIndex={0}
        role='region'
        aria-label='Report page. Pinch, scroll with Ctrl, or press + and − to zoom; drag to move.'
        className='sr-zoom relative min-h-0 flex-1 cursor-grab touch-none overflow-hidden bg-muted/50 outline-none select-none active:cursor-grabbing'
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={onDoubleClick}
        onKeyDown={onKeyDown}
        onClickCapture={(e) => {
          if (moved.current > 6) {
            e.preventDefault()
            e.stopPropagation()
          }
        }}
      >
        <div
          className='sr-zoom-inner absolute top-0 left-0 origin-top-left shadow-lg'
          style={{ width, transform: `translate(${x}px, ${y}px) scale(${s})` }}
        >
          {children}
        </div>
      </div>
    )
  },
)
