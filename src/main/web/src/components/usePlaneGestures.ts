// The spectrogram's and heatmap's canvas gestures (design/web-parity.md WP2): drag a box to zoom to it (shown while
// dragging, with the range it would pick), the wheel to zoom about the pointer, a double click for the whole title, Escape
// to drop a drag, and where the pointer is, for a readout.
import { useEffect, useRef, useState, type RefObject } from 'react'

import { hzAt, inside, ordered, timeAt, type Axes } from '../chart/plane'
import type { Range } from '../chart/view'

export interface Selection {
  /** The box in canvas pixels. */
  left: number
  top: number
  width: number
  height: number
  x: Range
  t: Range
}

export interface Hover {
  pane: number
  px: number
  py: number
  hz: number
  seconds: number
}

interface Options {
  canvas: RefObject<HTMLCanvasElement | null>
  /** Each plot on the canvas; a drag stays in the one it starts in. */
  panes: Axes[]
  onSelect: (x: Range, t: Range) => void
  /** Zoom by `factor` (under 1 is in) about this point. */
  onWheel: (factor: number, hz: number, seconds: number) => void
  onReset: () => void
}

const LEAST_PX = 4   // a box thinner than this on one side keeps that side's range

function point(canvas: HTMLCanvasElement, event: { clientX: number; clientY: number }) {
  const box = canvas.getBoundingClientRect()
  const sx = box.width ? canvas.width / box.width : 1, sy = box.height ? canvas.height / box.height : 1
  return { px: (event.clientX - box.left) * sx, py: (event.clientY - box.top) * sy }
}

export function usePlaneGestures({ canvas, panes, onSelect, onWheel, onReset }: Options) {
  const [selection, setSelection] = useState<Selection | null>(null)
  const [hover, setHover] = useState<Hover | null>(null)
  const start = useRef<{ pane: number; px: number; py: number } | null>(null)
  const latest = useRef({ panes, onWheel })
  useEffect(() => {
    latest.current = { panes, onWheel }
  })

  const paneAt = (px: number, py: number) => latest.current.panes.findIndex((a) => inside(a, px, py))

  function boxFrom(from: { pane: number; px: number; py: number }, px: number, py: number): Selection {
    const axes = panes[from.pane]!
    const clampX = (v: number) => Math.min(Math.max(v, axes.left), axes.left + axes.width)
    const clampY = (v: number) => Math.min(Math.max(v, axes.top), axes.top + axes.height)
    const [l, r] = ordered(clampX(from.px), clampX(px)), [tp, b] = ordered(clampY(from.py), clampY(py))
    const wide = r - l >= LEAST_PX, tall = b - tp >= LEAST_PX
    return {
      left: wide ? l : axes.left, width: wide ? r - l : axes.width, top: tall ? tp : axes.top, height: tall ? b - tp : axes.height,
      x: wide ? [hzAt(axes, l), hzAt(axes, r)] : axes.x, t: tall ? [timeAt(axes, tp), timeAt(axes, b)] : axes.t,
    }
  }

  useEffect(() => {
    const element = canvas.current
    if (!element) return
    const wheel = (event: WheelEvent) => {
      const { px, py } = point(element, event)
      const pane = paneAt(px, py)
      if (pane < 0) return
      event.preventDefault()
      const axes = latest.current.panes[pane]!
      latest.current.onWheel(event.deltaY < 0 ? 0.8 : 1.25, hzAt(axes, px), timeAt(axes, py))
    }
    element.addEventListener('wheel', wheel, { passive: false })
    return () => element.removeEventListener('wheel', wheel)
  })

  const handlers = {
    onPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
      if (event.button !== 0) return
      const { px, py } = point(event.currentTarget, event)
      const pane = paneAt(px, py)
      if (pane < 0) return
      start.current = { pane, px, py }
      event.currentTarget.setPointerCapture?.(event.pointerId)
    },
    onPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
      const { px, py } = point(event.currentTarget, event)
      const from = start.current
      if (from) {
        const moved = Math.max(Math.abs(px - from.px), Math.abs(py - from.py)) >= LEAST_PX
        setSelection(moved ? boxFrom(from, px, py) : null)
      }
      const pane = from ? from.pane : paneAt(px, py)
      if (pane < 0) {
        setHover(null)
        return
      }
      const axes = panes[pane]!
      setHover({ pane, px, py, hz: hzAt(axes, px), seconds: timeAt(axes, py) })
    },
    onPointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
      const from = start.current
      start.current = null
      setSelection(null)
      if (!from) return
      const { px, py } = point(event.currentTarget, event)
      if (Math.max(Math.abs(px - from.px), Math.abs(py - from.py)) < LEAST_PX) return
      const box = boxFrom(from, px, py)
      if (box.x[0] < box.x[1] && box.t[0] < box.t[1]) onSelect(box.x, box.t)
    },
    onPointerLeave() {
      if (!start.current) setHover(null)
    },
    onKeyDown(event: React.KeyboardEvent<HTMLCanvasElement>) {
      if (event.key === 'Escape') {
        start.current = null
        setSelection(null)
      }
    },
    onDoubleClick() {
      start.current = null
      setSelection(null)
      onReset()
    },
  }
  return { handlers, selection, hover }
}
