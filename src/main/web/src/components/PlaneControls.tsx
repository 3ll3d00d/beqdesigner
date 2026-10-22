// Navigating the spectrogram's and heatmap's frequency x time plane (design/web-parity.md WP2): the ranges shown, typed in
// exactly, how far in that is, zoom in and out about the middle, back to the view before, and the whole title. The drag,
// wheel and hover on the canvas itself are usePlaneGestures'.
import { useState } from 'react'

import { clock } from '../chart/grid'
import { clampRange, magnification, parseClock, zoomAbout } from '../chart/plane'
import type { Range } from '../chart/view'

export interface Plane {
  x: Range
  t: Range
}

/** Where the view has been, so *Back* can return to it. */
export function useZoomHistory(current: Plane, apply: (plane: Plane) => void) {
  const [stack, setStack] = useState<Plane[]>([])
  return {
    canBack: stack.length > 0,
    go(next: Plane) {
      if (next.x[0] === current.x[0] && next.x[1] === current.x[1] && next.t[0] === current.t[0] && next.t[1] === current.t[1]) return
      setStack((s) => [...s.slice(-49), current])
      apply(next)
    },
    back() {
      const previous = stack[stack.length - 1]
      if (!previous) return
      setStack((s) => s.slice(0, -1))
      apply(previous)
    },
    clear: () => setStack([]),
  }
}

const hz = (v: number) => String(Number(v.toPrecision(4)))

/** A typed number or time, applied on Enter or leaving the box, refused (and marked) when it does not read. */
function Bound({ label, value, show, read, apply }: {
  label: string; value: number; show: (v: number) => string; read: (text: string) => number | null
  apply: (v: number) => boolean
}) {
  const [text, setText] = useState(show(value))   // the parent keys this by its value, so a new value starts afresh
  const [bad, setBad] = useState(false)
  const commit = () => {
    const v = read(text)
    if (v === null || !apply(v)) {
      setBad(text.trim() !== show(value))
      if (v !== null) setText(show(value))
      return
    }
    setBad(false)
  }
  return (
    <input aria-label={label} className="bound" value={text} aria-invalid={bad || undefined}
           onChange={(e) => setText(e.target.value)} onBlur={commit}
           onKeyDown={(e) => {
             if (e.key === 'Enter') commit()
             if (e.key === 'Escape') { setText(show(value)); setBad(false) }
           }} />
  )
}

interface Props {
  plane: Plane
  full: Plane
  logX: boolean
  canBack: boolean
  onChange: (plane: Plane) => void
  onBack: () => void
  onReset: () => void
  /** Names the group, e.g. "Spectrogram view". */
  label: string
}

const LEAST_HZ = 0.5, LEAST_SECONDS = 1

export function PlaneControls({ plane, full, logX, canBack, onChange, onBack, onReset, label }: Props) {
  const { x, t } = plane
  const setX = (next: Range) => onChange({ x: clampRange(next, full.x, LEAST_HZ), t })
  const setT = (next: Range) => onChange({ x, t: clampRange(next, full.t, LEAST_SECONDS) })
  const number = (text: string) => (/^\s*\d+(\.\d+)?\s*$/.test(text) ? Number(text) : null)
  const zoom = (factor: number) => onChange({ x: zoomAbout(x, factor, full.x, logX), t: zoomAbout(t, factor, full.t) })
  const xIn = magnification(x, full.x), tIn = magnification(t, full.t)
  const whole = xIn < 1.0001 && tIn < 1.0001
  return (
    <div className="plane-controls" role="group" aria-label={label}>
      <span className="bounds">
        <span className="muted">Frequency</span>
        <Bound key={x[0]} label="Lowest frequency (Hz)" value={x[0]} show={hz} read={number}
               apply={(v) => v < x[1] && (setX([v, x[1]]), true)} />
        –
        <Bound key={x[1]} label="Highest frequency (Hz)" value={x[1]} show={hz} read={number}
               apply={(v) => v > x[0] && (setX([x[0], v]), true)} /> Hz
      </span>
      <span className="bounds">
        <span className="muted">Time</span>
        <Bound key={t[0]} label="Start time" value={t[0]} show={clock} read={parseClock}
               apply={(v) => v < t[1] && (setT([v, t[1]]), true)} />
        –
        <Bound key={t[1]} label="End time" value={t[1]} show={clock} read={parseClock}
               apply={(v) => v > t[0] && (setT([t[0], v]), true)} />
      </span>
      <span className="buttons">
        <button type="button" className="secondary small" onClick={() => zoom(0.5)} aria-label="Zoom in">＋</button>
        <button type="button" className="secondary small" onClick={() => zoom(2)} disabled={whole} aria-label="Zoom out">－</button>
        <button type="button" className="secondary small" onClick={onBack} disabled={!canBack}>Back</button>
        <button type="button" className="secondary small" onClick={onReset} disabled={whole && !canBack}>Whole title</button>
      </span>
      <span className="muted zoom-level" aria-live="polite">
        {whole ? 'Whole range' : `Zoomed ×${xIn.toFixed(1)} frequency, ×${tIn.toFixed(1)} time`}
      </span>
    </div>
  )
}
