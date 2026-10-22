// A title's spectrogram (design/web-parity.md WP2): frequency across -- the chart's range and log axis, so zooming one
// zooms the other -- and time down, as measured, after the design, or the difference; with the loudest level of each
// frequency in a strip under it ("where are the heavy hits"). The ranges are typed in, or zoomed by dragging a box (shown
// as it is dragged), the wheel about the pointer, or the zoom buttons; Back returns to the view before, and *Whole title*
// (or a double click) to the whole title and the chart's preset range.
import { useEffect, useRef, useState } from 'react'

import { useSpectrogram } from '../api/hooks'
import { clock, decodeGrid, differenceColour, differenceGradient, gridTimes, levelAt, levelColour, loudestByFrequency,
  paint, scaleGradient, type Grid, type GridMode } from '../chart/grid'
import { pxOf, pyOf, zoomAbout, type Axes } from '../chart/plane'
import { drawnRange, logTicks, type ChartView, type Range } from '../chart/view'
import { cssColour, useColourScheme } from '../theme'
import { PlaneControls, useZoomHistory, type Plane } from './PlaneControls'
import { SelectionBox } from './SelectionBox'
import { usePlaneGestures } from './usePlaneGestures'

export const MAX_HZ = 500
const SPAN_DB = 20
export const LEVELS_DB = 60   // the scale's span below the loudest cell
const HEIGHT = 360
const STRIP = 28
const MARGIN = { left: 64, bottom: 28 }

const MODES: { mode: GridMode; label: string }[] = [
  { mode: 'before', label: 'As measured' }, { mode: 'after', label: 'After the design' },
  { mode: 'difference', label: 'Difference' },
]

export function linearTicks([low, high]: Range, count = 6): number[] {
  const step = 10 ** Math.floor(Math.log10((high - low) / count))
  const nice = [1, 2, 5, 10].map((m) => m * step).find((s) => (high - low) / s <= count) ?? step * 10
  const ticks = []
  for (let v = Math.ceil(low / nice) * nice; v <= high; v += nice) ticks.push(Number(v.toPrecision(6)))
  return ticks
}

/** The loudest level a grid holds, dB: the top of the scale. */
export const topOf = (grid: Grid) => grid.floorDb + 254 * grid.stepDb

/** The width a canvas inside `holder` can take, followed as it resizes. */
export function useWidth(holder: React.RefObject<HTMLDivElement | null>, initial = 800) {
  const [width, setWidth] = useState(initial)
  useEffect(() => {
    const element = holder.current
    if (!element) return
    const observe = new ResizeObserver(() => setWidth(Math.max(320, element.clientWidth || initial)))
    observe.observe(element)
    return () => observe.disconnect()
  }, [holder, initial])
  return width
}

/** The level scale's colour bar, quietest on the left. */
export function ColourBar({ low, high, label, gradient = scaleGradient() }: { low: string; high: string; label: string
                                                                           gradient?: string }) {
  return (
    <span className="colour-bar" role="img" aria-label={label}>
      <span className="muted">{low}</span><span className="bar" style={{ background: gradient }} /><span className="muted">{high}</span>
    </span>
  )
}

interface Props {
  titleId: string
  candidate: number
  /** The design run `candidate` is of; null: the current one. */
  variant?: string | null
  view: ChartView
  onZoom: (x: Range) => void
  onReset: () => void
}

export function Spectrogram({ titleId, candidate, variant = null, view, onZoom, onReset }: Props) {
  const [mode, setMode] = useState<GridMode>('before')
  const [times, setTimes] = useState<Range | null>(null)
  const before = useSpectrogram(titleId, candidate, false, MAX_HZ, true, variant)
  const after = useSpectrogram(titleId, candidate, true, MAX_HZ, mode !== 'before', variant)
  const canvas = useRef<HTMLCanvasElement>(null)
  const strip = useRef<HTMLCanvasElement>(null)
  const holder = useRef<HTMLDivElement>(null)
  const width = useWidth(holder)
  const scheme = useColourScheme()

  const decoded = (data: typeof before.data): Grid | null => (data && !data.pending ? decodeGrid(data.data) : null)
  const grid = decoded(before.data)
  const afterGrid = decoded(after.data)
  const ready = grid !== null && (mode === 'before' || afterGrid !== null)
  const x = drawnRange(view)
  const fullT = grid ? gridTimes(grid) : [0, 1] as Range
  const t = times ?? fullT
  const full: Plane = { x: [view.logX ? 1 : 0, grid ? Math.min(MAX_HZ, grid.f0 + grid.df * (grid.freqs - 1)) : MAX_HZ], t: fullT }
  const top = grid ? topOf(grid) : 0
  const levels: Range = [top - LEVELS_DB, top]
  const plotWidth = width - MARGIN.left, plotHeight = HEIGHT - MARGIN.bottom
  const axes: Axes = { left: MARGIN.left, top: 0, width: plotWidth, height: plotHeight, x, t, logX: view.logX }

  const apply = (plane: Plane) => {
    if (plane.x[0] !== x[0] || plane.x[1] !== x[1]) onZoom(plane.x)
    const whole = plane.t[0] <= fullT[0] && plane.t[1] >= fullT[1]
    setTimes(whole ? null : plane.t)
  }
  const history = useZoomHistory({ x, t }, apply)
  const reset = () => {
    history.clear()
    setTimes(null)
    onReset()
  }
  const { handlers, selection, hover } = usePlaneGestures({
    canvas, panes: [axes],
    onSelect: (sx, st) => history.go({ x: sx, t: st }),
    onWheel: (factor, hz, seconds) => history.go({ x: zoomAbout(x, factor, full.x, view.logX, hz),
                                                   t: zoomAbout(t, factor, full.t, false, seconds) }),
    onReset: reset,
  })

  useEffect(() => {
    const context = canvas.current?.getContext('2d')
    if (!context || !grid || !ready) return
    context.clearRect(0, 0, width, HEIGHT)
    const pixels = paint(grid, afterGrid, mode, { width: plotWidth, height: plotHeight, x, t, logX: view.logX, levels,
                                                  span: SPAN_DB })
    context.putImageData(new ImageData(pixels, plotWidth, plotHeight), MARGIN.left, 0)
    context.strokeStyle = cssColour('line', '#ccc')
    context.strokeRect(MARGIN.left + 0.5, 0.5, plotWidth - 1, plotHeight - 1)
    context.fillStyle = cssColour('muted', '#888')
    context.font = '11px system-ui, sans-serif'
    for (const hz of view.logX ? logTicks(x) : linearTicks(x)) context.fillText(String(hz), pxOf(axes, hz) - 6, HEIGHT - 10)
    for (const s of linearTicks(t, 5)) context.fillText(clock(s), 2, pyOf(axes, s) + 4)
    const stripContext = strip.current?.getContext('2d')
    if (stripContext) {
      stripContext.clearRect(0, 0, width, STRIP)
      const shown = mode === 'after' && afterGrid ? afterGrid : grid
      const loudest = loudestByFrequency(shown, t)
      for (let px = 0; px < plotWidth; px++) {
        const along = view.logX ? 10 ** (Math.log10(x[0]) + ((Math.log10(x[1]) - Math.log10(x[0])) * px) / plotWidth)
          : x[0] + ((x[1] - x[0]) * px) / plotWidth
        const db = loudest[Math.round((along - shown.f0) / shown.df)]
        if (db === undefined || db === null) continue
        const [r, g, b] = levelColour(db, levels)
        stripContext.fillStyle = `rgb(${r},${g},${b})`
        stripContext.fillRect(MARGIN.left + px, 0, 1, STRIP)
      }
    }
  }, [grid, afterGrid, ready, mode, width, plotWidth, plotHeight, x[0], x[1], t[0], t[1], view.logX, top, scheme])   // eslint-disable-line react-hooks/exhaustive-deps

  function readout() {
    if (!hover || !grid) return null
    const shown = mode === 'after' && afterGrid ? afterGrid : grid
    const db = mode === 'difference'
      ? (() => {
          const a = afterGrid ? levelAt(afterGrid, hover.seconds, hover.hz) : null, b = levelAt(grid, hover.seconds, hover.hz)
          return a === null && b === null ? null : (a ?? grid.floorDb) - (b ?? grid.floorDb)
        })()
      : levelAt(shown, hover.seconds, hover.hz)
    return `${hover.hz.toFixed(1)} Hz · ${clock(hover.seconds)} · ${db === null ? 'below the scale'
      : `${mode === 'difference' && db > 0 ? '+' : ''}${db.toFixed(1)} dB`}`
  }

  const failed = before.error ?? after.error
  const waiting = before.data?.pending || (mode !== 'before' && after.data?.pending)
  return (
    <div className="spectrogram" ref={holder}>
      <div className="row" role="group" aria-label="Spectrogram">
        {MODES.map((m) => (
          <label key={m.mode} className="inline">
            <input type="radio" name="spectrogram-mode" checked={mode === m.mode} onChange={() => setMode(m.mode)} /> {m.label}
          </label>
        ))}
        {mode === 'difference'
          ? <ColourBar low={`−${SPAN_DB} dB`} high={`+${SPAN_DB} dB`} gradient={differenceGradient()}
                       label={`Blue quieter, red louder after the design (±${SPAN_DB} dB)`} />
          : <ColourBar low={`${Math.round(levels[0])} dB`} high={`${Math.round(levels[1])} dB`}
                       label={`${Math.round(levels[0])} to ${Math.round(levels[1])} dB`} />}
      </div>
      {grid && <PlaneControls label="Spectrogram view" plane={{ x, t }} full={full} logX={view.logX}
                              canBack={history.canBack} onChange={history.go} onBack={history.back} onReset={reset} />}
      {failed && <p role="alert" className="problem">{failed.message}</p>}
      {!failed && waiting && <p className="muted" role="status">Drawing the spectrogram… (the service renders it once, then keeps it)</p>}
      {!failed && ready && (
        <>
          <div className="plane">
            <canvas ref={canvas} width={width} height={HEIGHT} role="img" tabIndex={0}
                    aria-label={`Spectrogram, ${MODES.find((m) => m.mode === mode)!.label.toLowerCase()}`} {...handlers} />
            {selection && <SelectionBox selection={selection} width={width} height={HEIGHT} />}
          </div>
          <p className="muted readout" aria-live="off">{readout() ?? 'Drag a box to zoom to it, scroll to zoom about the pointer, double-click for the whole title.'}</p>
          <canvas ref={strip} width={width} height={STRIP} aria-label="The loudest level of each frequency" role="img" />
          {mode === 'difference' && (
            <p className="muted legend">{[-SPAN_DB, 0, SPAN_DB].map((db) => {
              const [r, g, b] = differenceColour(db, SPAN_DB)
              return <span key={db}><span className="swatch" style={{ background: `rgb(${r},${g},${b})` }} /> {db} dB </span>
            })}</p>
          )}
        </>
      )}
    </div>
  )
}
