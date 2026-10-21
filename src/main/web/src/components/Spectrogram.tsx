// A title's spectrogram (design/web-parity.md WP2): frequency across -- the chart's range and log axis, so zooming one
// zooms the other -- and time down, as measured, after the design, or the difference; with the loudest level of each
// frequency in a strip under it ("where are the heavy hits"). Drag across to zoom the frequencies, down to zoom the
// times; a double click goes back to the whole title and the chart's preset range.
import { useEffect, useRef, useState } from 'react'

import { useSpectrogram } from '../api/hooks'
import { clock, decodeGrid, differenceColour, gridTimes, levelColour, loudestByFrequency, paint, type Grid,
  type GridMode } from '../chart/grid'
import { drawnRange, logTicks, type ChartView, type Range } from '../chart/view'

const MAX_HZ = 500
const SPAN_DB = 20
const LEVELS_DB = 60   // the scale's span below the loudest cell
const HEIGHT = 360
const STRIP = 28
const MARGIN = { left: 64, bottom: 28 }

const MODES: { mode: GridMode; label: string }[] = [
  { mode: 'before', label: 'As measured' }, { mode: 'after', label: 'After the design' },
  { mode: 'difference', label: 'Difference' },
]

function linearTicks([low, high]: Range, count = 6): number[] {
  const step = 10 ** Math.floor(Math.log10((high - low) / count))
  const nice = [1, 2, 5, 10].map((m) => m * step).find((s) => (high - low) / s <= count) ?? step * 10
  const ticks = []
  for (let v = Math.ceil(low / nice) * nice; v <= high; v += nice) ticks.push(Number(v.toPrecision(6)))
  return ticks
}

interface Props {
  titleId: string
  candidate: number
  view: ChartView
  onZoom: (x: Range) => void
  onReset: () => void
}

export function Spectrogram({ titleId, candidate, view, onZoom, onReset }: Props) {
  const [mode, setMode] = useState<GridMode>('before')
  const [times, setTimes] = useState<Range | null>(null)
  const before = useSpectrogram(titleId, candidate, false, MAX_HZ, true)
  const after = useSpectrogram(titleId, candidate, true, MAX_HZ, mode !== 'before')
  const canvas = useRef<HTMLCanvasElement>(null)
  const strip = useRef<HTMLCanvasElement>(null)
  const drag = useRef<{ x: number; y: number } | null>(null)
  const [width, setWidth] = useState(800)
  const holder = useRef<HTMLDivElement>(null)

  const decoded = (data: typeof before.data): Grid | null => (data && !data.pending ? decodeGrid(data.data) : null)
  const grid = decoded(before.data)
  const afterGrid = decoded(after.data)
  const ready = grid !== null && (mode === 'before' || afterGrid !== null)
  const x = drawnRange(view)
  const t = times ?? (grid ? gridTimes(grid) : [0, 1] as Range)
  const top = grid ? grid.floorDb + 254 * grid.stepDb : 0
  const levels: Range = [top - LEVELS_DB, top]

  useEffect(() => {
    const element = holder.current
    if (!element) return
    const observe = new ResizeObserver(() => setWidth(Math.max(320, element.clientWidth || 800)))
    observe.observe(element)
    return () => observe.disconnect()
  }, [])

  useEffect(() => {
    const context = canvas.current?.getContext('2d')
    if (!context || !grid || !ready) return
    const plotWidth = width - MARGIN.left, plotHeight = HEIGHT - MARGIN.bottom
    context.clearRect(0, 0, width, HEIGHT)
    const pixels = paint(grid, afterGrid, mode, { width: plotWidth, height: plotHeight, x, t, logX: view.logX, levels,
                                                  span: SPAN_DB })
    context.putImageData(new ImageData(pixels, plotWidth, plotHeight), MARGIN.left, 0)
    const ink = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() || '#888'
    context.fillStyle = ink
    context.font = '11px system-ui, sans-serif'
    const across = (hz: number) => MARGIN.left + plotWidth * (view.logX
      ? (Math.log10(hz) - Math.log10(x[0])) / (Math.log10(x[1]) - Math.log10(x[0])) : (hz - x[0]) / (x[1] - x[0]))
    for (const hz of view.logX ? logTicks(x) : linearTicks(x)) context.fillText(String(hz), across(hz) - 6, HEIGHT - 10)
    for (const s of linearTicks(t, 5)) {
      context.fillText(clock(s), 2, (plotHeight * (s - t[0])) / (t[1] - t[0]) + 4)
    }
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
  })

  function released(event: React.MouseEvent<HTMLCanvasElement>) {
    const start = drag.current
    drag.current = null
    if (!start) return
    const box = event.currentTarget.getBoundingClientRect()
    const plotWidth = width - MARGIN.left, plotHeight = HEIGHT - MARGIN.bottom
    const dx = event.clientX - box.left - start.x, dy = event.clientY - box.top - start.y
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 4) return
    if (Math.abs(dx) >= Math.abs(dy)) {
      const at = (px: number) => {
        const u = Math.min(Math.max((px - MARGIN.left) / plotWidth, 0), 1)
        return view.logX ? 10 ** (Math.log10(x[0]) + (Math.log10(x[1]) - Math.log10(x[0])) * u) : x[0] + (x[1] - x[0]) * u
      }
      const [a, b] = [at(start.x), at(start.x + dx)].sort((p, q) => p - q) as Range
      if (a < b) onZoom([a, b])
    } else {
      const at = (py: number) => t[0] + ((t[1] - t[0]) * Math.min(Math.max(py, 0), plotHeight)) / plotHeight
      const [a, b] = [at(start.y), at(start.y + dy)].sort((p, q) => p - q) as Range
      if (a < b) setTimes([a, b])
    }
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
        <button type="button" className="secondary small" disabled={!times} onClick={() => setTimes(null)}>Whole title</button>
        <span className="muted">
          {mode === 'difference' ? `Blue quieter, red louder after the design (±${SPAN_DB} dB)`
            : `${Math.round(levels[0])} to ${Math.round(levels[1])} dB`}
          {times && ` · ${clock(t[0])}–${clock(t[1])}`}
        </span>
      </div>
      {failed && <p role="alert" className="problem">{failed.message}</p>}
      {!failed && waiting && <p className="muted" role="status">Drawing the spectrogram… (the service renders it once, then keeps it)</p>}
      {!failed && ready && (
        <>
          <canvas ref={canvas} width={width} height={HEIGHT} role="img"
                  aria-label={`Spectrogram, ${MODES.find((m) => m.mode === mode)!.label.toLowerCase()}`}
                  onMouseDown={(e) => {
                    const box = e.currentTarget.getBoundingClientRect()
                    drag.current = { x: e.clientX - box.left, y: e.clientY - box.top }
                  }}
                  onMouseUp={released}
                  onDoubleClick={() => {
                    setTimes(null)
                    onReset()
                  }} />
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
