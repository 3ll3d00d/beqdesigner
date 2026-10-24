// The review chart (design/web-app.md §4.1, design/web-parity.md WP1): a title's measured average and peak curves, and the
// same after one design or several, drawn as the view says -- its limits, a log frequency axis, which curves, normalised to
// one -- with each design's own response on a second axis when asked. Drag across it to zoom to a frequency range, the
// wheel to zoom about the cursor, a double click to go back to the view's preset. uPlot stays behind this one component,
// so its 1.7 release changes only this file.
import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

import { chartData, drawnRange, logTicks, normalise, type ChartView, type Range, type Series, visibleSeries, zoomRange }
  from '../chart/view'
import { useColourScheme } from '../theme'

export { chartData, FREQUENCY_RANGE, resample } from '../chart/view'

const MEASURED = { average: '#2b8cbe', peak: '#e6550d' }
/** One colour per design when several are drawn (the measured curves keep theirs). */
export const DESIGN_COLOURS = ['#31a354', '#756bb1', '#d6616b', '#8c6d31', '#17becf', '#e377c2', '#7f7f7f', '#bcbd22']

/** A published BEQ's curves carry `candidate` CATALOGUE_BASE + its index, and take colours of their own. */
export const CATALOGUE_BASE = 1000
export const CATALOGUE_COLOURS = ['#f0c419', '#e377c2', '#17becf', '#bcbd22', '#ff9896', '#9edae5']

export function seriesColour(s: Series, several: boolean): string {
  if ((s.candidate ?? 0) >= CATALOGUE_BASE) return CATALOGUE_COLOURS[(s.candidate! - CATALOGUE_BASE) % CATALOGUE_COLOURS.length]!
  if (s.kind === 'filter') return DESIGN_COLOURS[(s.candidate ?? 0) % DESIGN_COLOURS.length]!
  if (!s.filtered || !several) return MEASURED[s.kind]
  return DESIGN_COLOURS[(s.candidate ?? 0) % DESIGN_COLOURS.length]!
}

/** How one curve is drawn: dashed before a design, solid after; a peak dotted among several designs; the highlighted widest. */
export function seriesStyle(s: Series, all: Series[], highlighted: number | null) {
  const designs = new Set(all.filter((c) => c.filtered).map((c) => c.candidate))
  const several = designs.size > 1
  const anyFiltered = all.some((c) => c.filtered && c.kind !== 'filter')
  const focused = !several || s.candidate === null || s.candidate === highlighted
  return {
    stroke: seriesColour(s, several),
    width: s.kind === 'filter' ? 1.25 : (s.candidate ?? 0) >= CATALOGUE_BASE ? 2 : s.filtered ? (focused ? 2.5 : 1.25) : 1.5,
    dash: s.kind === 'filter' ? [2, 3]
      : !s.filtered ? (anyFiltered ? [6, 4] : undefined)
        : several && s.kind === 'peak' ? [1, 3] : undefined,
  }
}

interface Props {
  series: Series[]
  view: ChartView
  /** The design highlighted on the page, drawn widest among several. */
  highlighted?: number | null
  /** A drag or wheel zoom: the frequencies (and, for a vertical drag, the levels) to draw. */
  onZoom?: (x: Range, y: Range | null) => void
  /** A double click: back to the preset. */
  onReset?: () => void
  /** Frequency ranges to shade (the bands a redesign is asked to ignore). */
  bands?: Range[]
  height?: number
}

const NO_BANDS: Range[] = []   // one empty list, so a chart with no bands is not drawn again on every render

export function MagnitudeChart({ series, view, highlighted = null, onZoom, onReset, bands = NO_BANDS, height = 360 }: Props) {
  const holder = useRef<HTMLDivElement>(null)
  const zoom = useRef(onZoom)
  const reset = useRef(onReset)
  useEffect(() => {
    zoom.current = onZoom
    reset.current = onReset
  }, [onZoom, onReset])
  const shown = normalise(visibleSeries(series, view), view.reference)
  const scheme = useColourScheme()   // the axes take the stylesheet's colours when drawn: draw again when they change
  useEffect(() => {
    const element = holder.current
    const shown = normalise(visibleSeries(series, view), view.reference)
    if (!element || !shown.length) return
    const style = getComputedStyle(document.documentElement)
    const ink = style.getPropertyValue('--muted').trim() || '#888'
    const grid = style.getPropertyValue('--line').trim() || '#ddd'
    const axis = { stroke: ink, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 } }
    const range = drawnRange(view)
    const responses = shown.some((s) => s.kind === 'filter')
    const levels = view.reference === null ? 'dB' : 'dB, relative'
    const plot = new uPlot({
      width: element.clientWidth || 640,
      height,
      scales: {
        x: { time: false, range, ...(view.logX ? { distr: 3, log: 10 } : {}) },
        y: view.y ? { auto: false, range: view.y } : { auto: true },
        ...(responses ? { response: { auto: true } } : {}),
      },
      axes: [
        { ...axis, label: 'Frequency (Hz)',
          // uPlot's own log filter keeps only powers of ten (null for the rest): every tick of logTicks() is labelled
          ...(view.logX ? { splits: () => logTicks(range), filter: (_u: uPlot, ticks: number[]) => ticks,
                            values: (_u: uPlot, ticks: (number | null)[]) => ticks.map((t) => (t === null ? '' : String(t))) }
            : {}) },
        { ...axis, label: levels, size: 56 },
        ...(responses ? [{ ...axis, scale: 'response', side: 1, grid: { show: false }, label: 'Filter dB', size: 56 }] : []),
      ],
      legend: { show: true },
      cursor: { points: { show: false }, drag: { x: true, y: true, uni: 30, setScale: false } },
      hooks: {
        draw: [(u: uPlot) => {   // the bands asked to be ignored, under a light wash
          if (!bands.length || !u.ctx) return
          const { top, height: tall } = u.bbox
          u.ctx.save()
          u.ctx.fillStyle = 'rgba(214, 97, 107, 0.15)'
          for (const [low, high] of bands) {
            const a = u.valToPos(low, 'x', true), b = u.valToPos(high, 'x', true)
            u.ctx.fillRect(Math.min(a, b), top, Math.abs(b - a), tall)
          }
          u.ctx.restore()
        }],
        setSelect: [(u: uPlot) => {
          const { left, top, width, height: tall } = u.select
          if (width < 2 && tall < 2) return
          const x: Range = width >= 2 ? [u.posToVal(left, 'x'), u.posToVal(left + width, 'x')] : range
          const vertical = tall >= 2 && tall < u.over.clientHeight - 1
          const y: Range | null = vertical ? [u.posToVal(top + tall, 'y'), u.posToVal(top, 'y')] : view.y
          u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false)
          if (x[0] < x[1]) zoom.current?.(x, y)
        }],
      },
      series: [
        { label: 'Hz', value: (_u, v) => (v === null || v === undefined ? '—' : `${v.toFixed(1)} Hz`) },
        ...shown.map((s) => ({
          label: s.name,
          ...seriesStyle(s, shown, highlighted),
          ...(s.kind === 'filter' ? { scale: 'response' } : {}),
          value: (_u: uPlot, v: number | null) => (v === null || v === undefined ? '—' : `${v.toFixed(1)} dB`),
        })),
      ],
    }, chartData(shown, range), element)
    const wheel = (event: WheelEvent) => {
      if (!zoom.current || plot.cursor.left === undefined || plot.cursor.left < 0) return
      event.preventDefault()
      const at = plot.posToVal(plot.cursor.left, 'x')
      zoom.current(zoomRange(range, at, event.deltaY < 0 ? 0.8 : 1.25, view.logX), view.y)
    }
    const double = () => reset.current?.()
    plot.over?.addEventListener('wheel', wheel, { passive: false })
    plot.over?.addEventListener('dblclick', double)
    const resize = new ResizeObserver(() => plot.setSize({ width: element.clientWidth || 640, height }))
    resize.observe(element)
    return () => {
      plot.over?.removeEventListener('wheel', wheel)
      plot.over?.removeEventListener('dblclick', double)
      resize.disconnect()
      plot.destroy()
    }
  }, [series, view, highlighted, height, bands, scheme])
  if (!series.length) return <p className="muted">No measured curve for this title.</p>
  if (!shown.length) return <p className="muted">Nothing to draw: every curve is hidden.</p>
  return <div ref={holder} className="chart" role="img" aria-label={`Chart: ${shown.map((s) => s.name).join('; ')}`} />
}
