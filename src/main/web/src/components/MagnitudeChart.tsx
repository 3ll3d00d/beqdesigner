// The review chart (design/web-app.md §4): a title's measured average and peak curves, and the same after the chosen
// design, on a log-frequency axis, as the desktop title page draws them -- dashed before, solid after. uPlot stays behind
// this one component, so its 1.7 release changes only this file.
import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

import type { Schemas } from '../api/client'

type Series = Schemas['ChartSeries']

const COLOURS = { average: '#2b8cbe', peak: '#e6550d' }

/** Linear interpolation of (x, y) at each of `at` (x ascending); outside the range, the end value. */
export function resample(x: number[], y: number[], at: number[]): number[] {
  if (x.length === at.length && x.every((v, i) => v === at[i])) return y
  const out: number[] = []
  let j = 0
  for (const v of at) {
    while (j < x.length - 2 && x[j + 1]! < v) j++
    const x0 = x[j]!, x1 = x[j + 1] ?? x0, y0 = y[j]!, y1 = y[j + 1] ?? y0
    out.push(v <= x0 ? y0 : v >= x1 ? y1 : y0 + ((y1 - y0) * (v - x0)) / (x1 - x0))
  }
  return out
}

/** The series on one shared frequency axis (the first's), above 0 Hz for the log scale: uPlot's data shape. */
export function chartData(series: Series[]): uPlot.AlignedData {
  if (!series.length) return [[]]
  const first = series[0]!
  const x = first.x.filter((v) => v > 0)
  return [x, ...series.map((s) => resample(s.x, s.y, x))]
}

export function MagnitudeChart({ series, height = 360 }: { series: Series[]; height?: number }) {
  const holder = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const element = holder.current
    if (!element || !series.length) return
    const filtered = series.some((s) => s.filtered)
    const style = getComputedStyle(document.documentElement)
    const ink = style.getPropertyValue('--muted').trim() || '#888'
    const grid = style.getPropertyValue('--line').trim() || '#ddd'
    const axis = { stroke: ink, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 } }
    const plot = new uPlot({
      width: element.clientWidth || 640,
      height,
      scales: { x: { time: false, distr: 3, log: 10 }, y: { auto: true } },
      axes: [{ ...axis, label: 'Frequency (Hz)' }, { ...axis, label: 'dB', size: 56 }],
      legend: { show: true },
      cursor: { points: { show: false } },
      series: [
        { label: 'Hz', value: (_u, v) => (v === null || v === undefined ? '—' : `${v.toFixed(1)} Hz`) },
        ...series.map((s) => ({
          label: s.name,
          stroke: COLOURS[s.kind],
          width: s.filtered ? 2 : 1.5,
          dash: filtered && !s.filtered ? [6, 4] : undefined,
          value: (_u: uPlot, v: number | null) => (v === null || v === undefined ? '—' : `${v.toFixed(1)} dB`),
        })),
      ],
    }, chartData(series), element)
    const resize = new ResizeObserver(() => plot.setSize({ width: element.clientWidth || 640, height }))
    resize.observe(element)
    return () => {
      resize.disconnect()
      plot.destroy()
    }
  }, [series, height])
  if (!series.length) return <p className="muted">No measured curve for this title.</p>
  return <div ref={holder} className="chart" role="img" aria-label={`Chart: ${series.map((s) => s.name).join('; ')}`} />
}
