// What the review chart shows and how (design/web-parity.md WP1): the frequency and level limits, a log frequency axis, the
// smoothing the service applies, which curves are drawn, every design at once, and normalising to one curve. Kept per
// browser, not per title, so a person's way of looking carries from title to title. No React and no uPlot here.
import type uPlot from 'uplot'

import type { Schemas } from '../api/client'

export type Series = Schemas['ChartSeries']
export type Range = [number, number]
export const SMOOTHINGS = ['none', '1', '2', '3', '6', '12', '24', '48', 'sg'] as const
export type Smoothing = (typeof SMOOTHINGS)[number]

export interface ChartView {
  /** The frequencies drawn, Hz. */
  x: Range
  /** The levels drawn, dB; null: fitted to what is drawn. */
  y: Range | null
  logX: boolean
  smoothing: Smoothing
  average: boolean
  peak: boolean
  /** The measured curves, before any design. */
  before: boolean
  /** The curves after a design. */
  after: boolean
  /** Each design's own response, on a second level axis. */
  responses: boolean
  /** Every design at once rather than the one highlighted. */
  allDesigns: boolean
  /** The curve (by name) every other is drawn relative to; null: none. */
  reference: string | null
}

/** 1 to 160 Hz, linear: BEQ is about the bass, and the desktop's charts default to the same range. */
export const FREQUENCY_RANGE: Range = [1, 160]

export const PRESETS: { id: string; label: string; x: Range | 'full' }[] = [
  { id: 'bass', label: 'Bass', x: FREQUENCY_RANGE },
  { id: 'sub', label: 'Sub', x: [1, 40] },
  { id: 'full', label: 'Full', x: 'full' },
]

export const DEFAULT_VIEW: ChartView = {
  x: FREQUENCY_RANGE, y: null, logX: false, smoothing: 'none', average: true, peak: true, before: true, after: true,
  responses: false, allDesigns: false, reference: null,
}

export function smoothingLabel(smoothing: Smoothing): string {
  return smoothing === 'none' ? 'None' : smoothing === 'sg' ? 'Savitzky-Golay' : `1/${smoothing} octave`
}

const STORAGE_KEY = 'beq.chartView'

function isRange(value: unknown): value is Range {
  return Array.isArray(value) && value.length === 2 && value.every((v) => typeof v === 'number' && Number.isFinite(v))
    && value[0] < value[1]
}

/** The view kept in this browser, each setting checked and any that is missing or damaged at its default. */
export function loadView(): ChartView {
  let kept: Record<string, unknown> = {}
  try {
    const text = window.localStorage.getItem(STORAGE_KEY)
    const parsed: unknown = text ? JSON.parse(text) : {}
    if (parsed && typeof parsed === 'object') kept = parsed as Record<string, unknown>
  } catch {
    // storage refused or the value is damaged: the defaults
  }
  const flag = (key: keyof ChartView) => (typeof kept[key] === 'boolean' ? kept[key] : DEFAULT_VIEW[key]) as boolean
  return {
    x: isRange(kept.x) ? kept.x : DEFAULT_VIEW.x,
    y: isRange(kept.y) ? kept.y : null,
    logX: flag('logX'),
    smoothing: SMOOTHINGS.includes(kept.smoothing as Smoothing) ? (kept.smoothing as Smoothing) : 'none',
    average: flag('average'), peak: flag('peak'), before: flag('before'), after: flag('after'),
    responses: flag('responses'), allDesigns: flag('allDesigns'),
    reference: typeof kept.reference === 'string' ? kept.reference : null,
  }
}

export function saveView(view: ChartView): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(view))
  } catch {
    // storage refused: kept for this page only
  }
}

/** The x range a log axis can draw: nothing at or below 0 Hz. */
export function drawnRange(view: ChartView): Range {
  const [low, high] = view.x
  return view.logX && low <= 0 ? [Math.min(1, high / 2), high] : view.x
}

/** The curves the view shows, in the order given. */
export function visibleSeries(series: Series[], view: ChartView): Series[] {
  return series.filter((s) => {
    if (s.kind === 'filter') return view.responses
    if (!(s.kind === 'average' ? view.average : view.peak)) return false
    return s.filtered ? view.after : view.before
  })
}

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

/**
 * Every curve relative to the one named `reference` (which becomes flat at 0 dB). A design's response is already relative,
 * so it is left as it is; so is everything when the reference is not among the curves.
 */
export function normalise(series: Series[], reference: string | null): Series[] {
  const base = reference === null ? undefined : series.find((s) => s.name === reference && s.kind !== 'filter')
  if (!base) return series
  return series.map((s) => {
    if (s.kind === 'filter') return s
    const under = resample(base.x, base.y, s.x)
    return { ...s, y: s.y.map((v, i) => v - under[i]!) }
  })
}

/**
 * The series on one shared frequency axis within `range`: uPlot's data shape. The axis is the frequencies of the curve with
 * the most points in range (the first, of equals), so the finest curve is drawn as it is and the others are interpolated.
 */
export function chartData(series: Series[], range: Range = FREQUENCY_RANGE): uPlot.AlignedData {
  if (!series.length) return [[]]
  const [low, high] = range
  const inRange = (s: Series) => s.x.filter((v) => v >= low && v <= high)
  let x = inRange(series[0]!)
  for (const s of series.slice(1)) {
    const candidate = inRange(s)
    if (candidate.length > x.length) x = candidate
  }
  return [x, ...series.map((s) => resample(s.x, s.y, x))]
}

/** The widest frequency any curve reaches: the *Full* preset's top. */
export function highestFrequency(series: Series[]): number {
  return series.reduce((most, s) => Math.max(most, ...s.x), FREQUENCY_RANGE[1])
}

/** `range` zoomed about `at` by `factor` (< 1 zooms in), on a log scale when `log`. */
export function zoomRange(range: Range, at: number, factor: number, log: boolean): Range {
  const into = (v: number) => (log ? Math.log10(v) : v)
  const out = (v: number) => (log ? 10 ** v : v)
  const [a, b, c] = [into(range[0]), into(range[1]), into(Math.min(Math.max(at, range[0]), range[1]))]
  return [out(c - (c - a) * factor), out(c + (b - c) * factor)]
}

/**
 * Where a log frequency axis is labelled within `range`: 1, 2 and 5 of each decade, or every whole number of a decade
 * when that gives fewer than four (a zoom inside one decade), else five evenly spaced: a narrow range always has labels.
 */
export function logTicks([low, high]: Range): number[] {
  const within = (steps: number[]) => {
    const ticks: number[] = []
    for (let decade = Math.floor(Math.log10(Math.max(low, 1e-3))); 10 ** decade <= high; decade++) {
      for (const step of steps) {
        const value = Number((step * 10 ** decade).toPrecision(6))
        if (value >= low && value <= high) ticks.push(value)
      }
    }
    return ticks
  }
  const coarse = within([1, 2, 5])
  if (coarse.length >= 4) return coarse
  const fine = within([1, 2, 3, 4, 5, 6, 7, 8, 9])
  if (fine.length >= 3) return fine
  // narrower still: five evenly spaced, to the precision the range needs
  const places = Math.max(0, 2 - Math.floor(Math.log10(high - low)))
  return [0, 1, 2, 3, 4].map((i) => Number((low + ((high - low) * i) / 4).toFixed(places)))
}
