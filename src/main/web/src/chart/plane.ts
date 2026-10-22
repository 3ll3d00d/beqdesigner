// The frequency x time plane the spectrogram and heatmap draw (design/web-parity.md WP2): where a pixel is, zooming about
// a point, keeping a range inside what there is, and reading a typed time. No DOM, so it is testable.
import type { Range } from './view'

/** A plot's box in canvas pixels and what it shows: frequencies across (log or linear), times down. */
export interface Axes {
  left: number
  top: number
  width: number
  height: number
  x: Range
  t: Range
  logX: boolean
}

const scaled = (log: boolean) => (v: number) => (log ? Math.log10(v) : v)
const unscaled = (log: boolean) => (v: number) => (log ? 10 ** v : v)

/** The frequency at canvas x `px`, clamped to the box. */
export function hzAt(axes: Axes, px: number): number {
  const s = scaled(axes.logX), u = Math.min(Math.max((px - axes.left) / axes.width, 0), 1)
  return unscaled(axes.logX)(s(axes.x[0]) + (s(axes.x[1]) - s(axes.x[0])) * u)
}

/** The canvas x of `hz`. */
export function pxOf(axes: Axes, hz: number): number {
  const s = scaled(axes.logX)
  return axes.left + (axes.width * (s(hz) - s(axes.x[0]))) / (s(axes.x[1]) - s(axes.x[0]))
}

/** The time at canvas y `py`, clamped to the box. */
export function timeAt(axes: Axes, py: number): number {
  const u = Math.min(Math.max((py - axes.top) / axes.height, 0), 1)
  return axes.t[0] + (axes.t[1] - axes.t[0]) * u
}

/** The canvas y of `seconds`. */
export const pyOf = (axes: Axes, seconds: number): number =>
  axes.top + (axes.height * (seconds - axes.t[0])) / (axes.t[1] - axes.t[0])

/** Whether canvas (`px`, `py`) is inside the box. */
export const inside = (axes: Axes, px: number, py: number): boolean =>
  px >= axes.left && px <= axes.left + axes.width && py >= axes.top && py <= axes.top + axes.height

/**
 * `range` kept inside `full`: moved back in if it runs over an end, and cut to `full` if it is wider. A range narrower than
 * `least` is widened about its middle.
 */
export function clampRange([low, high]: Range, full: Range, least = 0): Range {
  if (high - low < least) {
    const middle = (low + high) / 2
    ;[low, high] = [middle - least / 2, middle + least / 2]
  }
  const span = Math.min(high - low, full[1] - full[0])
  if (low < full[0]) [low, high] = [full[0], full[0] + span]
  if (high > full[1]) [low, high] = [full[1] - span, full[1]]
  return [Math.max(low, full[0]), Math.min(high, full[1])]
}

/** `range` scaled by `factor` (under 1 zooms in) about `at` (default its middle), log or linear, inside `full`. */
export function zoomAbout(range: Range, factor: number, full: Range, log = false, at?: number): Range {
  const s = scaled(log), u = unscaled(log)
  const [a, b] = [s(range[0]), s(range[1])]
  const centre = at === undefined ? (a + b) / 2 : s(at)
  const zoomed: Range = [u(centre - (centre - a) * factor), u(centre + (b - centre) * factor)]
  return clampRange(zoomed, full)
}

/** The two ends of a drag, low first. */
export const ordered = (a: number, b: number): Range => (a <= b ? [a, b] : [b, a])

/**
 * A typed time in seconds: h:mm:ss, m:ss or plain seconds (decimals allowed in the last part); null for anything else.
 */
export function parseClock(text: string): number | null {
  const parts = text.trim().split(':')
  if (!parts.length || parts.length > 3 || parts.some((p) => !/^\d+(\.\d+)?$/.test(p))) return null
  return parts.reduce((total, p) => total * 60 + Number(p), 0)
}

/** How many times narrower `shown` is than `full`, for saying how far in a view is. */
export const magnification = (shown: Range, full: Range): number =>
  (full[1] - full[0]) / Math.max(shown[1] - shown[0], Number.EPSILON)
