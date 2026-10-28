// The spectrogram grid the service sends (design/web-parity.md WP2; `pipeline.library.spectrum.SpectrogramGrid`): decoding
// it, and painting it into pixels -- as measured, after the design, or the difference -- with no DOM, so it is testable.
import type { Range } from './view'

export interface Grid {
  t0: number
  dt: number
  f0: number
  df: number
  floorDb: number
  stepDb: number
  times: number
  freqs: number
  /** times x freqs, time-major; 0 is below the floor. */
  cells: Uint8Array
}

export type GridMode = 'before' | 'after' | 'difference'

const HEADER = 40

export function decodeGrid(buffer: ArrayBuffer): Grid {
  const view = new DataView(buffer)
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4))
  if (magic !== 'BEQS' || view.getUint16(4, true) !== 1) throw new Error('not a version 1 spectrogram grid')
  const times = view.getUint32(8, true), freqs = view.getUint32(12, true)
  const f32 = (offset: number) => view.getFloat32(offset, true)
  return { times, freqs, t0: f32(16), dt: f32(20), f0: f32(24), df: f32(28), floorDb: f32(32), stepDb: f32(36),
           cells: new Uint8Array(buffer, HEADER, times * freqs) }
}

/** The level of a cell, dB; null below the floor. */
export function level(grid: Grid, time: number, freq: number): number | null {
  const value = grid.cells[time * grid.freqs + freq]!
  return value === 0 ? null : grid.floorDb + (value - 1) * grid.stepDb
}

// a perceptual dark-to-bright scale (after matplotlib's inferno), and blue-white-red for a difference
const SCALE = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39],
               [252, 255, 164]]
const DIVERGING = [[33, 102, 172], [146, 197, 222], [247, 247, 247], [244, 165, 130], [178, 24, 43]]

function ramp(stops: number[][], at: number): [number, number, number] {
  const x = Math.min(Math.max(at, 0), 1) * (stops.length - 1)
  const i = Math.min(Math.floor(x), stops.length - 2), u = x - i
  const a = stops[i]!, b = stops[i + 1]!
  return [0, 1, 2].map((k) => Math.round(a[k]! + (b[k]! - a[k]!) * u)) as [number, number, number]
}

/** The level scale as a CSS gradient, quietest on the left, for a colour bar. */
export const scaleGradient = (stops: number[][] = SCALE): string =>
  `linear-gradient(to right, ${stops.map(([r, g, b], i) => `rgb(${r},${g},${b}) ${(100 * i) / (stops.length - 1)}%`).join(', ')})`
export const differenceGradient = (): string => scaleGradient(DIVERGING)

/** The level at `seconds` and `hz`, dB; null outside the grid or below its floor. */
export function levelAt(grid: Grid, seconds: number, hz: number): number | null {
  const row = Math.round((seconds - grid.t0) / grid.dt), col = Math.round((hz - grid.f0) / grid.df)
  return row < 0 || row >= grid.times || col < 0 || col >= grid.freqs ? null : level(grid, row, col)
}

/** The colour of a level within `range` dB (quieter is darker), or of a difference within ±`span` dB. */
export const levelColour = (db: number, [low, high]: Range) => ramp(SCALE, (db - low) / (high - low))
export const differenceColour = (db: number, span: number) => ramp(DIVERGING, (db + span) / (2 * span))

export interface Frame {
  width: number
  height: number
  /** Frequencies across, Hz. */
  x: Range
  /** Times down, seconds (earliest at the top). */
  t: Range
  logX: boolean
  /** The levels the scale spans, dB. */
  levels: Range
  /** A difference's span either side of 0, dB. */
  span: number
}

/**
 * RGBA pixels (row-major) of `grid` drawn in `frame`: each pixel the nearest cell. `after` is the grid after the design,
 * needed for 'after' and 'difference'; a pixel outside the grid, or below the floor, is transparent.
 */
export function paint(before: Grid, after: Grid | null, mode: GridMode, frame: Frame): Uint8ClampedArray<ArrayBuffer> {
  const { width, height } = frame
  const pixels = new Uint8ClampedArray(new ArrayBuffer(width * height * 4))
  const grid = mode === 'after' && after ? after : before
  const [x0, x1] = frame.logX ? [Math.log10(frame.x[0]), Math.log10(frame.x[1])] : frame.x
  const columns = new Int32Array(width)
  for (let px = 0; px < width; px++) {
    const along = x0 + ((x1 - x0) * (px + 0.5)) / width
    const hz = frame.logX ? 10 ** along : along
    const i = Math.round((hz - grid.f0) / grid.df)
    columns[px] = i >= 0 && i < grid.freqs ? i : -1
  }
  for (let py = 0; py < height; py++) {
    const seconds = frame.t[0] + ((frame.t[1] - frame.t[0]) * (py + 0.5)) / height
    const row = Math.round((seconds - grid.t0) / grid.dt)
    if (row < 0 || row >= grid.times) continue
    for (let px = 0; px < width; px++) {
      const col = columns[px]!
      if (col < 0) continue
      let colour: [number, number, number] | null = null
      if (mode === 'difference') {
        const a = after ? level(after, row, col) : null, b = level(before, row, col)
        if (a !== null || b !== null) colour = differenceColour((a ?? before.floorDb) - (b ?? before.floorDb), frame.span)
      } else {
        const db = level(grid, row, col)
        if (db !== null) colour = levelColour(db, frame.levels)
      }
      if (!colour) continue
      const at = (py * width + px) * 4
      pixels[at] = colour[0]
      pixels[at + 1] = colour[1]
      pixels[at + 2] = colour[2]
      pixels[at + 3] = 255
    }
  }
  return pixels
}

/** The loudest level of each frequency over the times in `t`: the "where are the heavy hits" strip. */
export function loudestByFrequency(grid: Grid, t: Range): (number | null)[] {
  const first = Math.max(0, Math.floor((t[0] - grid.t0) / grid.dt))
  const last = Math.min(grid.times - 1, Math.ceil((t[1] - grid.t0) / grid.dt))
  return Array.from({ length: grid.freqs }, (_, col) => {
    let most: number | null = null
    for (let row = first; row <= last; row++) {
      const db = level(grid, row, col)
      if (db !== null && (most === null || db > most)) most = db
    }
    return most
  })
}

/** The full time range of a grid, seconds. */
export const gridTimes = (grid: Grid): Range => [grid.t0, grid.t0 + grid.dt * Math.max(grid.times - 1, 1)]

/** h:mm:ss of seconds. */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

/** The frequency of each of a grid's columns, Hz. */
export const gridFreqs = (grid: Grid): number[] => Array.from({ length: grid.freqs }, (_, i) => grid.f0 + i * grid.df)

/**
 * The top of one colour scale for a title, the same whichever of its designs is drawn: the loudest the measured grid
 * reaches, or the loudest any design lifts it to -- each frequency's loudest measured cell plus that design's response
 * there (`responses`, dB at `gridFreqs(before)`), which is what the design's own grid would show at its loudest.
 */
export function sharedTop(before: Grid, responses: number[][]): number {
  let top = -Infinity
  loudestByFrequency(before, gridTimes(before)).forEach((db, f) => {
    if (db !== null) top = Math.max(top, db, ...responses.map((r) => db + (r[f] ?? 0)))
  })
  return Number.isFinite(top) ? top : before.floorDb + 254 * before.stepDb
}
