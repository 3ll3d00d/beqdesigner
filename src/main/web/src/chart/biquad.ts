// A design's filters in the browser (design/web-parity.md WP8): the RBJ Audio EQ Cookbook biquads model/iir.py computes
// (LowShelf, HighShelf, PeakingEQ, Q convention, a shelf stacked `count` times), their magnitude response, and the
// stored form a design's filters come in (CompleteFilter.to_json()). Nothing here draws.
export type FilterType = 'low_shelf' | 'high_shelf' | 'peaking_eq'

export interface FilterRow {
  type: FilterType
  freq_hz: number
  gain_db: number
  q: number
  count: number
}

const STORED: Record<string, FilterType> = { LowShelf: 'low_shelf', HighShelf: 'high_shelf', PeakingEQ: 'peaking_eq' }
export const TYPE_NAMES: Record<FilterType, string> = { low_shelf: 'Low shelf', high_shelf: 'High shelf', peaking_eq: 'Peaking' }

/** A design's stored filters as rows; null when it has one this editor cannot show (it was not made by a designer). */
export function rowsOf(stored: unknown): { rows: FilterRow[]; fs: number } | null {
  const list = (stored as { filters?: unknown[] } | null)?.filters ?? []
  const rows: FilterRow[] = []
  let fs = 96000
  for (const item of list) {
    const f = item as { _type?: string; fs?: number; fc?: number; q?: number; gain?: number; count?: number }
    const type = STORED[f._type ?? '']
    if (!type || f.fc === undefined || f.q === undefined || f.gain === undefined) return null
    if (f.fs) fs = f.fs
    rows.push({ type, freq_hz: f.fc, gain_db: f.gain, q: f.q, count: f.count ?? 1 })
  }
  return { rows, fs }
}

/** [b0, b1, b2, a1, a2], normalised by a0. */
export function coefficients({ type, freq_hz, gain_db, q }: FilterRow, fs: number): number[] {
  const A = 10 ** (gain_db / 40)
  const w0 = (2 * Math.PI * freq_hz) / fs
  const cos = Math.cos(w0), alpha = Math.sin(w0) / (2 * q), root = 2 * Math.sqrt(A) * alpha
  let b: number[], a: number[]
  if (type === 'peaking_eq') {
    b = [1 + alpha * A, -2 * cos, 1 - alpha * A]
    a = [1 + alpha / A, -2 * cos, 1 - alpha / A]
  } else if (type === 'low_shelf') {
    b = [A * ((A + 1) - (A - 1) * cos + root), 2 * A * ((A - 1) - (A + 1) * cos), A * ((A + 1) - (A - 1) * cos - root)]
    a = [(A + 1) + (A - 1) * cos + root, -2 * ((A - 1) + (A + 1) * cos), (A + 1) + (A - 1) * cos - root]
  } else {
    b = [A * ((A + 1) + (A - 1) * cos + root), -2 * A * ((A - 1) + (A + 1) * cos), A * ((A + 1) + (A - 1) * cos - root)]
    a = [(A + 1) - (A - 1) * cos + root, 2 * ((A - 1) - (A + 1) * cos), (A + 1) - (A - 1) * cos - root]
  }
  return [b[0]! / a[0]!, b[1]! / a[0]!, b[2]! / a[0]!, a[1]! / a[0]!, a[2]! / a[0]!]
}

/** |H| in dB of one section at `hz`. */
export function sectionDb(c: number[], hz: number, fs: number): number {
  const w = (2 * Math.PI * hz) / fs
  const [b0, b1, b2, a1, a2] = c as [number, number, number, number, number]
  const re = (k0: number, k1: number, k2: number) => k0 + k1 * Math.cos(w) + k2 * Math.cos(2 * w)
  const im = (k1: number, k2: number) => -(k1 * Math.sin(w) + k2 * Math.sin(2 * w))
  const num = Math.hypot(re(b0, b1, b2), im(b1, b2)), den = Math.hypot(re(1, a1, a2), im(a1, a2))
  return 20 * Math.log10(num / den)
}

/** The cascade's response in dB at each of `freqs`. */
export function responseDb(rows: FilterRow[], fs: number, freqs: number[]): number[] {
  const sections = rows.map((r) => ({ c: coefficients(r, fs), count: Math.max(1, Math.round(r.count)) }))
  return freqs.map((hz) => (hz <= 0 ? 0 : sections.reduce((sum, s) => sum + s.count * sectionDb(s.c, hz, fs), 0)))
}

/** Why a row cannot be saved, in words (the service checks the same, and the section budget); '' when it can. */
export function rowProblem(row: FilterRow, fs: number): string {
  if (!(row.freq_hz > 0) || row.freq_hz >= fs / 2) return `the frequency must be above 0 and below ${fs / 2} Hz`
  if (!(row.q > 0)) return 'Q must be above 0'
  if (!Number.isFinite(row.gain_db)) return 'the gain must be a number'
  if (!(row.count >= 1 && row.count <= 10 && Number.isInteger(row.count))) return 'the count is 1 to 10'
  return ''
}

export const MAX_SECTIONS = 10
export const sections = (rows: FilterRow[]) => rows.reduce((n, r) => n + r.count, 0)
