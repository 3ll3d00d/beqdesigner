// The browser's biquads against model/iir.py's (design/web-parity.md WP8): the same response, so what is edited here is
// what the desktop and publish compute. The expected values are iir.py's, interpolated on its transfer-function grid.
import { responseDb, rowProblem, rowsOf, sections, type FilterRow } from './biquad'

const FS = 96000
const at = [5, 20, 40, 100, 200]
const row = (type: FilterRow['type'], freq_hz: number, q: number, gain_db: number, count = 1): FilterRow =>
  ({ type, freq_hz, q, gain_db, count })

it.each([
  [row('low_shelf', 20, 0.7, 5, 2), [9.941, 5.006, 0.659, 0.025, 0.003]],
  [row('peaking_eq', 40, 2, -3), [-0.013, -0.306, -2.986, -0.164, -0.033]],
  [row('high_shelf', 100, 0.9, 4), [-0.004, -0.057, -0.156, 2.0, 4.146]],
])('computes %j as model/iir.py does', (filter, expected) => {
  responseDb([filter], FS, at).forEach((db, i) => expect(db).toBeCloseTo(expected[i]!, 1))
})

it('adds the sections of a cascade, and is flat at 0 Hz and with nothing in it', () => {
  const both = responseDb([row('low_shelf', 20, 0.7, 5), row('peaking_eq', 40, 2, -3)], FS, [40])[0]!
  expect(both).toBeCloseTo(responseDb([row('low_shelf', 20, 0.7, 5)], FS, [40])[0]! + responseDb([row('peaking_eq', 40, 2, -3)], FS, [40])[0]!)
  expect(responseDb([], FS, [10, 0])).toEqual([0, 0])
})

it('reads a stored design, and refuses one it cannot show', () => {
  const stored = { filters: [{ _type: 'LowShelf', fs: 1000, fc: 20, q: 0.7, gain: 4.5, count: 2 },
                             { _type: 'PeakingEQ', fs: 1000, fc: 40, q: 2, gain: -3 }] }
  expect(rowsOf(stored)).toEqual({ fs: 1000, rows: [row('low_shelf', 20, 0.7, 4.5, 2), row('peaking_eq', 40, 2, -3)] })
  expect(rowsOf({ filters: [] })).toEqual({ fs: 96000, rows: [] })
  expect(rowsOf({ filters: [{ _type: 'AllPass', fc: 1 }] })).toBeNull()
  expect(sections(rowsOf(stored)!.rows)).toBe(3)
})

it('says why a row cannot be saved', () => {
  expect(rowProblem(row('low_shelf', 0, 0.7, 5), FS)).toMatch(/frequency/)
  expect(rowProblem(row('low_shelf', 20, 0, 5), FS)).toMatch(/Q/)
  expect(rowProblem(row('low_shelf', 20, 0.7, 5, 11), FS)).toMatch(/count/)
  expect(rowProblem(row('low_shelf', 20, 0.7, 5), FS)).toBe('')
})
