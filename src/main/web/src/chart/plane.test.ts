// The spectrogram's and heatmap's plane (design/web-parity.md WP2): pixels and values, zooming, keeping in range, times.
import { clampRange, hzAt, inside, magnification, parseClock, pxOf, pyOf, timeAt, zoomAbout, type Axes } from './plane'

const axes = (logX = false): Axes => ({ left: 50, top: 10, width: 200, height: 100, x: [10, 1000], t: [0, 100], logX })

describe('the plane', () => {
  it('turns pixels into values and back, linear or log, clamped to the box', () => {
    expect(hzAt(axes(), 50)).toBe(10)
    expect(hzAt(axes(), 150)).toBe(505)
    expect(hzAt(axes(true), 150)).toBeCloseTo(100)
    expect(pxOf(axes(true), 100)).toBeCloseTo(150)
    expect(hzAt(axes(), 0)).toBe(10)          // left of the box: its first frequency
    expect(timeAt(axes(), 60)).toBe(50)
    expect(pyOf(axes(), 25)).toBe(35)
    expect(inside(axes(), 49, 50)).toBe(false)
    expect(inside(axes(), 60, 50)).toBe(true)
  })

  it('keeps a range inside what there is, and at least as wide as asked', () => {
    expect(clampRange([-10, 20], [0, 100])).toEqual([0, 30])
    expect(clampRange([90, 130], [0, 100])).toEqual([60, 100])
    expect(clampRange([-50, 500], [0, 100])).toEqual([0, 100])
    expect(clampRange([40, 40.1], [0, 100], 2)).toEqual([39.05, 41.05])
  })

  it('zooms about the middle or a point, log or linear, and never past the whole', () => {
    expect(zoomAbout([0, 100], 0.5, [0, 100])).toEqual([25, 75])
    expect(zoomAbout([0, 100], 0.5, [0, 100], false, 0)).toEqual([0, 50])
    expect(zoomAbout([25, 75], 4, [0, 100])).toEqual([0, 100])
    const [a, b] = zoomAbout([10, 1000], 0.5, [1, 1000], true)
    expect(a).toBeCloseTo(31.62, 2)
    expect(b).toBeCloseTo(316.23, 2)
    expect(magnification([25, 75], [0, 100])).toBe(2)
  })

  it('reads a typed time as h:mm:ss, m:ss or seconds, and refuses the rest', () => {
    expect(parseClock('1:02:03')).toBe(3723)
    expect(parseClock('2:30')).toBe(150)
    expect(parseClock(' 90.5 ')).toBe(90.5)
    expect(parseClock('1:2:3:4')).toBeNull()
    expect(parseClock('soon')).toBeNull()
    expect(parseClock('')).toBeNull()
  })
})
