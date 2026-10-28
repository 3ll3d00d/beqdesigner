// The spectrogram grid (design/web-parity.md WP2): decoding the service's bytes and painting them.
import { encodeGrid } from '../test/grid'
import { byLevel } from '../components/HeatmapView'
import { scaleTop } from '../components/Spectrogram'
import { clock, decodeGrid, differenceColour, gridTimes, level, levelAt, levelColour, loudestByFrequency, paint,
  scaleGradient, sharedTop } from './grid'

// 3 times x 4 frequencies (0, 10, 20, 30 Hz); 0 is below the floor
const grid = () => decodeGrid(encodeGrid({ times: 3, freqs: 4, cells: [0, 50, 100, 1, 0, 200, 0, 1, 255, 0, 0, 0] }))

describe('the grid', () => {
  it('decodes the header and reads a cell as a level', () => {
    const g = grid()
    expect([g.times, g.freqs, g.dt, g.df, g.floorDb, g.stepDb]).toEqual([3, 4, 1, 10, -100, 1])
    expect(level(g, 0, 0)).toBeNull()
    expect(level(g, 0, 2)).toBe(-1)          // floor + (100 - 1) steps
    expect(level(g, 2, 0)).toBe(154)
    expect(gridTimes(g)).toEqual([0, 2])
    expect(() => decodeGrid(new ArrayBuffer(40))).toThrow('not a version 1')
  })

  it('paints each pixel the nearest cell, leaving what is below the floor or outside the grid clear', () => {
    const pixels = paint(grid(), null, 'before', { width: 4, height: 3, x: [-5, 35], t: [-0.5, 2.5], logX: false,
                                                   levels: [-100, 155], span: 20 })
    const alpha = (px: number, py: number) => pixels[(py * 4 + px) * 4 + 3]
    expect([alpha(0, 0), alpha(1, 0), alpha(2, 0), alpha(3, 0)]).toEqual([0, 255, 255, 255])
    expect(Array.from(pixels.slice(0, 3))).toEqual([0, 0, 0])
    const loud = (2 * 4 + 0) * 4
    expect(Array.from(pixels.slice(loud, loud + 3))).toEqual(levelColour(154, [-100, 155]))
  })

  it('paints the difference after the design, blue quieter and red louder', () => {
    const before = grid()
    const after = decodeGrid(encodeGrid({ times: 3, freqs: 4, cells: [0, 60, 90, 1, 0, 200, 0, 1, 255, 0, 0, 0] }))
    const pixels = paint(before, after, 'difference', { width: 4, height: 1, x: [-5, 35], t: [0, 0.4], logX: false,
                                                        levels: [-100, 0], span: 20 })
    expect(Array.from(pixels.slice(4, 7))).toEqual(differenceColour(10, 20))    // 10 dB louder
    expect(Array.from(pixels.slice(8, 11))).toEqual(differenceColour(-10, 20))  // 10 dB quieter
    expect(differenceColour(-20, 20)[2]).toBeGreaterThan(differenceColour(-20, 20)[0])
  })

  it('finds the loudest level of each frequency in a time range, and says times as a clock', () => {
    expect(loudestByFrequency(grid(), [0, 2])).toEqual([154, 99, -1, -100])
    expect(loudestByFrequency(grid(), [0, 0])).toEqual([null, -51, -1, -100])
    expect(clock(3725)).toBe('1:02:05')
  })
})

describe('reading the grid for a readout and a colour bar', () => {
  it('reads the level at a time and frequency, null outside or below the floor', () => {
    const g = grid()
    expect(levelAt(g, 0, 20)).toBe(-1)
    expect(levelAt(g, 0.4, 21)).toBe(-1)      // the nearest cell
    expect(levelAt(g, 0, 0)).toBeNull()
    expect(levelAt(g, 9, 0)).toBeNull()
  })

  it('spells the scale as a gradient, quietest first', () => {
    expect(scaleGradient()).toMatch(/^linear-gradient\(to right, rgb\(0,0,4\) 0%, .* rgb\(252,255,164\) 100%\)$/)
  })

  it("groups a heatmap's cells by level, leaving out what is below the floor", () => {
    const groups = byLevel(grid())
    expect(groups[0]!.length).toBe(0)
    expect(Array.from(groups[1]!)).toEqual([3, 7])
    expect(Array.from(groups[255]!)).toEqual([8])
    expect(groups.reduce((n, g) => n + g.length, 0)).toBe(6)
  })
})

describe('one colour scale for every design of a title', () => {
  // the loudest of each column: 154 dB at 0 Hz, 99 at 10, -1 at 20, -100 at 30
  it("is the measured grid's top, or the most any design lifts a frequency's loudest cell to", () => {
    expect(sharedTop(grid(), [])).toBe(154)
    expect(sharedTop(grid(), [[0, 60, 0, 0]])).toBe(159)                     // 99 + 60 at 10 Hz
    expect(sharedTop(grid(), [[0, 60, 0, 0], [0, 0, 0, 300]])).toBe(200)     // -100 + 300 at 30 Hz: the larger
    expect(sharedTop(grid(), [[-10, 0, 0, 0]])).toBe(154)                    // a cut never lowers it below the measured
  })

  it('is the same whichever design is drawn, and covers one whose filters cannot be read', () => {
    const designs = [{ rows: [{ type: 'low_shelf' as const, freq_hz: 20, gain_db: 10, q: 0.7, count: 1 }], fs: 96000 },
                     { rows: [], fs: 96000 }]
    const loud = decodeGrid(encodeGrid({ times: 1, freqs: 4, cells: [255, 255, 255, 255], floorDb: 100 }))
    expect(scaleTop(grid(), designs, loud)).toBe(scaleTop(grid(), designs, null))   // the drawn grid does not move it
    expect(scaleTop(grid(), [...designs, null], loud)).toBe(354)                    // unless a design cannot be read
  })
})

