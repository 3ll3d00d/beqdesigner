// The spectrogram grid (design/web-parity.md WP2): decoding the service's bytes and painting them.
import { encodeGrid } from '../test/grid'
import { clock, decodeGrid, differenceColour, gridTimes, level, levelColour, loudestByFrequency, paint } from './grid'

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
