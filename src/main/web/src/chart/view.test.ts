// The chart's view model (design/web-parity.md WP1): what is kept in the browser, what is drawn, and how it is placed.
import { chart } from '../test/fixtures'
import { chartData, DEFAULT_VIEW, drawnRange, loadView, logTicks, normalise, saveView, type Series, visibleSeries, zoomRange }
  from './view'

const line = (name: string, x: number[], y: number[], extra: Partial<Series> = {}): Series =>
  ({ name, kind: 'average', filtered: false, x, y, ...extra })

describe('the view kept in this browser', () => {
  it('is the defaults when nothing, or something damaged, is kept', () => {
    expect(loadView()).toEqual(DEFAULT_VIEW)
    window.localStorage.setItem('beq.chartView', '{not json')
    expect(loadView()).toEqual(DEFAULT_VIEW)
    window.localStorage.setItem('beq.chartView', JSON.stringify({ x: [50, 10], y: 'loud', smoothing: '5', peak: 'no' }))
    expect(loadView()).toEqual(DEFAULT_VIEW)
  })

  it('round-trips, and survives storage that refuses', () => {
    const view = { ...DEFAULT_VIEW, x: [5, 40] as [number, number], y: [-20, 10] as [number, number], logX: true,
                   smoothing: '6' as const, peak: false, allDesigns: true, reference: 'Average' }
    saveView(view)
    expect(loadView()).toEqual(view)
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('quota') })
    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('denied') })
    expect(() => saveView(DEFAULT_VIEW)).not.toThrow()
    expect(loadView()).toEqual(DEFAULT_VIEW)
    setItem.mockRestore()
    getItem.mockRestore()
  })
})

describe('what is drawn', () => {
  it('hides curves by kind and by before or after a design, and shows responses only when asked', () => {
    const series = chart(0, [0], true).series
    const names = (view: Partial<typeof DEFAULT_VIEW>) => visibleSeries(series, { ...DEFAULT_VIEW, ...view }).map((s) => s.name)
    expect(names({})).toEqual(['Average audio track (all channels mixed)', 'Peak audio track (all channels mixed)',
                               'Filtered average audio track (all channels mixed)'])
    expect(names({ peak: false, before: false, responses: true })).toEqual(
      ['Filtered average audio track (all channels mixed)', 'Design 1'])
  })

  it('normalises every curve but a response to the reference, which becomes flat', () => {
    const series = [line('ref', [1, 2, 3], [5, 6, 7]), line('other', [1, 3], [10, 10]),
                    line('resp', [1, 2], [3, 3], { kind: 'filter', filtered: true })]
    expect(normalise(series, 'ref').map((s) => s.y)).toEqual([[0, 0, 0], [5, 3], [3, 3]])
    expect(normalise(series, 'gone')).toBe(series)
    expect(normalise(series, null)).toBe(series)
  })

  it('places every curve on the frequencies of the finest one in range', () => {
    const coarse = line('coarse', [1, 100], [0, 99])
    const fine = line('fine', [1, 25, 50, 75, 100, 500], [1, 1, 1, 1, 1, 1])
    expect(chartData([coarse, fine], [1, 100])).toEqual([[1, 25, 50, 75, 100], [0, 24, 49, 74, 99], [1, 1, 1, 1, 1]])
  })

  it('keeps a log axis above 0 Hz, and zooms about a point on either scale', () => {
    expect(drawnRange({ ...DEFAULT_VIEW, x: [0, 160], logX: true })).toEqual([1, 160])
    expect(drawnRange({ ...DEFAULT_VIEW, x: [0, 160] })).toEqual([0, 160])
    expect(zoomRange([0, 100], 50, 0.5, false)).toEqual([25, 75])
    const [low, high] = zoomRange([1, 100], 10, 0.5, true)
    expect(low).toBeCloseTo(10 ** 0.5)
    expect(high).toBeCloseTo(10 ** 1.5)
  })
})

describe('a log frequency axis', () => {
  it('is labelled at 1, 2 and 5 of each decade, or every step inside one decade', () => {
    expect(logTicks([1, 160])).toEqual([1, 2, 5, 10, 20, 50, 100])
    expect(logTicks([5.57, 46.9])).toEqual([6, 7, 8, 9, 10, 20, 30, 40])
    expect(logTicks([12, 18])).toEqual([12, 13.5, 15, 16.5, 18])
  })
})
