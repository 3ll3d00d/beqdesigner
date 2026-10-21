// The review chart's workbench (design/web-parity.md WP1) against a mocked service: limits, presets, a log axis, smoothing,
// every design at once, filter responses, normalising, zooming by drag, and the view kept in this browser. uPlot draws on a
// canvas jsdom does not have, so it is replaced by a recorder of what it was given.
import { act, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { chart, index, METADATA_FIELDS, review } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

type Drawn = {
  options: {
    scales: Record<string, { distr?: number; range?: number[]; auto?: boolean }>
    axes: { scale?: string; label?: string }[]
    series: { label?: string; scale?: string; dash?: number[]; width?: number }[]
    hooks: { setSelect: ((u: unknown) => void)[] }
  }
  data: number[][]
}
const drawn = vi.hoisted(() => [] as Drawn[])
vi.mock('uplot', () => ({
  default: class {
    over = document.createElement('div')
    cursor = { left: -1 }
    constructor(options: Drawn['options'], data: number[][]) {
      drawn.push({ options, data })
    }
    setSize() {}
    setSelect() {}
    destroy() {}
  },
}))

beforeEach(() => {
  saveToken(TOKEN, false)
  drawn.length = 0
})

const json = (body: unknown) => HttpResponse.json(body as never)

function routes() {
  const asked: URLSearchParams[] = []
  server.use(
    statusRoute(status({ index: index() })),
    http.get('/v1/titles/a/review', () => json(review())),
    http.get('/v1/metadata/fields', () => json(METADATA_FIELDS)),
    http.get('/v1/titles/a/chart', ({ request }) => {
      const query = new URL(request.url).searchParams
      asked.push(query)
      const one = query.get('candidate')
      const designs = query.get('candidates') === 'all' ? [0, 1, 2] : one === null ? [] : [Number(one)]
      return json(chart(one === null ? null : Number(one), designs, query.get('responses') === 'true'))
    }),
  )
  return asked
}

const last = () => drawn.at(-1)!.options
const labels = () => last().series.slice(1).map((s) => s.label)

async function open() {
  const asked = routes()
  renderApp('/titles/a')
  await waitFor(() => expect(drawn.length).toBeGreaterThan(0))
  return asked
}

describe('the chart workbench', () => {
  it('sends the smoothing chosen and draws what comes back', async () => {
    const asked = await open()
    await userEvent.selectOptions(screen.getByLabelText('Smoothing'), '1/3 octave')
    await waitFor(() => expect(asked.at(-1)!.get('smoothing')).toBe('3'))
    expect(asked.at(-1)!.get('candidate')).toBe('0')
  })

  it('draws every design at once, the highlighted widest, each in a colour of its own', async () => {
    const asked = await open()
    await userEvent.click(screen.getByLabelText('All designs'))
    await waitFor(() => expect(asked.at(-1)!.get('candidates')).toBe('all'))
    await waitFor(() => expect(labels()).toHaveLength(5))
    const after = last().series.slice(3)
    expect(after.map((s) => s.width)).toEqual([2.5, 1.25, 1.25])
    expect(new Set(after.map((s) => (s as { stroke?: string }).stroke)).size).toBe(3)
  })

  it('puts each design\'s response on a second axis', async () => {
    const asked = await open()
    await userEvent.click(screen.getByLabelText('Filter response'))
    await waitFor(() => expect(asked.at(-1)!.get('responses')).toBe('true'))
    await waitFor(() => expect(labels()).toContain('Design 1'))
    expect(last().series.find((s) => s.label === 'Design 1')!.scale).toBe('response')
    expect(last().axes.map((a) => a.scale)).toContain('response')
  })

  it('changes the frequencies by preset, by typed limits and to a log axis', async () => {
    await open()
    await userEvent.click(screen.getByRole('button', { name: 'Sub' }))
    await waitFor(() => expect(last().scales.x!.range).toEqual([1, 40]))
    const from = screen.getByLabelText('Frequency from'), to = screen.getByLabelText('Frequency to')
    await userEvent.clear(from)
    await userEvent.type(from, '5')
    await userEvent.clear(to)
    await userEvent.type(to, '80{Enter}')
    await waitFor(() => expect(last().scales.x!.range).toEqual([5, 80]))
    await userEvent.click(screen.getByLabelText('Log frequency'))
    await waitFor(() => expect(last().scales.x!.distr).toBe(3))
  })

  it('says why limits cannot be used, and fits the level to what is drawn when they are blank', async () => {
    await open()
    const from = screen.getByLabelText('Level from'), to = screen.getByLabelText('Level to')
    await userEvent.type(from, '10')
    await userEvent.type(to, '-10{Enter}')
    expect(screen.getByRole('alert')).toHaveTextContent('The first must be the lower')
    await userEvent.clear(to)
    await userEvent.type(to, '20{Enter}')
    await waitFor(() => expect(last().scales.y!.range).toEqual([10, 20]))
    await userEvent.clear(from)
    await userEvent.clear(to)
    await userEvent.tab()
    await waitFor(() => expect(last().scales.y!.auto).toBe(true))
  })

  it('draws every curve relative to the one chosen', async () => {
    await open()
    await userEvent.selectOptions(screen.getByLabelText('Relative to'), 'Peak audio track (all channels mixed)')
    await waitFor(() => expect(drawn.at(-1)!.data[2]).toEqual([0, 0, 0, 0]))
    expect(drawn.at(-1)!.data[1]).toEqual([-10, -10, -10, -10])
  })

  it('zooms to the range dragged across, keeps the view for the next title, and resets', async () => {
    await open()
    const u = { select: { left: 10, top: 0, width: 30, height: 300 }, over: { clientHeight: 300 },
                posToVal: (px: number) => px, setSelect: () => {} }
    act(() => last().hooks.setSelect[0]!(u))
    await waitFor(() => expect(last().scales.x!.range).toEqual([10, 40]))
    expect(JSON.parse(window.localStorage.getItem('beq.chartView')!).x).toEqual([10, 40])
    await userEvent.click(screen.getByRole('button', { name: 'Reset view' }))
    await waitFor(() => expect(last().scales.x!.range).toEqual([1, 160]))
  })

  it('hides curves, and says so when nothing is left', async () => {
    await open()
    await userEvent.click(screen.getByLabelText('Average'))
    await waitFor(() => expect(labels()).toEqual(['Peak audio track (all channels mixed)']))
    await userEvent.click(screen.getByLabelText('Peak'))
    expect(await screen.findByText('Nothing to draw: every curve is hidden.')).toBeInTheDocument()
  })
})
