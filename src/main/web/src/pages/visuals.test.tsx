// The review page's tabs (design/web-parity.md WP2): the spectrogram, waited for while the service draws it, and the
// publication heatmap. jsdom has no canvas: getContext answers null, so these test what is asked for and shown, and
// src/chart/grid.test.ts tests the painting.
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { chart, index, METADATA_FIELDS, review } from '../test/fixtures'
import { encodeGrid } from '../test/grid'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

vi.mock('uplot', () => ({
  default: class {
    over = document.createElement('div')
    cursor = { left: -1 }
    setSize() {}
    destroy() {}
  },
}))

beforeEach(() => saveToken(TOKEN, false))

const json = (body: unknown) => HttpResponse.json(body as never)

function routes(draws = 1) {
  const asked: URLSearchParams[] = []
  const answered: Record<string, number> = {}
  const later = (key: string, body: () => Response) => {
    answered[key] = (answered[key] ?? 0) + 1
    return answered[key]! <= draws ? new HttpResponse(null, { status: 202, headers: { 'Retry-After': '1' } }) : body()
  }
  server.use(
    statusRoute(status({ index: index() })),
    http.get('/v1/titles/a/review', () => json(review())),
    http.get('/v1/metadata/fields', () => json(METADATA_FIELDS)),
    http.get('/v1/titles/a/chart', () => json(chart(0))),
    http.get('/v1/titles/a/spectrogram', ({ request }) => {
      const query = new URL(request.url).searchParams
      asked.push(query)
      return later(`s${query.get('filtered')}`, () => new HttpResponse(encodeGrid({ times: 2, freqs: 2, cells: [1, 2, 3, 4] }),
                                                                 { headers: { 'Content-Type': 'application/octet-stream' } }))
    }),
    http.get('/v1/titles/a/heatmap.png', ({ request }) => {
      asked.push(new URL(request.url).searchParams)
      return later('h', () => new HttpResponse(new Blob(['png'], { type: 'image/png' })))
    }),
  )
  return asked
}

it('shows the magnitude chart first, and the spectrogram once the service has drawn it', async () => {
  const asked = routes()
  renderApp('/titles/a')
  expect(await screen.findByRole('tab', { name: 'Magnitude' })).toHaveAttribute('aria-selected', 'true')
  await userEvent.click(screen.getByRole('tab', { name: 'Spectrogram' }))
  expect(await screen.findByText(/Drawing the spectrogram/)).toBeInTheDocument()
  expect(await screen.findByRole('img', { name: 'Spectrogram, as measured' }, { timeout: 10_000 })).toBeInTheDocument()
  expect(screen.getByRole('img', { name: 'The loudest level of each frequency' })).toBeInTheDocument()
  expect(asked[0]!.get('filtered')).toBe('false')
  expect(asked[0]!.get('candidate')).toBe('0')
  await userEvent.click(screen.getByLabelText('Difference'))
  await waitFor(() => expect(asked.some((q) => q.get('filtered') === 'true')).toBe(true))
  expect(await screen.findByRole('img', { name: 'Spectrogram, difference' }, { timeout: 10_000 })).toBeInTheDocument()
  expect(screen.getByRole('img', { name: /Blue quieter, red louder/ })).toBeInTheDocument()
}, 30_000)   // two drawings, each waited for through one 2 s poll: past vitest's 5 s default under a loaded run

// ten minutes at a second a row, 0-199 Hz at 1 Hz a column: room to zoom
const BIG = { times: 600, freqs: 200, df: 1, cells: Array.from({ length: 600 * 200 }, (_, i) => 1 + (i % 254)) }

function bigGrid(asked: URLSearchParams[] = []) {
  server.use(http.get('/v1/titles/a/spectrogram', ({ request }) => {
    asked.push(new URL(request.url).searchParams)
    return new HttpResponse(encodeGrid(BIG), { headers: { 'Content-Type': 'application/octet-stream' } })
  }))
}

const value = (group: HTMLElement, name: string) => (within(group).getByLabelText(name) as HTMLInputElement).value

async function type(group: HTMLElement, name: string, text: string) {
  const box = within(group).getByLabelText(name)
  await userEvent.clear(box)
  await userEvent.type(box, `${text}{Enter}`)
}

it('zooms the spectrogram to typed ranges, says how far in it is, and goes back and out again', async () => {
  routes(0)
  bigGrid()
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('tab', { name: 'Spectrogram' }))
  const group = await screen.findByRole('group', { name: 'Spectrogram view' })
  expect([value(group, 'Lowest frequency (Hz)'), value(group, 'Highest frequency (Hz)')]).toEqual(['1', '160'])
  expect([value(group, 'Start time'), value(group, 'End time')]).toEqual(['0:00:00', '0:09:59'])

  await type(group, 'Start time', '2:00')
  expect(value(group, 'Start time')).toBe('0:02:00')
  await type(group, 'Highest frequency (Hz)', '40')
  expect(value(group, 'Highest frequency (Hz)')).toBe('40')
  expect(within(group).getByText(/Zoomed ×5\.1 frequency, ×1\.3 time/)).toBeInTheDocument()

  await userEvent.click(within(group).getByRole('button', { name: 'Back' }))
  expect(value(group, 'Highest frequency (Hz)')).toBe('160')
  expect(value(group, 'Start time')).toBe('0:02:00')

  await type(group, 'Start time', 'soon')
  expect(within(group).getByLabelText('Start time')).toHaveAttribute('aria-invalid', 'true')

  await userEvent.click(within(group).getByRole('button', { name: 'Whole title' }))
  expect(value(group, 'Start time')).toBe('0:00:00')
  expect(within(group).getByRole('button', { name: 'Back' })).toBeDisabled()
  await userEvent.click(within(group).getByRole('button', { name: 'Zoom in' }))
  expect(within(group).getByText(/Zoomed ×2\.5 frequency, ×2\.0 time/)).toBeInTheDocument()
  await userEvent.click(within(group).getByRole('button', { name: 'Zoom out' }))
  expect(value(group, 'Start time')).toBe('0:00:00')
})

it('shows the box being dragged on the spectrogram, with its range, and zooms to it', async () => {
  routes(0)
  bigGrid()
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('tab', { name: 'Spectrogram' }))
  const canvas = await screen.findByRole('img', { name: 'Spectrogram, as measured' })
  const group = screen.getByRole('group', { name: 'Spectrogram view' })
  // the plot starts 64 px in, 736 px wide (an 800 px canvas) and 332 px tall
  fireEvent.pointerDown(canvas, { clientX: 64 + 184, clientY: 83, button: 0, pointerId: 1 })
  fireEvent.pointerMove(canvas, { clientX: 64 + 368, clientY: 166, pointerId: 1 })
  expect(screen.getByText(/40\.8–80\.5 Hz · 0:02:30–0:05:00/)).toBeInTheDocument()
  fireEvent.pointerUp(canvas, { clientX: 64 + 368, clientY: 166, pointerId: 1 })
  expect(screen.queryByText(/40\.8–80\.5 Hz ·/)).not.toBeInTheDocument()
  expect(value(group, 'Lowest frequency (Hz)')).toBe('40.75')
  expect(value(group, 'Start time')).toBe('0:02:30')
  expect(within(group).getByRole('button', { name: 'Back' })).toBeEnabled()
})

it('draws the heatmap here, 1-40 Hz, with the same controls, and the published image on request', async () => {
  URL.createObjectURL = vi.fn(() => 'blob:heatmap')
  URL.revokeObjectURL = vi.fn()
  const asked = routes(0)
  bigGrid(asked)
  server.use(http.get('/v1/titles/a/heatmap.png', ({ request }) => {
    asked.push(new URL(request.url).searchParams)
    return new HttpResponse(new Blob(['png'], { type: 'image/png' }))
  }))
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('tab', { name: 'Heatmap' }))
  expect(await screen.findByRole('img', { name: 'The heatmap of Alien, design 1: after the design, and as measured' }))
    .toBeInTheDocument()
  const group = screen.getByRole('group', { name: 'Heatmap view' })
  expect([value(group, 'Lowest frequency (Hz)'), value(group, 'Highest frequency (Hz)')]).toEqual(['1', '40'])
  expect(screen.getByRole('img', { name: /^-?\d+ to -?\d+ dB$/ })).toBeInTheDocument()   // the colour bar
  expect(asked.some((q) => q.has('filtered'))).toBe(true)                                // drawn from the grids
  expect(asked.some((q) => !q.has('filtered'))).toBe(false)                              // the PNG only when asked

  await userEvent.click(screen.getByRole('button', { name: 'Published image…' }))
  expect(await screen.findByRole('link', { name: 'Open the published image' })).toHaveAttribute('href', 'blob:heatmap')
  expect(asked.at(-1)!.get('candidate')).toBe('0')
})

it('says why a title cannot be drawn', async () => {
  routes()
  server.use(http.get('/v1/titles/a/spectrogram', () => HttpResponse.json(
    { title: 'Nothing to draw', status: 409, detail: 'No extracted mono audio is available for this title.' } as never,
    { status: 409 })))
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('tab', { name: 'Spectrogram' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('No extracted mono audio')
})
