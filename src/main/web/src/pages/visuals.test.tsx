// The review page's tabs (design/web-parity.md WP2): the spectrogram, waited for while the service draws it, and the
// publication heatmap. jsdom has no canvas: getContext answers null, so these test what is asked for and shown, and
// src/chart/grid.test.ts tests the painting.
import { screen, waitFor } from '@testing-library/react'
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
  expect(screen.getByText(/Blue quieter, red louder/)).toBeInTheDocument()
}, 30_000)   // two drawings, each waited for through one 2 s poll: past vitest's 5 s default under a loaded run

it('shows the publication heatmap of the design highlighted', async () => {
  URL.createObjectURL = vi.fn(() => 'blob:heatmap')
  URL.revokeObjectURL = vi.fn()
  const asked = routes(0)
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('tab', { name: 'Heatmap' }))
  expect(await screen.findByRole('img', { name: /publication heatmap of Alien, design 1/ })).toHaveAttribute('src', 'blob:heatmap')
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
