// The filters behind a design and behind each published BEQ, readable on the review page and on the Published BEQs tab
// (src/components/FilterTable.tsx), not only drawn on the chart.
import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { candidate, chart, index, METADATA_FIELDS, review } from '../test/fixtures'
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
const shelf = { _type: 'LowShelf', fs: 96000, fc: 20, q: 0.9, gain: 5.5, count: 2 }
const peak = { _type: 'PeakingEQ', fs: 96000, fc: 32.5, q: 2, gain: -3, count: 1 }

function routes() {
  server.use(
    statusRoute(status({ index: index() })),
    http.get('/v1/titles/a/review', () => json(review({
      candidates: [candidate(0, { filters: { filters: [shelf, peak] } }), candidate(1, { filters: { filters: [peak] } })],
    }))),
    http.get('/v1/metadata/fields', () => json(METADATA_FIELDS)),
    http.get('/v1/titles/a/chart', () => json(chart(0))),
    http.get('/v1/titles/a/catalogue', () => json({ key: 'film-alien_348', matched_by: 'tmdb', compare_url: null, message: '',
      entries: [{ index: 0, author: 'aron7awol', audio_types: [], edition: '', season: '', episodes: '', same_track: true,
                  catalogue_url: null, filter_count: 2,
                  filters: [{ type: 'low_shelf', freq_hz: 25, gain_db: 4, q: 0.707, count: 3 },
                            { type: 'peaking_eq', freq_hz: 40, gain_db: 2.5, q: 1.5, count: 1 }] }] })),
  )
}

const cells = (table: HTMLElement) =>
  within(table).getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('cell').map((c) => c.textContent))

it("shows the highlighted design's filters, and follows the highlight", async () => {
  routes()
  renderApp('/titles/a')
  const table = await screen.findByRole('table', { name: "Design 1's filters" })
  expect(cells(table)).toEqual([['Low shelf', '20', '0.900', '+5.5', '×2'], ['Peaking', '32.5', '2', '-3', '']])
  await userEvent.click(screen.getAllByRole('radio')[1]!)
  expect(cells(await screen.findByRole('table', { name: "Design 2's filters" }))).toEqual([['Peaking', '32.5', '2', '-3', '']])
})

it("lists each published BEQ's filters", async () => {
  routes()
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('tab', { name: 'Published BEQs' }))
  await userEvent.click(await screen.findByText('2 filters'))
  expect(cells(screen.getByRole('table', { name: "aron7awol's filters" })))
    .toEqual([['Low shelf', '25', '0.707', '+4', '×3'], ['Peaking', '40', '1.500', '+2.5', '']])
})
