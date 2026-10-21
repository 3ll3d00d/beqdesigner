// The Published BEQs tab (design/web-parity.md WP3) against a mocked service: the catalogue's entries for the title,
// the same track first, and any of them on the chart beside our design with how far apart they are.
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { comparison } from '../components/PublishedBeqs'
import { chart, index, METADATA_FIELDS, review } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

const drawn = vi.hoisted(() => [] as { series: { label?: string; stroke?: string }[] }[])
vi.mock('uplot', () => ({
  default: class {
    over = document.createElement('div')
    cursor = { left: -1 }
    constructor(options: { series: { label?: string }[] }) {
      drawn.push(options)
    }
    setSize() {}
    destroy() {}
  },
}))

beforeEach(() => {
  saveToken(TOKEN, false)
  drawn.length = 0
})

const json = (body: unknown, code?: number) => HttpResponse.json(body as never, code ? { status: code } : undefined)
const x = [0, 10, 20, 40, 80]
const entries = [
  { index: 0, author: 'aron7awol', audio_types: ['DTS-HD MA 5.1'], edition: '', season: '', episodes: '', same_track: true,
    catalogue_url: 'https://example.org/aron7awol/1/#alien', filter_count: 2 },
  { index: 1, author: 'mobe1969', audio_types: ['Atmos'], edition: "Director's Cut", season: '', episodes: '',
    same_track: false, catalogue_url: null, filter_count: 1 },
]

function routes(listed = entries) {
  const charted: string[] = []
  server.use(
    statusRoute(status({ index: index() })),
    http.get('/v1/titles/a/review', () => json(review())),
    http.get('/v1/metadata/fields', () => json(METADATA_FIELDS)),
    http.get('/v1/titles/a/chart', () => json(chart(0))),
    http.get('/v1/titles/a/catalogue', () => json({ key: 'film-alien_348', matched_by: 'tmdb', entries: listed,
                                                   compare_url: 'https://example.org/compare/?t=film-alien_348',
                                                   message: listed.length ? '' : 'The catalogue has no BEQ for this title.' })),
    http.get('/v1/titles/a/catalogue/:n/chart', ({ params }) => {
      charted.push(String(params.n))
      return json({ candidate: null, candidates: [], series: [
        { name: "aron7awol's BEQ", kind: 'filter', filtered: true, x, y: [10, 10, 6, 2, 0] },
        { name: "Average audio track (all channels mixed) under aron7awol's BEQ", kind: 'average', filtered: true, x,
          y: [10, 10, 6, 2, 0] }] })
    }),
  )
  return charted
}

it('says how far apart two BEQs are', () => {
  const theirs = [{ name: 'r', kind: 'filter' as const, filtered: true, x, y: [3, 9, 5, 1, 0] },
                  { name: 'u', kind: 'average' as const, filtered: true, x, y: [0, 2, 2, 2, 2] }]
  const ours = { name: 'o', kind: 'average' as const, filtered: true, x, y: [0, 4, 4, 4, 4] }
  expect(comparison(theirs, ours)).toEqual({ peak: { db: 9, hz: 10 }, rms: 2 })
  expect(comparison([], undefined)).toEqual({ peak: null, rms: null })
})

it('lists the same track first, and puts a chosen one beside our design', async () => {
  const charted = routes()
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('tab', { name: 'Published BEQs' }))
  const list = await screen.findByRole('list', { name: 'Published BEQs' })
  expect(within(list).getAllByRole('listitem')).toHaveLength(1)
  expect(screen.getByText(/Matched by TMDB id/)).toBeInTheDocument()
  await userEvent.click(screen.getByLabelText('Include other tracks and editions'))
  expect(within(list).getAllByRole('listitem')).toHaveLength(2)
  expect(within(list).getByText(/another track/)).toBeInTheDocument()
  await userEvent.click(within(list).getByRole('checkbox', { name: /aron7awol/ }))
  await waitFor(() => expect(charted).toEqual(['0']))
  await waitFor(() => expect(drawn.at(-1)!.series.map((s) => s.label))
    .toContain("Average audio track (all channels mixed) under aron7awol's BEQ"))
  expect(await screen.findByLabelText("How aron7awol's BEQ compares")).toHaveTextContent('boosts up to 10.0 dB at 0.0 Hz')
  expect(within(list).getByRole('link', { name: 'page' })).toHaveAttribute('href', 'https://example.org/aron7awol/1/#alien')
})

it('says when the catalogue has none', async () => {
  routes([])
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('tab', { name: 'Published BEQs' }))
  expect(await screen.findByText('The catalogue has no BEQ for this title.')).toBeInTheDocument()
})

it('says when the catalogue cannot be reached', async () => {
  routes()
  server.use(http.get('/v1/titles/a/catalogue', () => json({ title: 'Catalogue unavailable', status: 502,
                                                              detail: 'the catalogue could not be read' }, 502)))
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('tab', { name: 'Published BEQs' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('the catalogue could not be read')
})
