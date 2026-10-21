// Editing a design's filters in the browser (design/web-parity.md WP8) against a mocked service: the design's sections
// in a table, the edit drawn beside it, what cannot be saved said, and the edit saved as a new design run.
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { editedSeries } from '../components/FilterEditor'
import { candidate, chart, index, METADATA_FIELDS, review } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

const drawn = vi.hoisted(() => [] as { series: { label?: string }[] }[])
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
const SHELF = { filters: [{ _type: 'LowShelf', fs: 1000, fc: 20, q: 0.7, gain: 4.5, count: 2 }] }

function routes(over: Parameters<typeof review>[0] = {}) {
  const saved: unknown[] = []
  server.use(
    statusRoute(status({ index: index() })),
    http.get('/v1/titles/a/review', () => json(review({ candidates: [candidate(0, { filters: SHELF }), candidate(1)], ...over }))),
    http.get('/v1/metadata/fields', () => json(METADATA_FIELDS)),
    http.get('/v1/titles/a/chart', () => json(chart(0))),
    http.put('/v1/titles/a/edited', async ({ request }) => {
      saved.push(await request.json())
      return json(review({ variant_id: 'v2', candidates: [candidate(0, { method: 'edited' })], rejected: [] }))
    }),
  )
  return saved
}

it('draws an edit as its response and the track under it', () => {
  const measured = chart(null).series
  const drawnSeries = editedSeries(measured, [{ type: 'low_shelf', freq_hz: 20, q: 0.7, gain_db: 6, count: 1 }], 1000)
  expect(drawnSeries.map((s) => s.name)).toEqual(['Average audio track (all channels mixed) under the edit',
                                                  'Peak audio track (all channels mixed) under the edit', 'The edit'])
  expect(drawnSeries[0]!.y[1]!).toBeGreaterThan(drawnSeries[0]!.y[4]!)   // lifted at 10 Hz, not at 80
})

it('edits the highlighted design and saves it as a new design run', async () => {
  const saved = routes()
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('button', { name: 'Edit filters…' }))
  const editor = screen.getByRole('region', { name: 'Edit design 1' })
  expect(within(editor).getByLabelText('Section 1 frequency')).toHaveValue('20')
  expect(within(editor).getByLabelText('Section 1 stacked')).toHaveValue('2')
  await waitFor(() => expect(drawn.at(-1)!.series.map((s) => s.label)).toContain('The edit'))
  const gain = within(editor).getByLabelText('Section 1 gain')
  await userEvent.clear(gain)
  await userEvent.type(gain, '6')
  await userEvent.click(within(editor).getByRole('button', { name: 'Add a peak' }))
  await userEvent.click(within(editor).getByRole('button', { name: 'Move section 2 up' }))
  await userEvent.click(within(editor).getByRole('button', { name: 'Save as a new design' }))
  await waitFor(() => expect(saved).toEqual([{ digest: 'd1', source: 0, filters: [
    { type: 'peaking_eq', freq_hz: 30, q: 1, gain_db: 0, count: 1 },
    { type: 'low_shelf', freq_hz: 20, q: 0.7, gain_db: 6, count: 2 }] }]))
  await waitFor(() => expect(screen.queryByRole('region', { name: 'Edit design 1' })).not.toBeInTheDocument())
})

it('says what cannot be saved', async () => {
  routes()
  renderApp('/titles/a')
  await userEvent.click(await screen.findByRole('button', { name: 'Edit filters…' }))
  const editor = screen.getByRole('region', { name: 'Edit design 1' })
  const q = within(editor).getByLabelText('Section 1 Q')
  await userEvent.clear(q)
  await userEvent.type(q, '0')
  expect(within(editor).getByRole('alert')).toHaveTextContent('Q must be above 0')
  expect(within(editor).getByRole('button', { name: 'Save as a new design' })).toBeDisabled()
  await userEvent.click(within(editor).getByRole('button', { name: 'Remove section 1' }))
  expect(within(editor).getByRole('alert')).toHaveTextContent('at least one section')
})

it('is not offered on an accepted title', async () => {
  routes({ status: 'accepted', chosen_index: 0 })
  renderApp('/titles/a')
  expect(await screen.findByRole('button', { name: 'Edit filters…' })).toBeDisabled()
})

it('says on the page, and in the editor, when a project was edited by hand since the design', async () => {
  routes()
  server.use(http.get('/v1/titles/a/projects', () => json({
    projects: [{ kind: 'mono', name: 'a.mono.beq', exists: true, edited: true, flat: false, error: '' }],
    summary: 'Modified since design: mono project', warn: true, note: 'What you save in the project is what gets published' })))
  renderApp('/titles/a')
  expect(await screen.findByLabelText('Projects')).toHaveTextContent('Modified since design: mono project')
  await userEvent.click(screen.getByRole('button', { name: 'Edit filters…' }))
  expect(within(screen.getByRole('region', { name: 'Edit design 1' })).getByRole('note'))
    .toHaveTextContent('what is saved in the project, not this edit')
})
