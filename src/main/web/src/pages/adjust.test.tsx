// Redesigning with adjustments and the title's design runs (design/web-parity.md WP7) against a mocked service: the
// editor built from the kinds the designer lists, what is sent, the designer's answers, and earlier runs drawn, used
// and discarded.
import { act, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { askProblem, blankAsk, describeAsk } from '../components/AdjustmentsEditor'
import { outcomeText } from '../components/VariantsPanel'
import { chart, index, METADATA_FIELDS, review, runJob } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

const drawn = vi.hoisted(() => [] as { series: { label?: string; dash?: number[] }[];
                                       hooks: { setSelect: ((u: unknown) => void)[]; draw: ((u: unknown) => void)[] } }[])
vi.mock('uplot', () => ({
  default: class {
    over = document.createElement('div')
    cursor = { left: -1 }
    constructor(options: (typeof drawn)[number]) {
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
const KINDS = {
  exclude_band: { description: 'leave a band alone', params: { low_hz: { type: 'number', unit: 'Hz', minimum: 1 },
                                                               high_hz: { type: 'number', unit: 'Hz' } } },
  max_boost: { description: 'cap the boost', params: { db: { type: 'number', unit: 'dB', default: 15, minimum: 1 } } },
  note: { description: 'say it in words' },
}
const ASKED = [{ id: 'a1', kind: 'max_boost', params: { db: 10 }, text: null }]
const V1 = { variant_id: 'v1', designed_at: '2026-10-01T10:00:00Z', designer: 'rolloff', designer_build: 'beqforge 0.2.8',
             design_version: 4, designs: 2, declined: false, adjusted_with: [], adjustment_outcomes: [] }

function routes(over: Parameters<typeof review>[0] = {}, kinds: Record<string, unknown> = KINDS) {
  const sent: Record<string, unknown>[] = []
  const charts: URLSearchParams[] = []
  const used: unknown[] = []
  server.use(
    statusRoute(status({ index: index(), designer: { name: 'rolloff', reachable: true, detail: '', checked_at: null,
                                                      endpoints: [], design_disagreement: [], adjustment_kinds: kinds } })),
    http.get('/v1/titles/a/review', () => json(review(over))),
    http.get('/v1/metadata/fields', () => json(METADATA_FIELDS)),
    http.get('/v1/titles/a/chart', ({ request }) => {
      const query = new URL(request.url).searchParams
      charts.push(query)
      const variant = query.get('variant')
      const base = chart(0)
      return json(variant ? { ...base, series: base.series.map((s) => (s.filtered ? { ...s, variant, name: `${s.name} (${variant})` } : s)) }
        : base)
    }),
    http.post('/v1/revise', async ({ request }) => {
      sent.push(await request.json() as Record<string, unknown>)
      return json({ revised: [], failed: [], job: runJob() })
    }),
    http.get('/v1/jobs/job-1', () => json(runJob())), http.get('/v1/jobs/job-1/log', () => json([])),
    http.post('/v1/titles/a/variants/:v/use', async ({ params, request }) => {
      used.push([params.v, await request.json()])
      return json(review({ ...over, variant_id: String(params.v), variants: [] }))
    }),
    http.delete('/v1/titles/a/variants/:v', ({ params }) => {
      used.push(['discard', params.v])
      return json(review({ ...over, variants: [] }))
    }),
  )
  return { sent, charts, used }
}

describe('the words', () => {
  it('says an ask, what is wrong with one, and what became of it', () => {
    expect(describeAsk({ kind: 'exclude_band', params: { low_hz: 28, high_hz: 34 } })).toBe('ignore 28-34 Hz')
    expect(describeAsk({ kind: 'goal_tilt', params: { db_per_octave: 1 } })).toBe('tilt the goal +1 dB/oct')
    expect(describeAsk({ kind: 'x-beqforge-strategy', params: { names: 'parametric' } }))
      .toBe('x-beqforge-strategy (names parametric)')
    expect(blankAsk('max_boost', KINDS.max_boost)).toEqual({ kind: 'max_boost', params: { db: 15 }, text: null })
    expect(askProblem({ kind: 'max_boost', params: { db: 0 } }, KINDS.max_boost)).toBe('db is at least 1')
    expect(askProblem({ kind: 'exclude_band', params: { low_hz: 40, high_hz: 30 } }, KINDS.exclude_band))
      .toMatch(/low end must be below/)
    expect(askProblem({ kind: 'note', params: {}, text: ' ' }, KINDS.note)).toMatch(/Say what/)
    const outcomes = [{ id: 'a1', outcome: 'adapted', applied: { db: 12 }, reason: '10 left the rolloff' }]
    expect(outcomeText({ id: 'a1' }, outcomes)).toBe('adapted, as db 12: 10 left the rolloff')
    expect(outcomeText({ id: 'a2' }, outcomes)).toMatch(/not acknowledged/)
  })
})

describe('redesigning with adjustments', () => {
  it('starts from what the title is asked, offers the designer\'s kinds and presets, and sends the list', async () => {
    const { sent } = routes({ adjustments: ASKED, adjusted_with: ASKED })
    renderApp('/titles/a')
    await userEvent.click(await screen.findByRole('button', { name: 'Revise…' }))
    const dialog = screen.getByRole('group', { name: 'Revise' })
    await userEvent.click(await within(dialog).findByLabelText(/Ask the designer to change something/))
    const editor = within(dialog).getByRole('group', { name: 'Adjustments' })
    expect(within(editor).getByLabelText(/^db/)).toHaveValue('10')
    await userEvent.click(within(editor).getByRole('button', { name: 'Ignore the chart\'s range' }))
    expect(within(editor).queryByRole('button', { name: /Bass boost/ })).not.toBeInTheDocument()   // goal_tilt not listed
    await userEvent.click(within(editor).getByRole('button', { name: 'Add an adjustment' }))
    await userEvent.selectOptions(within(editor).getAllByLabelText('Kind').at(-1)!, 'note')
    expect(within(dialog).getByRole('button', { name: 'Redesign' })).toBeDisabled()          // the note says nothing yet
    await userEvent.type(within(editor).getAllByLabelText('Note').at(-1)!, 'the logo is not the film')
    await userEvent.click(within(dialog).getByRole('button', { name: 'Redesign' }))
    await waitFor(() => expect(sent).toHaveLength(1))
    expect(sent[0]).toEqual({ ids: ['a'], to: 'design', reason: '', run: true, adjustments: [
      { kind: 'max_boost', params: { db: 10 }, text: null },
      { kind: 'exclude_band', params: { low_hz: 1, high_hz: 160 } },
      { kind: 'note', params: {}, text: 'the logo is not the film' }] })
  })

  it('takes a band to ignore from a drag across the chart, and shades the bands asked', async () => {
    const { sent } = routes()
    renderApp('/titles/a')
    await userEvent.click(await screen.findByRole('button', { name: 'Revise…' }))
    const dialog = screen.getByRole('group', { name: 'Revise' })
    await userEvent.click(await within(dialog).findByLabelText(/Ask the designer to change something/))
    await userEvent.click(within(dialog).getByRole('button', { name: 'Pick a band on the chart' }))
    expect(screen.getByText(/Drag across the chart to choose the band/)).toBeInTheDocument()
    const u = { select: { left: 28.04, top: 0, width: 5.96, height: 300 }, over: { clientHeight: 300 },
                posToVal: (px: number) => px, setSelect: () => {} }
    act(() => drawn.at(-1)!.hooks.setSelect[0]!(u))
    await waitFor(() => expect(within(dialog).getByLabelText(/^low_hz/)).toHaveValue('28'))
    expect(within(dialog).getByLabelText(/^high_hz/)).toHaveValue('34')
    expect(screen.queryByText(/Drag across the chart to choose the band/)).not.toBeInTheDocument()
    const filled: number[][] = []
    const ctx = { save() {}, restore() {}, fillRect: (...r: number[]) => filled.push(r), fillStyle: '' }
    await waitFor(() => {
      drawn.at(-1)!.hooks.draw[0]!({ ctx, bbox: { top: 0, height: 100 }, valToPos: (v: number) => v })
      expect(filled.at(-1)).toEqual([28, 0, 6, 100])
    })
    await userEvent.click(within(dialog).getByRole('button', { name: 'Redesign' }))
    await waitFor(() => expect(sent.at(-1)!.adjustments).toEqual([{ kind: 'exclude_band', params: { low_hz: 28, high_hz: 34 } }]))
  })

  it('asks nothing of a designer that lists no kinds', async () => {
    const { sent } = routes({}, {})
    renderApp('/titles/a')
    await userEvent.click(await screen.findByRole('button', { name: 'Revise…' }))
    const dialog = screen.getByRole('group', { name: 'Revise' })
    expect(within(dialog).getByLabelText(/Ask the designer to change something/)).toBeDisabled()
    expect(within(dialog).getByText(/lists no adjustments it takes/)).toBeInTheDocument()
    await userEvent.click(within(dialog).getByRole('button', { name: 'Redesign' }))
    await waitFor(() => expect(sent).toEqual([{ ids: ['a'], to: 'design', reason: '', run: true }]))
  })
})

describe('the design runs', () => {
  const answered = { variant_id: 'v2', adjustments: ASKED, adjusted_with: ASKED, variants: [V1],
                     adjustment_outcomes: [{ id: 'a1', outcome: 'declined', applied: null, reason: 'it would cut the rolloff' }] }

  it('show what the current one was asked and how the designer answered', async () => {
    routes(answered)
    renderApp('/titles/a')
    const panel = await screen.findByRole('region', { name: 'Design runs' })
    expect(panel).toHaveTextContent('boost at most 10 dB — declined: it would cut the rolloff')
    expect(panel).toHaveTextContent('v1')
    expect(panel).toHaveTextContent('asked nothing')
  })

  it('draw an earlier one dashed on the chart, use it, and discard one after asking', async () => {
    const { charts, used } = routes(answered)
    renderApp('/titles/a')
    const panel = await screen.findByRole('region', { name: 'Design runs' })
    await userEvent.click(within(panel).getByRole('checkbox', { name: /v1/ }))
    await waitFor(() => expect(charts.some((q) => q.get('variant') === 'v1' && q.get('candidates') === 'all')).toBe(true))
    await waitFor(() => expect(drawn.at(-1)!.series.some((s) => s.label?.endsWith('(v1)') && !!s.dash)).toBe(true))
    await userEvent.click(within(panel).getByRole('button', { name: 'Discard…' }))
    await userEvent.click(within(panel).getByRole('button', { name: 'Keep it' }))
    await userEvent.click(within(panel).getByRole('button', { name: 'Use this one' }))
    await waitFor(() => expect(used).toEqual([['v1', { digest: 'd1' }]]))
  })

  it('are not used again on a decided title', async () => {
    routes({ ...answered, status: 'accepted', chosen_index: 0 })
    renderApp('/titles/a')
    const panel = await screen.findByRole('region', { name: 'Design runs' })
    expect(within(panel).getByRole('button', { name: 'Use this one' })).toBeDisabled()
  })
})
