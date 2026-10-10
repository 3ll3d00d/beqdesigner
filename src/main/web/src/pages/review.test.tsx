// The Titles and Review screens (W5) against a mocked service. uPlot draws on a canvas jsdom does not have, so it is
// replaced by a recorder of what it was given.
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { chartData, FREQUENCY_RANGE, resample } from '../components/MagnitudeChart'
import { chart, index, review, runJob, title } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

const drawn = vi.hoisted(() => [] as { options: Record<string, unknown>; data: unknown[] }[])
vi.mock('uplot', () => ({
  default: class {
    constructor(options: Record<string, unknown>, data: unknown[]) {
      drawn.push({ options, data })
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

function reviewRoutes(over: Parameters<typeof review>[0] = {}, nextId: string | null = 'b') {
  const decisions: unknown[] = []
  const charts: (string | null)[] = []
  const after: (string | null)[] = []
  server.use(
    statusRoute(status({ index: index() })),
    http.get('/v1/titles/a/review', () => json(review(over))),
    http.get('/v1/titles/a/chart', ({ request }) => {
      const asked = new URL(request.url).searchParams.get('candidate')
      charts.push(asked)
      return json(chart(asked === null ? null : Number(asked)))
    }),
    http.post('/v1/titles/a/decision', async ({ request }) => {
      const body = await request.json() as Record<string, unknown>
      decisions.push(body)
      const accepted = body.decision === 'accept'
      return json(review({ ...over, status: accepted ? 'accepted' : 'rejected', status_text: accepted ? 'Accepted' : 'Rejected',
                           chosen_index: (body.candidate as number) ?? null }))
    }),
    http.get('/v1/review/next', ({ request }) => {
      after.push(new URL(request.url).searchParams.get('after'))
      return nextId ? json({ id: nextId, title: nextId.toUpperCase() }) : new HttpResponse(null, { status: 204 })
    }),
    http.get('/v1/titles/b/review', () => json(review({ id: 'b', title: 'Brazil' }))),
    http.get('/v1/titles/b/chart', () => json(chart(0))),
  )
  return { decisions, charts, after }
}

describe('the chart', () => {
  it('puts every curve on the first one\'s frequencies, within the drawn range', () => {
    expect(resample([10, 20, 40], [0, 10, 30], [10, 15, 30, 50])).toEqual([0, 5, 20, 30])
    expect(chartData(chart(0).series)).toEqual([[10, 20, 40, 80], [0, 0, 0, 0], [10, 10, 10, 10], [4, 3, 1, 0]])
  })

  it('draws 1 to 160 Hz on a linear axis, as the desktop does', () => {
    expect(FREQUENCY_RANGE).toEqual([1, 160])
    const x = [0, 0.5, 1, 80, 160, 161, 1000]
    const data = chartData([{ name: 'a', kind: 'average', filtered: false, x, y: x.map((_, i) => i) }])
    expect(data).toEqual([[1, 80, 160], [2, 3, 4]])
  })
})

describe('titles', () => {
  it('lists the titles the filter in the address bar selects, each linked to its review in the same list', async () => {
    let asked: URLSearchParams | undefined
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', ({ request }) => {
                 asked = new URL(request.url).searchParams
                 return json({ total: 2, offset: 0, limit: 100, titles: [title('a'), title('b', { confidence: null })] })
               }))
    const router = renderApp('/titles?needs=review&needs=design&kind=movie')

    expect(await screen.findByText('A')).toHaveAttribute('href', '/titles/a?needs=review&needs=design&kind=movie')
    expect(asked?.getAll('needs')).toEqual(['review', 'design'])
    expect(asked?.get('kind')).toBe('movie')
    expect(screen.getByText(/2 titles \(needing review or design, movies\)/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('checkbox', { name: 'Design' }))
    await waitFor(() => expect(router.state.location.search).toBe('?needs=review&kind=movie'))
  })

  it('opens the next title waiting, or says there is none', async () => {
    let none = false
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', () => json({ total: 0, offset: 0, limit: 100, titles: [] })),
               http.get('/v1/review/next', () => (none ? new HttpResponse(null, { status: 204 }) : json({ id: 'a', title: 'Alien' }))),
               http.get('/v1/titles/a/review', () => json(review())),
               http.get('/v1/titles/a/chart', () => json(chart(0))))
    const router = renderApp('/titles?needs=review')
    await userEvent.click(await screen.findByRole('button', { name: 'Review next waiting' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/titles/a'))
    expect(router.state.location.search).toBe('?needs=review')

    none = true
    const again = renderApp('/titles')
    await userEvent.click((await screen.findAllByRole('button', { name: 'Review next waiting' })).at(-1)!)
    expect(await screen.findByText('No title in this list is waiting for a decision.')).toBeInTheDocument()
    expect(again.state.location.pathname).toBe('/titles')
  })

  it('pages through a long list', async () => {
    const offsets: (string | null)[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', ({ request }) => {
                 offsets.push(new URL(request.url).searchParams.get('offset'))
                 return json({ total: 250, offset: 0, limit: 100, titles: [title('a')] })
               }))
    renderApp('/titles')
    await userEvent.click(await screen.findByRole('button', { name: 'Next' }))
    await waitFor(() => expect(offsets).toContain('100'))
    expect(screen.getByText('101–200 of 250')).toBeInTheDocument()
  })

  it('does not offer publish or commit unless the service allows them', async () => {
    server.use(statusRoute(status({ index: index(), repository_writes: true, repositories_configured: false })),
               http.get('/v1/titles', () => json({ total: 0, offset: 0, limit: 100, titles: [] })))
    renderApp('/titles')
    expect(await screen.findByText('The profile names no filter repository to publish to.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Publish accepted…' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Commit published…' })).toBeDisabled()
  })

  it('publishes the accepted titles in the list after saying how many', async () => {
    const bodies: unknown[] = []
    server.use(statusRoute(status({ index: index(), repository_writes: true, repositories_configured: true })),
               http.get('/v1/titles', () => json({ total: 0, offset: 0, limit: 100, titles: [] })),
               http.post('/v1/plan', async ({ request }) => {
                 bodies.push(await request.json())
                 return json({ through: 'publish', label: 'Publish 3', skipped: [],
                               planned: ['a', 'b', 'c'].map((id) => ({ id, title: id, stages: ['publish'] })) })
               }),
               http.post('/v1/jobs/run', async ({ request }) => {
                 bodies.push(await request.json())
                 return json(runJob({ id: 'pub', state: 'queued', result: null }), 202)
               }),
               http.get('/v1/jobs/pub', () => json(runJob({ id: 'pub', state: 'queued', result: null }))),
               http.get('/v1/jobs/pub/events', () => new HttpResponse(new ReadableStream())))
    const router = renderApp('/titles?kind=movie')

    await waitFor(() => expect(screen.getByRole('button', { name: 'Publish accepted…' })).toBeEnabled())
    await userEvent.click(screen.getByRole('button', { name: 'Publish accepted…' }))
    const confirm = await screen.findByRole('group', { name: 'Confirm' })
    expect(within(confirm).getByText("Publish 3 accepted titles to the catalogue's filter repository?")).toBeInTheDocument()
    await userEvent.click(within(confirm).getByRole('button', { name: 'Publish' }))

    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/pub'))
    const expected = { filter: { needs: ['publish'], kind: 'movie', new_since_scan: false }, through: 'publish',
                       scan_first: false, retry_failed: false }
    expect(bodies).toEqual([expected, expected])
  })
})

describe('reviewing a title', () => {
  it('shows its designs, the rejected ones apart, commentary, metadata and the chart of the top pick', async () => {
    const { charts } = reviewRoutes()
    renderApp('/titles/a')

    expect(await screen.findByRole('heading', { name: 'Alien (1979)' })).toBeInTheDocument()
    const designs = screen.getByRole('radiogroup', { name: 'Designs' })
    expect(within(designs).getAllByRole('radio')).toHaveLength(2)
    expect(within(designs).getAllByRole('radio')[0]).toBeChecked()
    expect(within(screen.getByRole('list', { name: 'Rejected designs' })).getByText(/1 reason/)).toBeInTheDocument()
    expect(screen.getByText('top pick')).toBeInTheDocument()
    expect(screen.getByText('shelf at 20 Hz')).toBeInTheDocument()
    expect(screen.getByText('DTS-HD MA 5.1')).toBeInTheDocument()
    expect(screen.getByText(/Designed by rolloff \(beqforge 0.2.0\)/)).toBeInTheDocument()
    await waitFor(() => expect(drawn.length).toBeGreaterThan(0))
    expect(charts).toContain('0')
    const options = drawn.at(-1)!.options as { scales: { x: { distr?: number; range: number[] } }; series: { dash?: number[] }[] }
    expect(options.scales.x.distr).toBeUndefined()              // linear frequency
    expect(options.scales.x.range).toEqual([1, 160])
    expect(options.series.map((s) => !!s.dash)).toEqual([false, true, true, false])   // before dashed, after solid
  })

  it('accepts the design highlighted, then opens the next title waiting in the same list', async () => {
    const { decisions, charts, after } = reviewRoutes()
    const router = renderApp('/titles/a?needs=review')

    await userEvent.click(await screen.findByText(/^2\./))
    await waitFor(() => expect(charts).toContain('1'))
    expect(screen.getByText('alternative')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Accept design 2' }))

    await waitFor(() => expect(router.state.location.pathname).toBe('/titles/b'))
    expect(router.state.location.search).toBe('?needs=review')
    expect(decisions).toEqual([{ decision: 'accept', candidate: 1, digest: 'd1', override_rejection: false }])
    expect(after).toEqual(['a'])
    expect(await screen.findByRole('heading', { name: 'Brazil (1979)' })).toBeInTheDocument()
  })

  it('says when no other title is waiting after a decision, and shows the title as decided', async () => {
    reviewRoutes({}, null)
    renderApp('/titles/a')
    await userEvent.click(await screen.findByRole('button', { name: 'Reject' }))
    expect(await screen.findByText('Rejected. No other title in this list is waiting for a decision.')).toBeInTheDocument()
    expect(screen.getByText('Rejected', { selector: '.badge' })).toBeInTheDocument()
  })

  it('asks before accepting a design the designer rejected, and sends the override', async () => {
    const { decisions } = reviewRoutes()
    renderApp('/titles/a')

    await userEvent.click(await screen.findByText(/^3\./))
    await userEvent.click(screen.getByRole('button', { name: 'Accept design 3' }))
    const asking = screen.getByRole('group', { name: 'Override' })
    expect(within(asking).getByText('introduces a cliff of 53 dB/oct at 17 Hz')).toBeInTheDocument()
    expect(decisions).toEqual([])
    await userEvent.click(within(asking).getByRole('button', { name: 'Accept anyway' }))
    await waitFor(() => expect(decisions).toEqual([{ decision: 'accept', candidate: 2, digest: 'd1', override_rejection: true }]))
  })

  it('says why a decision was refused and shows the title as it is now', async () => {
    let reads = 0
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles/a/review', () => {
                 reads += 1
                 return json(review(reads > 1 ? { digest: 'd2', status_text: 'Redesigned' } : {}))
               }),
               http.get('/v1/titles/a/chart', () => json(chart(0))),
               http.post('/v1/titles/a/decision', () => json({ title: 'Changed since it was read', status: 409,
                 detail: 'Not accepted: the design of this title changed while it was open. Look at it again.' }, 409)))
    renderApp('/titles/a')

    await userEvent.click(await screen.findByRole('button', { name: 'Accept design 1' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Changed since it was read: Not accepted: the design')
    await waitFor(() => expect(reads).toBe(2))
    expect(await screen.findByText('Redesigned')).toBeInTheDocument()
  })

  it('holds Accept back with the reason, and lists what the metadata lacks', async () => {
    reviewRoutes({ metadata_problems: ['audio types are missing'],
                   blocked: { accept: 'Fill in the missing metadata first (Metadata tab): audio types are missing.', reject: '' } })
    renderApp('/titles/a')
    expect(await screen.findByRole('button', { name: 'Accept design 1' })).toBeDisabled()
    expect(screen.getByText(/^Accept: Fill in the missing metadata first/)).toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent('audio types are missing')
    expect(screen.getByRole('button', { name: 'Reject' })).toBeEnabled()
  })

  it('decides from the keyboard: a digit highlights a design, R rejects, S skips, A does nothing when not offered', async () => {
    const { decisions, charts } = reviewRoutes({ blocked: { accept: 'A run is working on this title now.', reject: '' } })
    const router = renderApp('/titles/a')
    await screen.findByRole('heading', { name: 'Alien (1979)' })

    fireEvent.keyDown(window, { key: '2' })
    await waitFor(() => expect(charts).toContain('1'))
    fireEvent.keyDown(window, { key: 'a' })
    expect(decisions).toEqual([])
    fireEvent.keyDown(window, { key: 'r' })
    await waitFor(() => expect(decisions).toEqual([{ decision: 'reject', digest: 'd1', override_rejection: false }]))
    await waitFor(() => expect(router.state.location.pathname).toBe('/titles/b'))
    fireEvent.keyDown(window, { key: 's' })
  })

  it('leaves the keys alone while a person types', async () => {
    const { decisions } = reviewRoutes()
    renderApp('/titles/a')
    await screen.findByRole('heading', { name: 'Alien (1979)' })
    const box = document.createElement('input')
    document.body.appendChild(box)
    fireEvent.keyDown(box, { key: 'r' })
    expect(decisions).toEqual([])
    box.remove()
  })

  it('says a title has no design yet and offers to run it', async () => {
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles/x/review', () => json({ title: 'Not designed', status: 409,
                                                             detail: 'x has no design to review yet' }, 409)))
    renderApp('/titles/x')
    expect(await screen.findByRole('alert')).toHaveTextContent('Not designed: x has no design to review yet')
    expect(screen.getByText('Run it through design…')).toHaveAttribute('href', '/jobs/new?ids=x')
  })
})
