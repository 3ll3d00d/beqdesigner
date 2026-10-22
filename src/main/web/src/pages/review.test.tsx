// The Titles and Review screens (W5) against a mocked service. uPlot draws on a canvas jsdom does not have, so it is
// replaced by a recorder of what it was given.
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { chartData, FREQUENCY_RANGE, resample } from '../components/MagnitudeChart'
import { candidate, chart, index, METADATA_FIELDS, review, runJob, title } from '../test/fixtures'
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
    http.get('/v1/metadata/fields', () => json(METADATA_FIELDS)),
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

  it('sorts by a column from its heading: ascending, descending, then back to work-list order', async () => {
    const asked: URLSearchParams[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', ({ request }) => {
                 asked.push(new URL(request.url).searchParams)
                 return json({ total: 1, offset: 0, limit: 100, titles: [title('a')] })
               }))
    const router = renderApp('/titles?needs=review&offset=100')
    const heading = await screen.findByRole('columnheader', { name: 'Confidence' })
    expect(heading).toHaveAttribute('aria-sort', 'none')

    await userEvent.click(within(heading).getByRole('button'))
    await waitFor(() => expect(router.state.location.search).toBe('?needs=review&sort=confidence'))   // back to page 1
    expect(heading).toHaveAttribute('aria-sort', 'ascending')
    await waitFor(() => expect(asked.at(-1)?.get('sort')).toBe('confidence'))
    expect(asked.at(-1)?.get('descending')).toBe('false')
    // the review it links to keeps the order, so the list it goes back to is the same
    expect(screen.getByText('A')).toHaveAttribute('href', '/titles/a?needs=review&sort=confidence')

    await userEvent.click(within(heading).getByRole('button'))
    await waitFor(() => expect(router.state.location.search).toBe('?needs=review&sort=confidence&descending=true'))
    expect(heading).toHaveAttribute('aria-sort', 'descending')
    await waitFor(() => expect(asked.at(-1)?.get('descending')).toBe('true'))

    await userEvent.click(within(heading).getByRole('button'))
    await waitFor(() => expect(router.state.location.search).toBe('?needs=review'))
    expect(asked.at(-1)?.has('sort')).toBe(false)
  })

  it('filters from the row under the headings, a typed one once typing pauses', async () => {
    const asked: URLSearchParams[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', ({ request }) => {
                 const query = new URL(request.url).searchParams
                 asked.push(query)
                 return query.get('year') === '20'
                   ? json({ title: 'Invalid', status: 422, detail: 'year: not a year' }, 422)
                   : json({ total: 1, offset: 0, limit: 100, titles: [title('a')] })
               }))
    const router = renderApp('/titles')
    // a typed filter is sent once typing pauses (300 ms): under a loaded parallel run that and the render can pass 1 s
    const typing = { timeout: 5000 }

    await userEvent.type(await screen.findByLabelText('Filter title'), 'star wars')
    await waitFor(() => expect(router.state.location.search).toBe('?match=star+wars'), typing)
    expect(asked.some((q) => q.get('match') === 'star')).toBe(false)   // nothing sent mid-word
    expect(screen.getByLabelText('Filter title')).toHaveValue('star wars')

    await userEvent.type(screen.getByLabelText('Filter detail'), 'Rolloff')
    await userEvent.selectOptions(screen.getByLabelText('Filter flags'), 'Gone')
    await userEvent.selectOptions(screen.getByLabelText('Filter kind'), 'tv')
    await userEvent.click(within(screen.getByRole('group', { name: 'Needs' })).getByRole('checkbox', { name: 'Review' }))
    await waitFor(() => expect(router.state.location.search)
      .toBe('?needs=review&kind=tv&match=star+wars&detail=Rolloff&flags=Gone'), typing)
    await waitFor(() => expect(asked.at(-1)?.get('detail')).toBe('Rolloff'), typing)
    expect(asked.at(-1)?.getAll('flags')).toEqual(['Gone'])
    expect(screen.getByText(/detail matching “Rolloff”, flagged Gone/)).toBeInTheDocument()

    // a filter the service refuses leaves the row there to correct it
    await userEvent.type(screen.getByLabelText('Filter year'), '20')
    await waitFor(() => expect(asked.at(-1)?.get('year')).toBe('20'), typing)
    expect(await screen.findByRole('alert', {}, typing)).toHaveTextContent('not a year')
    await userEvent.type(screen.getByLabelText('Filter year'), '20')
    await waitFor(() => expect(asked.at(-1)?.get('year')).toBe('2020'), typing)
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument(), typing)
  }, 30_000)   // types a great deal and waits for five pauses: past vitest's 5 s default under a loaded run

  it('filters on whether the designer declined', async () => {
    const asked: URLSearchParams[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', ({ request }) => {
                 asked.push(new URL(request.url).searchParams)
                 return json({ total: 1, offset: 0, limit: 100, titles: [title('a')] })
               }))
    const router = renderApp('/titles')

    await userEvent.selectOptions(await screen.findByLabelText('Designer'), 'Declined')
    await waitFor(() => expect(router.state.location.search).toBe('?declined=true'))
    await waitFor(() => expect(asked.at(-1)?.get('declined')).toBe('true'))
    expect(screen.getByText(/declined by the designer/)).toBeInTheDocument()

    await userEvent.selectOptions(screen.getByLabelText('Designer'), 'Not declined')
    await waitFor(() => expect(asked.at(-1)?.get('declined')).toBe('false'))
    await userEvent.selectOptions(screen.getByLabelText('Designer'), 'Any')
    await waitFor(() => expect(router.state.location.search).toBe(''))
    await waitFor(() => expect(asked.at(-1)?.has('declined')).toBe(false))
  })

  it('filters on the design version, and redesigns what is out of date in one job', async () => {
    const asked: URLSearchParams[] = []
    const sent: unknown[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', ({ request }) => {
                 asked.push(new URL(request.url).searchParams)
                 return json({ total: 2, offset: 0, limit: 100, titles: [title('a'), title('b')] })
               }),
               http.post('/v1/jobs/run', async ({ request }) => {
                 sent.push(await request.json())
                 return json(runJob({ id: 'j1' }), 202)
               }),
               http.get('/v1/jobs/j1', () => json(runJob({ id: 'j1' }))))
    const router = renderApp('/titles')

    await screen.findByLabelText('Design version')
    expect(screen.queryByRole('button', { name: 'Redesign out of date…' })).not.toBeInTheDocument()
    await userEvent.selectOptions(screen.getByLabelText('Design version'), 'Out of date')
    await waitFor(() => expect(router.state.location.search).toBe('?out_of_date=true'))
    await waitFor(() => expect(asked.at(-1)?.get('out_of_date')).toBe('true'))
    expect(screen.getByText(/designed out of date/)).toBeInTheDocument()

    await userEvent.click(await screen.findByRole('button', { name: 'Redesign out of date…' }))
    const confirm = screen.getByRole('group', { name: 'Confirm redesign' })
    expect(confirm).toHaveTextContent('Send 2 titles back to the designer?')
    await userEvent.click(within(confirm).getByRole('button', { name: 'Redesign' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/j1'))
    expect(sent).toEqual([{ filter: { out_of_date: true, new_since_scan: false }, through: 'design', redesign: true, revise_to: 'design',
                            scan_first: false, retry_failed: false }])
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
                       scan_first: false, retry_failed: false, redesign: false, revise_to: 'design' }
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
    expect(screen.getByLabelText('Metadata')).toHaveTextContent('DTS-HD MA 5.1')
    expect(screen.getByText(/Designed by rolloff \(beqforge 0.2.0\)/)).toBeInTheDocument()
    await waitFor(() => expect(drawn.length).toBeGreaterThan(0))
    expect(charts).toContain('0')
    const options = drawn.at(-1)!.options as { scales: { x: { distr?: number; range: number[] } }; series: { dash?: number[] }[] }
    expect(options.scales.x.distr).toBeUndefined()              // linear frequency
    expect(options.scales.x.range).toEqual([1, 160])
    expect(options.series.map((s) => !!s.dash)).toEqual([false, true, true, false])   // before dashed, after solid
  })

  it('puts the metadata on one line under the heading, with genres by name', async () => {
    reviewRoutes({ metadata: { title: 'Alien', year: '1979', content_type: 'film', audio_types: ['DTS-HD MA 5.1', 'AC3 5.1'],
                               genres: [{ id: 27, name: 'Horror' }, { name: 'Science Fiction' }] } })
    renderApp('/titles/a')

    const line = await screen.findByLabelText('Metadata')
    expect(line).toHaveTextContent(/^film · DTS-HD MA 5.1, AC3 5.1 · Horror, Science Fiction$/)
    expect(screen.queryByText(/object Object/)).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Metadata' })).not.toBeInTheDocument()
    // straight under the title's heading, not in a card of its own beside the commentary
    expect(screen.getByRole('heading', { name: 'Alien (1979)' }).parentElement!.nextElementSibling).toBe(line)
    expect(screen.getByRole('heading', { name: 'Commentary' }).closest('section')).toHaveClass('commentary')
  })

  it('shows how well the highlighted design fits its target, when the designer measured it', async () => {
    reviewRoutes({ candidates: [candidate(0, { fit: '0.3 dB RMS over 5-212 Hz; 96% of the boost within ±0.5 dB; worst 1.1 dB short at 9 Hz' }),
                                candidate(1)] })
    renderApp('/titles/a')

    expect(await screen.findByText(/0\.3 dB RMS over 5-212 Hz; 96% of the boost within ±0\.5 dB; worst 1\.1 dB short at 9 Hz/))
      .toBeInTheDocument()
    await userEvent.click(screen.getByText(/^2\./))
    expect(screen.queryByText(/dB RMS over/)).not.toBeInTheDocument()   // the second design reported none
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
    expect(screen.getByRole('note')).toHaveTextContent('fix it in the Metadata panel')
    expect(screen.getByLabelText('What the metadata lacks')).toHaveTextContent('audio types are missing')
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


describe('revising one title', () => {
  function reviseRoutes() {
    const sent: unknown[] = []
    server.use(http.post('/v1/revise', async ({ request }) => {
      const body = await request.json() as { run: boolean; ids: string[] }
      sent.push(body)
      return json(body.run ? { revised: [], failed: [], job: runJob() }
        : { revised: [{ id: 'a', reverted: [], extract_invalidated: false }], failed: [], job: null })
    }), http.get('/v1/jobs/job-1', () => json(runJob())), http.get('/v1/jobs/job-1/log', () => json([])))
    return sent
  }

  it('redesigns now in one job, sending only this title, and opens the job', async () => {
    reviewRoutes({ status: 'accepted', chosen_index: 0 })
    const sent = reviseRoutes()
    const router = renderApp('/titles/a?source=films&needs=review')
    await userEvent.click(await screen.findByRole('button', { name: 'Revise…' }))
    expect(sent).toEqual([])
    const dialog = screen.getByRole('group', { name: 'Revise' })
    expect(dialog).toHaveTextContent('marked out of date')
    await userEvent.type(within(dialog).getByLabelText(/Why/), 'too much boost')
    await userEvent.click(within(dialog).getByRole('button', { name: 'Redesign' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/job-1'))
    expect(sent).toEqual([{ ids: ['a'], to: 'design', reason: 'too much boost', run: true }])
  })

  it('re-extracts, or reopens for review now, and says what came of it', async () => {
    reviewRoutes({ status: 'accepted', chosen_index: 0 })
    const sent = reviseRoutes()
    const router = renderApp('/titles/a')
    await userEvent.click(await screen.findByRole('button', { name: 'Revise…' }))
    let dialog = screen.getByRole('group', { name: 'Revise' })
    await userEvent.click(within(dialog).getByLabelText('Re-extract and redesign'))
    await userEvent.click(within(dialog).getByLabelText(/Extract & design now/))
    expect(dialog).toHaveTextContent('Nothing is redesigned, extracted or published now')
    await userEvent.click(within(dialog).getByRole('button', { name: 'Re-extract and redesign' }))
    expect(await screen.findByText('Sent back: 1 title.')).toBeInTheDocument()
    expect(sent.at(-1)).toEqual({ ids: ['a'], to: 'extract', reason: '', run: false })
    await userEvent.click(screen.getByRole('button', { name: 'Revise…' }))
    dialog = screen.getByRole('group', { name: 'Revise' })
    await userEvent.click(within(dialog).getByLabelText('Reopen for review'))
    expect(within(dialog).queryByLabelText(/Extract & design now/)).not.toBeInTheDocument()
    await userEvent.click(within(dialog).getByRole('button', { name: 'Reopen for review' }))
    await waitFor(() => expect(sent.at(-1)).toEqual({ ids: ['a'], to: 'review', reason: '', run: false }))
    expect(router.state.location.pathname).toBe('/titles/a')
  })

  it.each([
    [{ in_flight: true }, 'A run is working on this title.'],
    [{ status: 'published' as const }, 'Sending a published title back requires repository writes to be enabled.'],
  ])('blocks revise when %j', async (fields, reason) => {
    reviewRoutes(fields)
    renderApp('/titles/a')
    expect(await screen.findByRole('button', { name: 'Revise…' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Revise…' })).toHaveAttribute('title', reason)
  })

  it('shows submission errors and keeps the title open', async () => {
    reviewRoutes()
    server.use(http.post('/v1/revise', () => json({ title: 'Queue full', status: 409,
      detail: 'The job queue is full.' }, 409)))
    const router = renderApp('/titles/a')
    await userEvent.click(await screen.findByRole('button', { name: 'Revise…' }))
    await userEvent.click(screen.getByRole('button', { name: 'Redesign' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('The job queue is full.')
    expect(router.state.location.pathname).toBe('/titles/a')
  })
})

describe('choosing the audio stream', () => {
  const streams = { chosen: 1, source: '', streams: [
    { index: 0, text: '1: LPCM 2.0', preferred: false }, { index: 1, text: '2: DTS-HD MA 5.1', preferred: true }] }

  it('lists the streams, and extracts another in one job', async () => {
    reviewRoutes()
    const chosen: unknown[] = []
    server.use(http.get('/v1/titles/a/streams', () => json(streams)),
               http.put('/v1/titles/a/stream', async ({ request }) => {
                 chosen.push(await request.json())
                 return json(runJob(), 202)
               }),
               http.get('/v1/jobs/job-1', () => json(runJob())), http.get('/v1/jobs/job-1/log', () => json([])))
    const router = renderApp('/titles/a')
    await userEvent.click(await screen.findByRole('button', { name: 'Audio stream…' }))
    const picker = screen.getByRole('group', { name: 'Audio stream' })
    const select = await within(picker).findByRole('combobox')
    expect(select).toHaveValue('1')
    expect(within(picker).getByRole('button', { name: 'Extract this stream' })).toBeDisabled()   // already extracted
    await userEvent.selectOptions(select, '0')
    await userEvent.click(within(picker).getByRole('button', { name: 'Extract this stream' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/job-1'))
    expect(chosen).toEqual([{ audio_stream: 0 }])
  })

  it('says when the streams were read from the file, or why they could not be', async () => {
    reviewRoutes()
    server.use(http.get('/v1/titles/a/streams', () => json({ ...streams, probed: true })))
    renderApp('/titles/a')
    await userEvent.click(await screen.findByRole('button', { name: 'Audio stream…' }))
    expect(await screen.findByText(/read from the file/)).toBeInTheDocument()
  })

  it('says why a file could not be read', async () => {
    reviewRoutes()
    server.use(http.get('/v1/titles/a/streams', () => json({ title: 'Streams not read', status: 409,
                                                              detail: 'the media share is not mounted' }, 409)))
    renderApp('/titles/a')
    await userEvent.click(await screen.findByRole('button', { name: 'Audio stream…' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('the media share is not mounted')
  })
})

describe('revising titles from the list', () => {
  it('sends the titles selected back, and forgets the selection when the list changes', async () => {
    const sent: unknown[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', () => json({ total: 2, offset: 0, limit: 100, titles: [title('a'), title('b')] })),
               http.post('/v1/revise', async ({ request }) => {
                 sent.push(await request.json())
                 return json({ revised: [{ id: 'a', reverted: [], extract_invalidated: false }],
                               failed: [{ id: 'b', reason: 'not designed yet: nothing to send back' }], job: null })
               }))
    renderApp('/titles')
    await userEvent.click(await screen.findByLabelText('Select every title on this page'))
    expect(screen.getByRole('group', { name: 'Selected titles' })).toHaveTextContent('2 titles selected')
    await userEvent.click(screen.getByLabelText('Select B'))
    await userEvent.click(screen.getByRole('button', { name: 'Revise selected…' }))
    const dialog = screen.getByRole('group', { name: 'Revise' })
    await userEvent.click(within(dialog).getByLabelText(/Extract & design now/))
    await userEvent.click(within(dialog).getByRole('button', { name: 'Redesign' }))
    await waitFor(() => expect(sent).toEqual([{ ids: ['a'], to: 'design', reason: '', run: false }]))
    expect(await screen.findByRole('status')).toHaveTextContent('Sent back: 1 title. Not sent back: b (not designed yet')
    expect(screen.queryByRole('group', { name: 'Selected titles' })).not.toBeInTheDocument()
    await userEvent.click(screen.getByLabelText('Select A'))
    await userEvent.selectOptions(screen.getByLabelText('Filter kind'), 'tv')
    await waitFor(() => expect(screen.queryByRole('group', { name: 'Selected titles' })).not.toBeInTheDocument())
  })
})

describe('filtering on adjustments and design runs', () => {
  it('sends the two filters, keeps them in the address bar, and counts designs not runs', async () => {
    const asked: URLSearchParams[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', ({ request }) => {
                 asked.push(new URL(request.url).searchParams)
                 return json({ total: 1, offset: 0, limit: 100, titles: [
                   title('a', { adjusted: true, unmet_adjustments: 1, design_runs: 3, design_choices: 5 })] })
               }))
    const router = renderApp('/titles')
    expect(await screen.findByText('adjusted, 1 not honoured')).toBeInTheDocument()
    // how many runs is the review page's to say: the list counts the designs to choose from in all of them
    expect(screen.queryByText('3 runs')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Design runs')).not.toBeInTheDocument()
    expect(screen.getByRole('cell', { name: '5' })).toBeInTheDocument()
    await userEvent.selectOptions(screen.getByLabelText('Adjustments'), 'Asked for some')
    await userEvent.selectOptions(screen.getByLabelText('Answers'), 'Some not honoured')
    await waitFor(() => expect(router.state.location.search).toBe('?adjusted=true&unmet=true'))
    await waitFor(() => expect(asked.at(-1)?.get('unmet')).toBe('true'))
    expect(asked.at(-1)?.get('adjusted')).toBe('true')
    expect(screen.getByText(/asked to change something, with an adjustment not honoured/)).toBeInTheDocument()
  })
})

describe('selecting every title the filter matches', () => {
  it('collects the ids page by page, at most 500, and says when it stopped short', async () => {
    const offsets: string[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/titles', ({ request }) => {
                 const query = new URL(request.url).searchParams
                 const offset = Number(query.get('offset') ?? 0)
                 offsets.push(String(offset))
                 const titles = Array.from({ length: 100 }, (_, i) => title(`t${offset + i}`))
                 return json({ total: 620, offset, limit: 100, titles })
               }))
    renderApp('/titles?kind=movie')
    await userEvent.click(await screen.findByLabelText('Select every title on this page'))
    await userEvent.click(screen.getByRole('button', { name: 'Select all 620 matching (the first 500)' }))
    await waitFor(() => expect(screen.getByRole('group', { name: 'Selected titles' })).toHaveTextContent('500 titles selected'))
    expect(offsets.slice(1)).toEqual(['0', '100', '200', '300', '400'])
    expect(screen.getByRole('status')).toHaveTextContent('Selected the first 500 of 620')
  })
})
