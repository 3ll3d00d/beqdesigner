// The Status, Jobs and Executions screens (W4, design/jobs-and-executions.md §5) against a mocked service.
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { filterFromQuery, filterToQuery, sortFromQuery, sortToQuery } from '../api/filter'
import { saveToken } from '../api/token'
import { when } from '../format'
import { event, eventStream, index, runExecution, runJob, runResult, scanExecution, scanJob } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

beforeEach(() => saveToken(TOKEN, false))

function json(body: unknown, init?: number) {
  return HttpResponse.json(body as never, init ? { status: init } : undefined)
}

describe('the filter in the address bar', () => {
  it('goes there and back', () => {
    const filter = { needs: ['extract' as const, 'design' as const], kind: 'movie' as const, year: '>=2020', match: 'alien',
                     source: 'films', new_since_scan: true, ids: ['a'], detail: 'rolloff', flags: ['Gone' as const], declined: true }
    expect(filterFromQuery(filterToQuery(filter))).toEqual(filter)
    expect(filterFromQuery(new URLSearchParams('needs=nonsense&kind=film&year=%20&flags=Purple'))).toEqual({})
  })

  it('carries the titles table\'s sort beside the filter, ignoring a column that does not exist', () => {
    const order = { sort: 'year' as const, descending: true }
    expect(sortFromQuery(sortToQuery(order, new URLSearchParams()))).toEqual(order)
    expect(sortFromQuery(new URLSearchParams('sort=colour&descending=true'))).toEqual({ sort: null, descending: false })
    expect(filterFromQuery(new URLSearchParams('sort=year'))).toEqual({})
  })
})

describe('scan sources shortcut', () => {
  it.each([
    ['/', false], ['/', true], ['/titles?source=films', true],
  ])('queues every source from %s (scanned: %s) and opens the job', async (path, scanned) => {
    const requests: unknown[] = []
    server.use(statusRoute(status({ index: scanned ? index() : null })),
      http.get('/v1/titles', () => json({ titles: [], total: 0, offset: 0, limit: 100 })),
      http.post('/v1/jobs', async ({ request }) => {
        requests.push(await request.json())
        return json(scanJob(), 201)
      }),
      http.get('/v1/jobs/job-scan', () => json(scanJob())))
    const router = renderApp(path as string)
    await userEvent.click(await screen.findByRole('button', { name: scanned ? 'Rescan sources' : 'Scan sources' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/job-scan'))
    expect(requests).toEqual([{ kind: 'scan', request: { sources: [], allow_empty: false } }])
  })

  it('prevents duplicate submissions while pending and displays a refusal', async () => {
    let refuse!: () => void
    server.use(statusRoute(status({ index: index() })),
      http.post('/v1/jobs', async () => {
        await new Promise<void>((resolve) => { refuse = resolve })
        return json({ title: 'Queue full', status: 409, detail: 'The job queue is full.' }, 409)
      }))
    const router = renderApp('/')
    await userEvent.click(await screen.findByRole('button', { name: 'Rescan sources' }))
    expect(screen.getByRole('button', { name: 'Queuing scan…' })).toBeDisabled()
    await waitFor(() => expect(refuse).toBeDefined())
    refuse()
    expect(await screen.findByRole('alert')).toHaveTextContent('The job queue is full.')
    expect(screen.getByRole('button', { name: 'Rescan sources' })).toBeEnabled()
    expect(router.state.location.pathname).toBe('/')
  })
})

describe('status', () => {
  it('shows what titles need, each linked to the list of them', async () => {
    server.use(statusRoute(status({ index: index() })))
    const router = renderApp('/')

    const strip = await screen.findByRole('list', { name: 'What titles need' })
    expect(within(strip).getByText('To review').closest('a')).toHaveAttribute('href', '/titles?needs=review')
    expect(within(strip).getByText('30')).toBeInTheDocument()
    // each count is what is waiting for the stage, and says so: not a bare stage that reads as how many were done
    expect(within(strip).getAllByRole('listitem').map((li) => li.querySelector('.label')!.textContent))
      .toEqual(['Need attention', 'To extract', 'To design', 'To review', 'To publish', 'To commit', 'Done'])
    expect(screen.getByText('share not mounted')).toBeInTheDocument()
    await userEvent.click(within(strip).getByText('Need attention'))
    expect(router.state.location.search).toBe('?needs=attention')
  })

  it('lists every endpoint of the designer in one table: designer, url, status, protocol, release, max jobs, activity', async () => {
    const designer = {
      name: 'beqforge', reachable: true, detail: 'http://a/design (1 of 2 endpoints down)', checked_at: '2026-10-22T08:00:00Z',
      endpoints: [{ url: 'http://a/design', parallel: 1, reachable: true, detail: 'http://a/design', design_version: 5,
                    designer_build: '517cc6930fd6+src:2628', designer_release: '0.2.10', checked_at: '2026-10-22T08:00:00Z',
                    designing: [{ title_id: 't1', title: 'Alien', started_at: '2026-10-22T07:58:00Z' }] },
                  { url: 'http://b/design', parallel: 3, reachable: false, detail: 'GET http://b/health failed: refused',
                    checked_at: '2026-10-22T08:00:00Z', designing: [] }],
    }
    server.use(statusRoute(status({ index: index(), designer })))
    renderApp('/')

    const table = await screen.findByRole('table', { name: 'Designers' })
    expect(within(table).getAllByRole('columnheader').map((th) => th.textContent))
      .toEqual(['Designer', 'URL', 'Status', 'Protocol', 'Version', 'Max jobs', 'Activity'])
    const [, a, b] = within(table).getAllByRole('row')
    const cells = (row: HTMLElement) => within(row).getAllByRole('cell').map((td) => td.textContent)
    const at = when('2026-10-22T08:00:00Z')
    expect(cells(a!)).toEqual(['beqforge', 'http://a/design', `up · ${at}`, '5', '0.2.10', '1',
                               `Alien since ${when('2026-10-22T07:58:00Z')}`])
    expect(cells(b!)).toEqual(['beqforge', 'http://b/design', `down · ${at}GET http://b/health failed: refused`, '—', '—', '3', '—'])
    // the build is a revision, not a release: it is there on hover, not in the cell
    expect(within(a!).getByText('0.2.10')).toHaveAttribute('title', '517cc6930fd6+src:2628')
    expect(within(a!).getByRole('link', { name: 'Alien' })).toHaveAttribute('href', '/titles/t1')
    expect(screen.getByRole('heading', { name: 'Designers' })).toBeInTheDocument()
    expect(screen.getAllByRole('table', { name: 'Designers' })).toHaveLength(1)
  })

  it('says an endpoint with nothing out is idle, and lists a designer with one endpoint too', async () => {
    const designer = { name: 'beqforge', reachable: true, detail: 'http://a/design', checked_at: null,
                       endpoints: [{ url: 'http://a/design', parallel: 2, reachable: true, detail: 'http://a/design' }],
                       design_version: null, design_disagreement: [] }
    server.use(statusRoute(status({ index: index(), designer })))
    renderApp('/')

    const table = await screen.findByRole('table', { name: 'Designers' })
    const [, a] = within(table).getAllByRole('row')
    expect(within(a!).getAllByRole('cell').map((td) => td.textContent))
      .toEqual(['beqforge', 'http://a/design', 'up · —', '—', '—', '2', 'idle'])
    expect(screen.queryByRole('alert', { name: 'Designers disagree' })).not.toBeInTheDocument()
  })

  it('warns when the endpoints up would design differently', async () => {
    const designer = {
      name: 'beqforge', reachable: true, detail: 'http://a/design, http://b/design', checked_at: null,
      endpoints: [{ url: 'http://a/design', parallel: 1, reachable: true, detail: 'http://a/design', design_version: 4 },
                  { url: 'http://b/design', parallel: 2, reachable: true, detail: 'http://b/design', design_version: 5 }],
      design_version: 5,
      design_disagreement: ['http://a/design designs with version 4, default settings',
                            'http://b/design designs with version 5, default settings'],
    }
    server.use(statusRoute(status({ index: index(), designer })))
    renderApp('/')

    const warning = await screen.findByRole('alert', { name: 'Designers disagree' })
    expect(within(warning).getAllByRole('listitem').map((li) => li.textContent)).toEqual(designer.design_disagreement)
    const rows = within(screen.getByRole('table', { name: 'Designers' })).getAllByRole('row')
    expect(rows.slice(1).map((r) => within(r).getAllByRole('cell')[3]!.textContent)).toEqual(['4', '5'])
  })

  it('lists a designer with no endpoints as one row saying why', async () => {
    const designer = { name: 'manual', reachable: null, detail: 'not an HTTP designer: nothing to ask', checked_at: null,
                       endpoints: [] }
    server.use(statusRoute(status({ index: index(), designer })))
    renderApp('/')

    const table = await screen.findByRole('table', { name: 'Designers' })
    const [, row] = within(table).getAllByRole('row')
    expect(within(row!).getAllByRole('cell').map((td) => td.textContent)).toEqual(['manual', 'not an HTTP designer: nothing to ask'])
  })

  it('follows the execution running now', async () => {
    const running = runExecution({ state: 'running', finished_at: null, result: null,
                                   progress: { done: 1, total: 4, title: 'Alien', stage: 'extract', id: 'a', remaining_seconds: 600 } })
    server.use(statusRoute(status({ index: index(), running_execution: running, waiting: 2 })),
               http.get('/v1/executions/ex-1/events', () => new HttpResponse(eventStream([
                 event(1, { type: 'state', state: 'running' }),
                 event(2, { type: 'extract_progress', title: 'Alien', title_id: 'a', done_ms: 500, percent: 42 }),
               ]), { headers: { 'Content-Type': 'text/event-stream' } })))
    renderApp('/')

    expect((await screen.findByText('Movies through design')).closest('a')).toHaveAttribute('href', '/executions/ex-1')
    expect(screen.getByRole('progressbar', { name: 'Progress' })).toHaveAttribute('value', '1')
    const facts = screen.getByText('Estimated end').closest('dl')!
    await within(facts).findByText('Any time')
    const rows = Object.fromEntries(within(facts).getAllByRole('term').map((dt) => [dt.textContent, dt.nextElementSibling!.textContent]))
    expect(Object.keys(rows)).toEqual(['Running', 'Started', 'Estimated end', 'Progress', 'Activity', 'Waiting', 'Jobs run'])
    expect(rows['Started']).toBe(when(running.started_at))
    expect(rows['Estimated end']).toBe('— (about 10 min 0 s to go)')
    expect(rows['Progress']).toBe('1 of 4')
    expect(await screen.findByText(/42% extracted/)).toBeInTheDocument()
    expect(within(facts).getByText('Alien')).toBeInTheDocument()
    expect(rows['Waiting']).toBe('2 executions')
  })

  it('says when the execution should end and lists the designs out on the designer as its activity', async () => {
    const running = runExecution({ state: 'running', finished_at: null, result: null,
                                   progress: { done: 10, total: 100, title: 'Brazil', stage: 'design', id: 'b', remaining_seconds: 3600,
                                               per_hour: 40, estimated_finish: '2026-10-23T07:00:00Z' } })
    const designer = { name: 'beqforge', reachable: true, detail: '', checked_at: null,
                       endpoints: [{ url: 'http://a/design', parallel: 2, reachable: true, detail: 'http://a/design',
                                     designing: [{ title_id: 'a', title: 'Alien', started_at: '2026-10-22T08:00:00Z' },
                                                 { title_id: 'b', title: 'Brazil', started_at: '2026-10-22T08:01:00Z' }] }] }
    server.use(statusRoute(status({ index: index(), running_execution: running, designer })),
               http.get('/v1/executions/ex-1/events', () => new HttpResponse(eventStream([]),
                                                                              { headers: { 'Content-Type': 'text/event-stream' } })))
    renderApp('/')

    const facts = (await screen.findByText('Estimated end')).closest('dl')!
    const rows = Object.fromEntries(within(facts).getAllByRole('term').map((dt) => [dt.textContent, dt.nextElementSibling!.textContent]))
    expect(rows['Estimated end']).toBe(`${when('2026-10-23T07:00:00Z')} (about 1 h 0 min to go)`)
    expect(rows['Progress']).toBe('10 of 100, 40 an hour')
    expect(rows['Activity']).toBe('designing Alien, Brazil')
  })

  it('lists each job that is active or paused, with pause, resume and run now, and says why one was refused', async () => {
    const asked: string[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/jobs', () => json({ jobs: [
                 runJob({ name: 'Nightly', every: 90, window: ['01:00-05:00'], next_at: '2026-10-23T01:00:00Z',
                          last_skip: 'busy', progress: null }),
                 scanJob({ state: 'paused' }),
                 runJob({ id: 'gone', name: 'Old', state: 'done' }),
               ] })),
               http.post('/v1/jobs/:id/:action', ({ params }) => {
                 asked.push(`${params.id} ${params.action}`)
                 return params.action === 'run'
                   ? json({ title: 'Service busy', status: 409, detail: 'another execution is running' }, 409)
                   : json(runJob())
               }))
    renderApp('/')

    const table = await screen.findByRole('table', { name: 'Jobs' })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map((r) => within(r).getAllByRole('cell').slice(0, 4).map((td) => td.textContent))).toEqual([
      ['NightlyRun through design: needing extract or design, movies', 'Every 1 h 30 min01:00-05:00', 'Activeskipped: busy',
       when('2026-10-23T01:00:00Z')],
      ['List the sources againScan every source', 'Once', 'Paused', '—'],
    ])
    expect(within(rows[0]!).getByRole('link', { name: 'Nightly' })).toHaveAttribute('href', '/jobs/job-1')
    expect(within(rows[0]!).getByRole('link', { name: 'Edit…' })).toHaveAttribute('href', '/jobs/job-1')
    await userEvent.click(within(rows[0]!).getByRole('button', { name: 'Pause' }))
    await userEvent.click(within(rows[1]!).getByRole('button', { name: 'Resume' }))
    await userEvent.click(within(rows[0]!).getByRole('button', { name: 'Run now' }))
    expect(await within(rows[0]!).findByRole('alert')).toHaveTextContent('Service busy: another execution is running')
    expect(asked).toEqual(['job-1 pause', 'job-scan resume', 'job-1 run'])
    expect(within(rows[1]!).queryByRole('button', { name: 'Run now' })).not.toBeInTheDocument()   // a one-off runs by itself
  })

  it('shows what runs, what waits and the jobs as one section, with every way to start one', async () => {
    server.use(statusRoute(status({ index: index() })))
    renderApp('/')

    const section = (await screen.findByRole('heading', { name: 'Jobs' })).closest('section')!
    const facts = section.querySelector('dl')!
    await within(facts).findByText('Any time')   // the default window, asked for apart from the status
    const rows = Object.fromEntries(within(facts).getAllByRole('term').map((dt) => [dt.textContent, dt.nextElementSibling!.textContent]))
    expect(rows).toEqual({ Running: 'Nothing', Waiting: 'Nothing', 'Jobs run': 'Any time — open now Change…' })
    expect(await within(section).findByText('No jobs are active or paused.')).toBeInTheDocument()
    expect(within(section).getAllByRole('button').map((b) => b.textContent)).toEqual(['Change…', 'Rescan sources'])
    expect(within(section).getByRole('link', { name: 'New job…' })).toHaveAttribute('href', '/jobs/new')
    expect(within(section).getByRole('link', { name: 'Every execution' })).toHaveAttribute('href', '/executions')
    const sources = within(section).getByRole('table', { name: 'Sources' })
    expect(within(sources).getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('cell')[0]!.textContent))
      .toEqual(['films', 'tv'])
    expect(screen.queryByRole('heading', { name: 'Schedule' })).not.toBeInTheDocument()
  })

  it('says when the library has not been scanned', async () => {
    server.use(statusRoute(status()))
    renderApp('/')
    expect(await screen.findByText('Scan the sources…')).toHaveAttribute('href', '/jobs/new?kind=scan')
  })
})

describe('a job', () => {
  it('says what it does and when, follows its execution running now, and lists the ones before', async () => {
    const running = runExecution({ id: 'ex-2', number: 2, state: 'running', finished_at: null, result: null,
                                   progress: { done: 3, total: 5, title: 'Heat', stage: 'design', id: 'h' } })
    server.use(http.get('/v1/jobs/job-1', () => json(runJob({ window: ['01:00-05:00'], next_at: null, remaining: 2,
                                                                progress: { done: 3, total: 5 }, running_execution: running,
                                                                executions: 2 }))),
               http.get('/v1/executions', ({ request }) => {
                 expect(new URL(request.url).searchParams.get('job')).toBe('job-1')
                 return json({ running, next: [], executions: [running, runExecution({ state: 'stopped', stopped_because: 'window' })] })
               }),
               http.get('/v1/executions/ex-2/events', () => new HttpResponse(new ReadableStream(), {
                 headers: { 'Content-Type': 'text/event-stream' } })))
    renderApp('/jobs/job-1')

    expect(await screen.findByRole('heading', { name: 'Movies through design' })).toBeInTheDocument()
    const facts = document.querySelector('dl')!
    await within(facts).findByText(/Europe\/London/)
    const rows = Object.fromEntries(within(facts).getAllByRole('term').map((dt) => [dt.textContent, dt.nextElementSibling!.textContent]))
    expect(rows).toMatchObject({ State: 'Active', Does: 'Run through design: needing extract or design, movies', When: 'Once',
                                 Runs: '01:00-05:00 (Europe/London)', Next: '—', Progress: '3 of 5 done (2 titles left)' })
    const now = screen.getByRole('region', { name: 'Running' })
    expect(within(now).getByRole('link', { name: 'Movies through design (run 2)' })).toHaveAttribute('href', '/executions/ex-2')
    expect(within(now).getByText('Cancelling it removes this job.')).toBeInTheDocument()
    const before = await screen.findByRole('table', { name: 'Its executions' })
    expect(within(before).getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('cell')[0]!.textContent))
      .toEqual(['Stopped its window closed'])
    expect(screen.queryByRole('button', { name: 'Run now' })).not.toBeInTheDocument()
  })

  it("changes a recurring job's name, interval and window", async () => {
    let body: unknown
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/jobs/job-1', () => json(runJob({ name: 'Nightly', every: 60, progress: null }))),
               http.patch('/v1/jobs/job-1', async ({ request }) => {
                 body = await request.json()
                 return json(runJob({ name: 'Hourly', every: 30, progress: null }))
               }))
    renderApp('/jobs/job-1')

    await userEvent.click(await screen.findByRole('button', { name: 'Edit…' }))
    const form = screen.getByRole('form', { name: 'Edit job' })
    await userEvent.clear(within(form).getByLabelText('Name'))
    await userEvent.type(within(form).getByLabelText('Name'), 'Hourly')
    await userEvent.clear(within(form).getByLabelText('Minutes'))
    await userEvent.type(within(form).getByLabelText('Minutes'), '30')
    await userEvent.click(within(form).getByRole('radio', { name: 'Any time' }))
    await userEvent.click(within(form).getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(body).toMatchObject({ name: 'Hourly', every: 30, window: [],
                                                     request: { through: 'design', filter: { needs: ['extract', 'design'] } } }))
  })

  it('removes a job once asked, and says why one ended by itself', async () => {
    let removed = false
    server.use(http.get('/v1/jobs/job-1', () => json(runJob({ every: 60 }))),
               http.delete('/v1/jobs/job-1', () => {
                 removed = true
                 return json(runJob({ state: 'removed' }))
               }),
               http.get('/v1/jobs/job-2', () => json(runJob({ id: 'job-2', state: 'done', ended: 'all 2 listed titles are done' }))))
    const router = renderApp('/jobs/job-1')
    await userEvent.click(await screen.findByRole('button', { name: 'Remove…' }))
    await userEvent.click(within(screen.getByRole('group', { name: 'Confirm remove' })).getByRole('button', { name: 'Remove' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/'))
    expect(removed).toBe(true)

    renderApp('/jobs/job-2')
    expect((await screen.findByText('Done')).closest('dd')).toHaveTextContent('all 2 listed titles are done')
  })
})

describe('executions', () => {
  it('lists them, with each job\'s next, and filters by job and state', async () => {
    const asked: string[] = []
    server.use(http.get('/v1/jobs', () => json({ jobs: [runJob()] })),
               http.get('/v1/executions', ({ request }) => {
                 asked.push(new URL(request.url).search)
                 return json({ running: null, next: [{ job_id: 'job-1', job_name: 'Movies through design', at: '2026-10-23T01:00:00Z' }],
                               executions: [runExecution(), scanExecution(), runExecution({ id: 'ex-0', job_name: null,
                                                                                            job_id: null, number: 0 })] })
               }))
    const router = renderApp('/executions')

    const next = await screen.findByRole('table', { name: 'Next' })
    expect(within(next).getByRole('link', { name: 'Movies through design' })).toHaveAttribute('href', '/jobs/job-1')
    const log = screen.getByRole('table', { name: 'Executions' })
    expect(within(log).getByRole('link', { name: 'Movies through design' })).toHaveAttribute('href', '/executions/ex-1')
    expect(within(log).getByText('List the sources again')).toBeInTheDocument()
    expect(within(log).getByText('Run through design: needing extract or design, movies')).toBeInTheDocument()   // no job kept
    expect(within(log).getAllByRole('row')[1]!.lastElementChild!.textContent).toBe('1 designed, 1 failed')
    await userEvent.selectOptions(screen.getByLabelText('State'), 'stopped')
    await userEvent.selectOptions(screen.getByLabelText('Job'), 'job-1')
    await waitFor(() => expect(asked.at(-1)).toContain('job=job-1'))
    expect(asked.at(-1)).toContain('state=stopped')
    expect(router.state.location.search).toBe('?state=stopped&job=job-1')
  })

  it("shows a finished execution's result, its failures linked to the titles, its job, and its log", async () => {
    server.use(http.get('/v1/executions/ex-1', () => json(runExecution())),
               http.get('/v1/executions/ex-1/log', () => json([event(1, { type: 'state', state: 'running', text: 'Started' }),
                                                                event(2, { type: 'state', state: 'succeeded', text: 'Finished' })])))
    renderApp('/executions/ex-1')

    expect(await screen.findByText(/3 titles selected: 2 extracted/)).toBeInTheDocument()
    expect(screen.getByText('b').closest('a')).toHaveAttribute('href', '/titles/b')
    expect(screen.getByText(/designer said no/)).toBeInTheDocument()
    expect(screen.getByText('Cube')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Movies through design' })).toHaveAttribute('href', '/jobs/job-1')
    const log = await screen.findByRole('list', { name: 'Log' })
    expect(within(log).getByText(/Finished/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument()
  })

  it('says a stopped run left its titles for its job, not that it was cancelled', async () => {
    server.use(http.get('/v1/executions/ex-1', () => json(runExecution({
                 state: 'stopped', stopped_because: 'window', result: runResult({ cancelled: true, not_run: ['c', 'd'] }) }))),
               http.get('/v1/executions/ex-1/log', () => json([])))
    renderApp('/executions/ex-1')
    expect(await screen.findByText("Stopped: 2 titles left for its job's next run.")).toBeInTheDocument()
    expect(screen.getByText(/its window closed/)).toBeInTheDocument()
  })

  it('shows why an execution itself failed', async () => {
    server.use(http.get('/v1/executions/scan-1', () => json(scanExecution())), http.get('/v1/executions/scan-1/log', () => json([])))
    renderApp('/executions/scan-1')
    expect(await screen.findByRole('alert')).toHaveTextContent('profile unreadable')
  })

  it('cancels a running execution, which stops after the title in hand', async () => {
    let cancelled = false
    const running = runExecution({ state: 'running', finished_at: null, result: null })
    server.use(http.get('/v1/executions/ex-1', () => json(running)),
               http.get('/v1/executions/ex-1/events', () => new HttpResponse(new ReadableStream(), {
                 headers: { 'Content-Type': 'text/event-stream' } })),
               http.post('/v1/executions/ex-1/cancel', () => {
                 cancelled = true
                 return json(running)
               }))
    renderApp('/executions/ex-1')

    expect(await screen.findByText('Cancelling it removes its job: nothing more of it runs.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(await screen.findByText('Stopping after the title in hand…')).toBeInTheDocument()
    expect(cancelled).toBe(true)
  })

  it('sends the old Jobs address to the status page', async () => {
    server.use(statusRoute(status()))
    const router = renderApp('/jobs')
    await waitFor(() => expect(router.state.location.pathname).toBe('/'))
  })
})

describe('a new job', () => {
  it('previews a run, then runs it and goes to the job', async () => {
    const planned: unknown[] = []
    server.use(statusRoute(status({ index: index() })),
               http.post('/v1/plan', async ({ request }) => {
                 planned.push(await request.json())
                 return json({ through: 'design', label: 'Extract & design 2 titles',
                               planned: [{ id: 'a', title: 'Alien', stages: ['extract', 'design'] },
                                         { id: 'h', title: 'Heat', stages: ['design'] }],
                               skipped: [{ id: 'c', title: 'Cube', reason: 'waiting for review' }] })
               }),
               http.post('/v1/jobs', async ({ request }) => {
                 planned.push(await request.json())
                 return json(runJob({ id: 'new-job' }), 201)
               }),
               http.get('/v1/jobs/new-job', () => json(runJob({ id: 'new-job' }))))
    const router = renderApp('/jobs/new?kind=movie')

    await userEvent.click(await screen.findByRole('checkbox', { name: 'Extract' }))
    await userEvent.type(screen.getByLabelText('Year'), '>=2020')
    await userEvent.click(screen.getByRole('button', { name: 'Preview' }))

    const preview = await screen.findByRole('region', { name: 'Preview' })
    expect(within(preview).getByText('Extract & design 2 titles')).toBeInTheDocument()
    expect(within(preview).getByText(/2 titles would run, 1 title skipped \(needing extract, movies, year >=2020\)/))
      .toBeInTheDocument()
    expect(within(preview).getByText(/waiting for review/)).toBeInTheDocument()
    expect(planned[0]).toEqual({ filter: { needs: ['extract'], kind: 'movie', year: '>=2020', new_since_scan: false },
                                 through: 'design', scan_first: true, retry_failed: false, redesign: false,
                                 revise_to: 'design' })

    await userEvent.click(screen.getByRole('checkbox', { name: 'Retry titles that failed before' }))
    expect(screen.getByText('The choices changed since this preview: preview again.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/new-job'))
    expect(planned[1]).toEqual({ kind: 'run', request: { ...planned[0] as object, retry_failed: true } })
  })

  it('makes a recurring run, named, every so many minutes, and offers it only through extract or design', async () => {
    let body: unknown
    server.use(statusRoute(status({ index: index(), repository_writes: true, repositories_configured: true })),
               http.post('/v1/jobs', async ({ request }) => {
                 body = await request.json()
                 return json(runJob({ id: 'nightly', every: 45 }), 201)
               }),
               http.get('/v1/jobs/nightly', () => json(runJob({ id: 'nightly', every: 45 }))))
    const router = renderApp('/jobs/new')

    await userEvent.type(await screen.findByLabelText('Name'), 'Nightly')
    await userEvent.click(screen.getByRole('radio', { name: 'Every' }))
    await userEvent.clear(screen.getByLabelText('Minutes'))
    await userEvent.type(screen.getByLabelText('Minutes'), '45')
    await userEvent.click(screen.getByRole('radio', { name: 'publish' }))
    expect(screen.getByRole('radio', { name: 'Every' })).toBeDisabled()
    expect(screen.getByRole('radio', { name: 'Once' })).toBeChecked()
    await userEvent.click(screen.getByRole('radio', { name: 'design' }))
    await userEvent.click(screen.getByRole('radio', { name: 'Every' }))
    await userEvent.click(screen.getByRole('button', { name: 'Make job' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/nightly'))
    expect(body).toMatchObject({ kind: 'run', name: 'Nightly', every: 45, request: { through: 'design' } })
  })

  it('offers publish and commit only when the service allows them, and says why not', async () => {
    let answer: () => void = () => {}
    const answered = new Promise<void>((resolve) => { answer = resolve })
    server.use(http.get('/v1/status', async () => {
      await answered
      return HttpResponse.json(status({ index: index(), repository_writes: false }))
    }))
    renderApp('/jobs/new')
    expect(await screen.findByRole('radio', { name: 'publish' })).toBeDisabled()   // not offered before the service says
    answer()
    expect(await screen.findByText(/allow_repository_writes/)).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'publish' })).toBeDisabled()
    expect(screen.getByRole('radio', { name: 'commit' })).toBeDisabled()
    expect(screen.getByText(/allow_repository_writes/)).toBeInTheDocument()
  })

  it('offers them when it does', async () => {
    server.use(statusRoute(status({ index: index(), repository_writes: true, repositories_configured: true })))
    renderApp('/jobs/new')
    await waitFor(() => expect(screen.getByRole('radio', { name: 'publish' })).toBeEnabled())
  })

  it('scans the sources chosen', async () => {
    let body: unknown
    server.use(statusRoute(status({ index: index() })),
               http.post('/v1/jobs', async ({ request }) => {
                 body = await request.json()
                 return json(scanJob({ id: 'scan-2' }), 201)
               }),
               http.get('/v1/jobs/scan-2', () => json(scanJob({ id: 'scan-2' }))))
    const router = renderApp('/jobs/new?kind=scan')

    await userEvent.click(await screen.findByRole('checkbox', { name: 'tv' }))
    await userEvent.click(screen.getByRole('button', { name: 'Scan' }))

    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/scan-2'))
    expect(body).toEqual({ kind: 'scan', request: { sources: ['tv'], allow_empty: false } })
  })

  it('says why a run was refused', async () => {
    server.use(statusRoute(status({ index: index() })),
               http.post('/v1/jobs', () => json({ title: 'Unknown source', status: 422, detail: 'no such source: x' }, 422)))
    renderApp('/jobs/new')
    await userEvent.click(await screen.findByRole('button', { name: 'Run' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Unknown source: no such source: x')
  })
})

it('keeps a run result that has every list empty readable', () => {
  expect(runResult({ failed: [], skipped: [] }).failed).toEqual([])
})
