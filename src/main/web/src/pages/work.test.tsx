// The Status and Jobs screens (W4) against a mocked service.
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { filterFromQuery, filterToQuery, sortFromQuery, sortToQuery } from '../api/filter'
import { saveToken } from '../api/token'
import { when } from '../format'
import { event, eventStream, index, runJob, runResult, scanJob, schedule } from '../test/fixtures'
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
      http.get('/v1/schedule', () => json(schedule())),
      http.get('/v1/titles', () => json({ titles: [], total: 0, offset: 0, limit: 100 })),
      http.post('/v1/jobs/scan', async ({ request }) => {
        requests.push(await request.json())
        return json(scanJob(), 202)
      }),
      http.get('/v1/jobs/scan-1', () => json(scanJob())),
      http.get('/v1/jobs/scan-1/log', () => json([])))
    const router = renderApp(path as string)
    await userEvent.click(await screen.findByRole('button', { name: scanned ? 'Rescan sources' : 'Scan sources' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/scan-1'))
    expect(requests).toEqual([{ sources: [], allow_empty: false }])
  })

  it('prevents duplicate submissions while pending and displays a refusal', async () => {
    let refuse!: () => void
    server.use(statusRoute(status({ index: index() })),
      http.get('/v1/schedule', () => json(schedule())),
      http.post('/v1/jobs/scan', async () => {
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
    server.use(statusRoute(status({ index: index() })), http.get('/v1/schedule', () => json(schedule())))
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
    server.use(statusRoute(status({ index: index(), designer })), http.get('/v1/schedule', () => json(schedule())))
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
    server.use(statusRoute(status({ index: index(), designer })), http.get('/v1/schedule', () => json(schedule())))
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
    server.use(statusRoute(status({ index: index(), designer })), http.get('/v1/schedule', () => json(schedule())))
    renderApp('/')

    const warning = await screen.findByRole('alert', { name: 'Designers disagree' })
    expect(within(warning).getAllByRole('listitem').map((li) => li.textContent)).toEqual(designer.design_disagreement)
    const rows = within(screen.getByRole('table', { name: 'Designers' })).getAllByRole('row')
    expect(rows.slice(1).map((r) => within(r).getAllByRole('cell')[3]!.textContent)).toEqual(['4', '5'])
  })

  it('lists a designer with no endpoints as one row saying why', async () => {
    const designer = { name: 'manual', reachable: null, detail: 'not an HTTP designer: nothing to ask', checked_at: null,
                       endpoints: [] }
    server.use(statusRoute(status({ index: index(), designer })), http.get('/v1/schedule', () => json(schedule())))
    renderApp('/')

    const table = await screen.findByRole('table', { name: 'Designers' })
    const [, row] = within(table).getAllByRole('row')
    expect(within(row!).getAllByRole('cell').map((td) => td.textContent)).toEqual(['manual', 'not an HTTP designer: nothing to ask'])
  })

  it('follows the job running now', async () => {
    const running = runJob({ state: 'running', finished_at: null, result: null,
                             progress: { done: 1, total: 4, title: 'Alien', stage: 'extract', id: 'a', remaining_seconds: 600 } })
    server.use(statusRoute(status({ index: index(), current_job: running, queued: 2 })),
               http.get('/v1/schedule', () => json(schedule())),
               http.get('/v1/jobs/job-1/events', () => new HttpResponse(eventStream([
                 event(1, { type: 'state', state: 'running' }),
                 event(2, { type: 'extract_progress', title: 'Alien', title_id: 'a', done_ms: 500, percent: 42 }),
               ]), { headers: { 'Content-Type': 'text/event-stream' } })))
    renderApp('/')

    expect(await screen.findByText(/Run through design: needing extract or design, movies/)).toBeInTheDocument()
    expect(screen.getByRole('progressbar', { name: 'Progress' })).toHaveAttribute('value', '1')
    await screen.findByText('Schedule')
    const facts = screen.getByText('Estimated end').closest('dl')!
    const rows = Object.fromEntries(within(facts).getAllByRole('term').map((dt) => [dt.textContent, dt.nextElementSibling!.textContent]))
    expect(Object.keys(rows)).toEqual(['Job', 'Started', 'Estimated end', 'Progress', 'Activity', 'Waiting', 'Jobs run',
                                       'Schedule', 'Next run'])
    expect(rows['Started']).toBe(when(running.started_at))
    expect(rows['Estimated end']).toBe('— (about 10 min 0 s to go)')
    expect(rows['Progress']).toBe('1 of 4')
    expect(await screen.findByText(/42% extracted/)).toBeInTheDocument()
    expect(within(facts).getByText('Alien')).toBeInTheDocument()
    expect(rows['Waiting']).toBe('2 jobs')
  })

  it('says when the job should end and lists the designs out on the designer as its activity', async () => {
    const running = runJob({ state: 'running', finished_at: null, result: null,
                             progress: { done: 10, total: 100, title: 'Brazil', stage: 'design', id: 'b', remaining_seconds: 3600,
                                         per_hour: 40, estimated_finish: '2026-10-23T07:00:00Z' } })
    const designer = { name: 'beqforge', reachable: true, detail: '', checked_at: null,
                       endpoints: [{ url: 'http://a/design', parallel: 2, reachable: true, detail: 'http://a/design',
                                     designing: [{ title_id: 'a', title: 'Alien', started_at: '2026-10-22T08:00:00Z' },
                                                 { title_id: 'b', title: 'Brazil', started_at: '2026-10-22T08:01:00Z' }] }] }
    server.use(statusRoute(status({ index: index(), current_job: running, designer })),
               http.get('/v1/schedule', () => json(schedule())),
               http.get('/v1/jobs/job-1/events', () => new HttpResponse(eventStream([]),
                                                                        { headers: { 'Content-Type': 'text/event-stream' } })))
    renderApp('/')

    const facts = (await screen.findByText('Estimated end')).closest('dl')!
    const rows = Object.fromEntries(within(facts).getAllByRole('term').map((dt) => [dt.textContent, dt.nextElementSibling!.textContent]))
    expect(rows['Estimated end']).toBe(`${when('2026-10-23T07:00:00Z')} (about 1 h 0 min to go)`)
    expect(rows['Progress']).toBe('10 of 100, 40 an hour')
    expect(rows['Activity']).toBe('designing Alien, Brazil')
  })

  it('pauses and runs the schedule, and says why a tick was refused', async () => {
    const saved: unknown[] = []
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/schedule', () => json(schedule())),
               http.put('/v1/schedule', async ({ request }) => {
                 const body = await request.json()
                 saved.push(body)
                 return json(schedule({ ...(body as object), next_run_at: null }))
               }),
               http.post('/v1/schedule/trigger', () =>
                 json({ title: 'Service busy', status: 409, detail: 'a job is queued or running' }, 409)))
    renderApp('/')

    await userEvent.click(await screen.findByRole('button', { name: 'Pause' }))
    expect(await screen.findByRole('button', { name: 'Resume' })).toBeInTheDocument()
    expect(saved).toEqual([{ enabled: false, interval_minutes: 60, filter: { new_since_scan: false, kind: 'movie' },
                             through: 'design', retry_failed: false }])
    await userEvent.click(screen.getByRole('button', { name: 'Run now' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Service busy: a job is queued or running')
  })

  it('shows the job, the schedule and the sources it scans as one section, with every way to start one', async () => {
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/schedule', () => json(schedule({
                 last_run: { job_id: 'job-0', state: 'failed', finished_at: '2026-10-22T07:00:00Z' },
                 last_skip: 'the designer is down' }))))
    renderApp('/')

    const section = (await screen.findByRole('heading', { name: 'Jobs' })).closest('section')!
    await within(section).findByText('Schedule')
    const facts = section.querySelector('dl')!
    await within(facts).findByText('Any time')   // the default window, asked for apart from the status
    const rows = Object.fromEntries(within(facts).getAllByRole('term').map((dt) => [dt.textContent, dt.nextElementSibling!.textContent]))
    expect(rows).toEqual({
      Job: 'None running', Waiting: 'Nothing', 'Jobs run': 'Any time — open now Change…',
      Schedule: 'Every 60 min: scan the sources, then run movies through design.',
      'Next run': when(schedule().next_run_at), 'Last run': `Failed ${when('2026-10-22T07:00:00Z')}`,
      'Last skipped': 'the designer is down',
    })
    expect(within(facts).getByRole('link', { name: when('2026-10-22T07:00:00Z') })).toHaveAttribute('href', '/jobs/job-0')
    expect(within(section).getAllByRole('button').map((b) => b.textContent)).toEqual(['Change…', 'Pause', 'Run now',
                                                                                         'Rescan sources'])
    expect(within(section).getByRole('link', { name: 'New job…' })).toHaveAttribute('href', '/jobs/new')
    const sources = within(section).getByRole('table', { name: 'Sources' })
    expect(within(sources).getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('cell')[0]!.textContent))
      .toEqual(['films', 'tv'])
    expect(screen.queryByRole('heading', { name: 'Schedule' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Sources' })).not.toBeInTheDocument()
  })

  it('says when the library has not been scanned', async () => {
    server.use(statusRoute(status()), http.get('/v1/schedule', () => json(schedule({ enabled: false }))))
    renderApp('/')
    expect(await screen.findByText('Scan the sources…')).toHaveAttribute('href', '/jobs/new?kind=scan')
    expect(await screen.findByText('Paused.')).toBeInTheDocument()
  })

  it('says why the schedule turned itself off', async () => {
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/schedule', () => json(schedule({ enabled: false, ended: 'all 2 listed titles are done' }))))
    renderApp('/')
    expect((await screen.findByText('Ended')).nextElementSibling).toHaveTextContent('all 2 listed titles are done')
  })
})

describe('jobs', () => {
  it('lists them and filters by state', async () => {
    const asked: (string | null)[] = []
    server.use(http.get('/v1/jobs', ({ request }) => {
      asked.push(new URL(request.url).searchParams.get('state'))
      return json({ jobs: [runJob(), scanJob()] })
    }))
    const router = renderApp('/jobs')

    expect(await screen.findByText('Scan every source')).toBeInTheDocument()
    expect(screen.getByText('Run through design: needing extract or design, movies').closest('a'))
      .toHaveAttribute('href', '/jobs/job-1')
    await userEvent.selectOptions(screen.getByLabelText('State'), 'failed')
    await waitFor(() => expect(asked).toContain('failed'))
    expect(router.state.location.search).toBe('?state=failed')
  })

  it("shows a finished job's result, its failures linked to the titles, and its log", async () => {
    server.use(http.get('/v1/jobs/job-1', () => json(runJob())),
               http.get('/v1/jobs/job-1/log', () => json([event(1, { type: 'state', state: 'running', text: 'Started' }),
                                                          event(2, { type: 'state', state: 'succeeded', text: 'Finished' })])))
    renderApp('/jobs/job-1')

    expect(await screen.findByText(/3 titles selected: 2 extracted/)).toBeInTheDocument()
    expect(screen.getByText('b').closest('a')).toHaveAttribute('href', '/titles/b')
    expect(screen.getByText(/designer said no/)).toBeInTheDocument()
    expect(screen.getByText('Cube')).toBeInTheDocument()
    const log = await screen.findByRole('list', { name: 'Log' })
    expect(within(log).getByText(/Finished/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument()
  })

  it('shows why a job itself failed', async () => {
    server.use(http.get('/v1/jobs/scan-1', () => json(scanJob())), http.get('/v1/jobs/scan-1/log', () => json([])))
    renderApp('/jobs/scan-1')
    expect(await screen.findByRole('alert')).toHaveTextContent('profile unreadable')
  })

  it('cancels a running job, which stops after the title in hand', async () => {
    let cancelled = false
    const running = runJob({ state: 'running', finished_at: null, result: null })
    server.use(http.get('/v1/jobs/job-1', () => json(running)),
               http.get('/v1/jobs/job-1/events', () => new HttpResponse(new ReadableStream(), {
                 headers: { 'Content-Type': 'text/event-stream' } })),
               http.post('/v1/jobs/job-1/cancel', () => {
                 cancelled = true
                 return json(running)
               }))
    renderApp('/jobs/job-1')

    await userEvent.click(await screen.findByRole('button', { name: 'Cancel' }))
    expect(await screen.findByText('Stopping after the title in hand…')).toBeInTheDocument()
    expect(cancelled).toBe(true)
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
               http.post('/v1/jobs/run', async ({ request }) => {
                 planned.push(await request.json())
                 return json(runJob({ id: 'new-job', state: 'queued' }), 202)
               }),
               http.get('/v1/jobs/new-job', () => json(runJob({ id: 'new-job', state: 'queued', result: null }))),
               http.get('/v1/jobs/new-job/events', () => new HttpResponse(new ReadableStream(), {
                 headers: { 'Content-Type': 'text/event-stream' } })))
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
    expect(planned[1]).toMatchObject({ retry_failed: true })
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
               http.post('/v1/jobs/scan', async ({ request }) => {
                 body = await request.json()
                 return json(scanJob({ id: 'scan-2', state: 'queued', error: null }), 202)
               }),
               http.get('/v1/jobs/scan-2', () => json(scanJob({ id: 'scan-2', state: 'queued', error: null }))),
               http.get('/v1/jobs/scan-2/events', () => new HttpResponse(new ReadableStream(), {
                 headers: { 'Content-Type': 'text/event-stream' } })))
    const router = renderApp('/jobs/new?kind=scan')

    await userEvent.click(await screen.findByRole('checkbox', { name: 'tv' }))
    await userEvent.click(screen.getByRole('button', { name: 'Scan' }))

    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/scan-2'))
    expect(body).toEqual({ sources: ['tv'], allow_empty: false })
  })

  it('says why a run was refused', async () => {
    server.use(statusRoute(status({ index: index() })),
               http.post('/v1/jobs/run', () => json({ title: 'Unknown source', status: 422, detail: 'no such source: x' }, 422)))
    renderApp('/jobs/new')
    await userEvent.click(await screen.findByRole('button', { name: 'Run' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Unknown source: no such source: x')
  })
})

it('keeps a run result that has every list empty readable', () => {
  expect(runResult({ failed: [], skipped: [] }).failed).toEqual([])
})
