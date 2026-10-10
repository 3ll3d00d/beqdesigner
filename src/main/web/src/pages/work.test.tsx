// The Status and Jobs screens (W4) against a mocked service.
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { filterFromQuery, filterToQuery } from '../api/filter'
import { saveToken } from '../api/token'
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
                     source: 'films', new_since_scan: true, ids: ['a'] }
    expect(filterFromQuery(filterToQuery(filter))).toEqual(filter)
    expect(filterFromQuery(new URLSearchParams('needs=nonsense&kind=film&year=%20'))).toEqual({})
  })
})

describe('status', () => {
  it('shows what titles need, each linked to the list of them', async () => {
    server.use(statusRoute(status({ index: index() })), http.get('/v1/schedule', () => json(schedule())))
    const router = renderApp('/')

    const strip = await screen.findByRole('list', { name: 'What titles need' })
    expect(within(strip).getByText('Review').closest('a')).toHaveAttribute('href', '/titles?needs=review')
    expect(within(strip).getByText('30')).toBeInTheDocument()
    expect(screen.getByText('share not mounted')).toBeInTheDocument()
    await userEvent.click(within(strip).getByText('Attention'))
    expect(router.state.location.search).toBe('?needs=attention')
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
    expect(await screen.findByText(/42% extracted/)).toBeInTheDocument()
    expect(screen.getByText(/About 10 min 0 s to go/)).toBeInTheDocument()
    expect(screen.getByText(/2 jobs waiting/)).toBeInTheDocument()
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
    expect(await screen.findByText('Ended: all 2 listed titles are done')).toBeInTheDocument()
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
                                 through: 'design', scan_first: true, retry_failed: false })

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
