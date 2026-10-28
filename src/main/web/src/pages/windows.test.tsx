// When jobs may run (design/job-windows.md §5, design/jobs-and-executions.md §5): a window given with a new job, a job
// waiting for it, its window changed, and the default on Status, against a mocked service.
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { describeWindow, parseRanges } from '../components/WindowField'
import { when } from '../format'
import { index, runExecution, runJob, scanJob } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

beforeEach(() => saveToken(TOKEN, false))

function json(body: unknown, init?: number) {
  return HttpResponse.json(body as never, init ? { status: init } : undefined)
}

const LATER = '2026-10-11T01:00:00Z'
const nightly = { window: ['01:00-07:00'], time_zone: 'Europe/London', open_now: false, next_change: LATER }

it('reads the ranges typed a line or a comma each, and says a window in words', () => {
  expect(parseRanges(' 01:00-07:00\n\n13:00-15:00 , 22:00-23:00 ')).toEqual(['01:00-07:00', '13:00-15:00', '22:00-23:00'])
  expect(describeWindow(null, nightly)).toBe('the default (01:00-07:00, Europe/London)')
  expect(describeWindow([], nightly)).toBe('any time (Europe/London)')
  expect(describeWindow(['22:00-06:00'])).toBe('22:00-06:00')
})

describe('a new job', () => {
  it('sends the times it may run in, and none when it follows the default', async () => {
    const bodies: unknown[] = []
    const made = scanJob({ id: 'scan-2', window: ['01:00-07:00'], next_at: LATER, executions: 0, last_execution: null })
    server.use(statusRoute(status({ index: index() })),
               http.post('/v1/jobs', async ({ request }) => {
                 bodies.push(await request.json())
                 return json(made, 201)
               }),
               http.get('/v1/jobs/scan-2', () => json(made)))
    const router = renderApp('/jobs/new?kind=scan')

    await userEvent.click(await screen.findByRole('radio', { name: 'Only between' }))
    await userEvent.type(screen.getByRole('textbox', { name: 'Times' }), '01:00-07:00')
    await userEvent.click(screen.getByRole('button', { name: 'Scan' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/scan-2'))
    expect(bodies).toEqual([{ kind: 'scan', request: { sources: [], allow_empty: false }, window: ['01:00-07:00'] }])
    expect((await screen.findByText('Next')).nextElementSibling).toHaveTextContent(when(LATER))
  })

  it('may run any time whatever the default says', async () => {
    let body: unknown
    server.use(statusRoute(status({ index: index() })),
               http.post('/v1/jobs', async ({ request }) => {
                 body = await request.json()
                 return json(runJob(), 201)
               }),
               http.get('/v1/jobs/job-1', () => json(runJob())))
    renderApp('/jobs/new')
    await userEvent.click(await screen.findByRole('radio', { name: 'Any time' }))
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await waitFor(() => expect(body).toMatchObject({ kind: 'run', window: [] }))
  })
})

describe('a job outside its window', () => {
  it('says on Status when a waiting job runs next', async () => {
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/jobs', () => json({ jobs: [runJob({ window: ['01:00-07:00'], next_at: LATER, executions: 1 })] })))
    renderApp('/')
    const table = await screen.findByRole('table', { name: 'Jobs' })
    expect(within(table).getByText(when(LATER))).toBeInTheDocument()
    expect(within(table).getByText('01:00-07:00')).toBeInTheDocument()
  })

  it('changes its window', async () => {
    let asked: unknown
    const waiting = runJob({ window: ['01:00-02:00'], next_at: LATER })
    server.use(statusRoute(status({ index: index() })), http.get('/v1/job-window', () => json(nightly)),
               http.get('/v1/jobs/job-1', () => json(asked ? { ...waiting, window: [], next_at: null } : waiting)),
               http.patch('/v1/jobs/job-1', async ({ request }) => {
                 asked = await request.json()
                 return json({ ...waiting, window: [], next_at: null })
               }))
    renderApp('/jobs/job-1')

    expect(await screen.findByText('01:00-02:00 (Europe/London)')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Edit…' }))
    await userEvent.click(screen.getByRole('radio', { name: 'Any time' }))
    await userEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(asked).toEqual({ name: 'Movies through design', window: [] }))
    expect(await screen.findByText(/any time \(Europe\/London\)/)).toBeInTheDocument()
  })

  it('says why a window was refused', async () => {
    server.use(statusRoute(status({ index: index() })), http.get('/v1/jobs/job-1', () => json(runJob())),
               http.patch('/v1/jobs/job-1', () => json({ title: 'Unprocessable', status: 422,
                                                         detail: "'night': a range is HH:MM-HH:MM" }, 422)))
    renderApp('/jobs/job-1')
    await userEvent.click(await screen.findByRole('button', { name: 'Edit…' }))
    await userEvent.click(screen.getByRole('radio', { name: 'Only between' }))
    await userEvent.type(screen.getByRole('textbox', { name: 'Times' }), 'night')
    await userEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('a range is HH:MM-HH:MM')
  })

  it('says an execution its window stopped is waiting for its job to carry on', async () => {
    server.use(http.get('/v1/executions/ex-1', () => json(runExecution({ state: 'stopped', stopped_because: 'window' }))),
               http.get('/v1/executions/ex-1/log', () => json([])))
    renderApp('/executions/ex-1')
    expect((await screen.findByText('State')).nextElementSibling).toHaveTextContent('Stopped — its window closed')
  })
})

describe('the default window', () => {
  it('is on Status, says when it next opens, and is saved there', async () => {
    let saved: unknown
    server.use(statusRoute(status({ index: index() })),
               http.get('/v1/job-window', () => json({ window: [], time_zone: 'Europe/London', open_now: true,
                                                       next_change: null })),
               http.put('/v1/job-window', async ({ request }) => {
                 saved = await request.json()
                 return json(nightly)
               }))
    renderApp('/')
    expect(await screen.findByText('Any time')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Change…' }))
    await userEvent.type(screen.getByRole('textbox', { name: 'Default times' }), '01:00-07:00')
    await userEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(saved).toEqual({ window: ['01:00-07:00'] }))
    expect(await screen.findByText('01:00-07:00 (Europe/London)')).toBeInTheDocument()
    expect(screen.getByText(new RegExp(`closed now from ${when(LATER).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`)))
      .toBeInTheDocument()
  })
})
