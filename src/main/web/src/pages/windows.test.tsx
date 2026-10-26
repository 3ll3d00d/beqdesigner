// When jobs may run (design/job-windows.md §5): a window given with a new job, a job waiting or paused, its window changed,
// and the default on Status, against a mocked service.
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { describeWindow, parseRanges } from '../components/WindowField'
import { when } from '../format'
import { index, runJob, scanJob, schedule } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

beforeEach(() => saveToken(TOKEN, false))

function json(body: unknown, init?: number) {
  return HttpResponse.json(body as never, init ? { status: init } : undefined)
}

const LATER = '2026-10-11T01:00:00Z'
const never = () => new HttpResponse(new ReadableStream(), { headers: { 'Content-Type': 'text/event-stream' } })
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
    const queued = scanJob({ id: 'scan-2', state: 'queued', error: null })
    server.use(statusRoute(status({ index: index() })),
               http.post('/v1/jobs/scan', async ({ request }) => {
                 bodies.push(await request.json())
                 return json(queued, 202)
               }),
               http.get('/v1/jobs/scan-2', () => json(queued)), http.get('/v1/jobs/scan-2/events', never))
    const router = renderApp('/jobs/new?kind=scan')

    await userEvent.click(await screen.findByRole('radio', { name: 'Only between' }))
    await userEvent.type(screen.getByRole('textbox', { name: 'Times' }), '01:00-07:00')
    await userEvent.click(screen.getByRole('button', { name: 'Scan' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/jobs/scan-2'))
    expect(bodies).toEqual([{ sources: [], allow_empty: false, window: ['01:00-07:00'] }])
  })

  it('may run any time whatever the default says', async () => {
    let body: unknown
    server.use(statusRoute(status({ index: index() })),
               http.post('/v1/jobs/run', async ({ request }) => {
                 body = await request.json()
                 return json(runJob({ state: 'queued', finished_at: null, result: null }), 202)
               }),
               http.get('/v1/jobs/job-1', () => json(runJob())), http.get('/v1/jobs/job-1/log', () => json([])))
    renderApp('/jobs/new')
    await userEvent.click(await screen.findByRole('radio', { name: 'Any time' }))
    await userEvent.click(screen.getByRole('button', { name: 'Run' }))
    await waitFor(() => expect(body).toMatchObject({ window: [] }))
  })
})

describe('a job outside its window', () => {
  it('says in the list when a waiting job is taken and a paused one carries on', async () => {
    server.use(http.get('/v1/jobs', () => json({ jobs: [
      runJob({ id: 'p', state: 'paused', finished_at: null, result: null, waiting_until: LATER, window: ['01:00-07:00'] }),
      scanJob({ id: 'w', state: 'queued', finished_at: null, error: null, waiting_until: LATER }),
    ] })))
    renderApp('/jobs')
    expect(await screen.findByText(`until ${when(LATER)}`)).toBeInTheDocument()
    expect(screen.getByText(`waits for ${when(LATER)}`)).toBeInTheDocument()
    expect(screen.getAllByText('Paused').length).toBeGreaterThan(0)
  })

  it('shows a paused job\'s window and parts, and changes its window', async () => {
    let asked: unknown
    const paused = runJob({ state: 'paused', finished_at: null, result: null, window: ['01:00-02:00'], window_open: false,
                            waiting_until: LATER, parts: 2, paused_at: ['2026-10-10T02:00:00Z'] })
    server.use(http.get('/v1/job-window', () => json(nightly)),
               http.get('/v1/jobs/job-1', () => json(paused)), http.get('/v1/jobs/job-1/events', never),
               http.put('/v1/jobs/job-1/window', async ({ request }) => {
                 asked = await request.json()
                 return json({ ...paused, window: [], waiting_until: null })
               }))
    renderApp('/jobs/job-1')

    const facts = await screen.findByText(/01:00-02:00 \(Europe\/London\)/)
    expect(within(facts).getByText(/paused until/)).toBeInTheDocument()
    expect(screen.getByText(/Started 2 times; paused/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Drop' })).toBeInTheDocument()   // a paused job is dropped, as a queued one

    await userEvent.click(screen.getByRole('button', { name: 'Change…' }))
    await userEvent.click(screen.getByRole('radio', { name: 'Any time' }))
    await userEvent.click(screen.getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(asked).toEqual({ window: [] }))
    expect(await screen.findByText(/any time \(Europe\/London\)/)).toBeInTheDocument()
  })

  it('says why a window was refused', async () => {
    const queued = runJob({ state: 'queued', finished_at: null, result: null })
    server.use(http.get('/v1/jobs/job-1', () => json(queued)), http.get('/v1/jobs/job-1/events', never),
               http.put('/v1/jobs/job-1/window', () => json({ title: 'Unprocessable', status: 422,
                                                              detail: "'night': a range is HH:MM-HH:MM" }, 422)))
    renderApp('/jobs/job-1')
    await userEvent.click(await screen.findByRole('button', { name: 'Change…' }))
    await userEvent.click(screen.getByRole('radio', { name: 'Only between' }))
    await userEvent.type(screen.getByRole('textbox', { name: 'Times' }), 'night')
    await userEvent.click(screen.getByRole('button', { name: 'Save' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('a range is HH:MM-HH:MM')
  })
})

describe('the default window', () => {
  it('is on Status, says when it next opens, and is saved there', async () => {
    let saved: unknown
    server.use(statusRoute(status({ index: index() })), http.get('/v1/schedule', () => json(schedule())),
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
