// The service, as the tests have it answer (MSW over fetch). Each test adds the handlers it needs with server.use().
import { http, HttpResponse } from 'msw/http'
import { setupServer } from 'msw/node'
import { afterAll, afterEach, beforeAll } from 'vitest'

import type { Schemas } from '../api/client'

export const TOKEN = 'test-token'

export const server = setupServer()

beforeAll(() => server.listen({ onUnhandledFrame: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

export function status(overrides: Partial<Schemas['ServiceStatus']> = {}): Schemas['ServiceStatus'] {
  return { version: '2.2.0', index: null, current_job: null, queued: 0, repository_writes: false,
           repositories_configured: false, ...overrides }
}

/** GET /v1/status answering `body` to the right token and 401 to any other. */
export function statusRoute(body: Schemas['ServiceStatus'] = status()) {
  return http.get('/v1/status', ({ request }) =>
    request.headers.get('Authorization') === `Bearer ${TOKEN}`
      ? HttpResponse.json(body)
      : HttpResponse.json({ title: 'Unauthorized', status: 401, detail: 'a valid bearer token is required' },
                          { status: 401 }))
}
