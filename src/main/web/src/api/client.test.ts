import { http, HttpResponse } from 'msw/http'

import { server, statusRoute, TOKEN } from '../test/server'
import { makeClient, problemText, unwrap } from './client'

describe('the client', () => {
  it('sends the bearer token with every call', async () => {
    server.use(statusRoute())
    const client = makeClient({ token: () => TOKEN, onUnauthorized: () => {} })
    const { data } = await client.GET('/v1/status')
    expect(data?.version).toBe('2.2.0')
  })

  it('signs the person out on a 401', async () => {
    server.use(statusRoute())
    const onUnauthorized = vi.fn()
    const client = makeClient({ token: () => 'wrong', onUnauthorized })
    const result = await client.GET('/v1/status')
    expect(result.response.status).toBe(401)
    expect(onUnauthorized).toHaveBeenCalledOnce()
    expect(() => unwrap(result)).toThrow('Unauthorized: a valid bearer token is required')
  })

  it("says a Problem's title and detail", async () => {
    server.use(http.get('/v1/titles/:id/review', () =>
      HttpResponse.json({ title: 'Not designed', status: 409, detail: 'a has no design' }, { status: 409 })))
    const client = makeClient({ token: () => TOKEN, onUnauthorized: () => {} })
    const result = await client.GET('/v1/titles/{title_id}/review', { params: { path: { title_id: 'a' } } })
    expect(() => unwrap(result)).toThrow('Not designed: a has no design')
    expect(problemText(undefined)).toBe('The service did not answer.')
  })
})
