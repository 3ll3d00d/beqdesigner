import { followJob, SseParser, type JobEvent } from './events'

function frame(event: Partial<JobEvent> & { seq: number; type: string }): string {
  return `id: ${event.seq}\nevent: ${event.type}\ndata: ${JSON.stringify({ at: '2026-10-10T00:00:00Z', text: '', ...event })}\n\n`
}

function streamOf(...chunks: string[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder()
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
}

describe('the event-stream parser', () => {
  it('completes a message only at its blank line, whatever the chunks', () => {
    const parser = new SseParser()
    expect(parser.push('id: 4\nevent: st')).toEqual([])
    expect(parser.push('ate\ndata: {"a":')).toEqual([])
    expect(parser.push('1}\r\n\r\n: keep-alive\n\ndata: x\ndata: y\n\n')).toEqual([
      { id: '4', event: 'state', data: '{"a":1}' },
      { id: '4', event: undefined, data: 'x\ny' },
    ])
  })
})

describe('following a job', () => {
  it('sends the token, reports each event, and ends at the final state', async () => {
    const seen: number[] = []
    const fetch = vi.fn(async (_url: string, init?: RequestInit) => {
      expect((init?.headers as Record<string, string>).Authorization).toBe('Bearer t')
      return streamOf(frame({ seq: 1, type: 'state', state: 'running' } as JobEvent),
                      frame({ seq: 2, type: 'run_progress', done: 1, total: 2 } as JobEvent),
                      frame({ seq: 3, type: 'state', state: 'succeeded' } as JobEvent), frame({ seq: 4, type: 'state', state: 'x' } as JobEvent))
    })
    const ended = await followJob('j', { token: () => 't', onEvent: (e) => seen.push(e.seq), fetch: fetch as never })
    expect(ended).toBe(true)
    expect(seen).toEqual([1, 2, 3])
    expect(fetch.mock.calls[0]![0]).toBe('/v1/jobs/j/events')
  })

  it('resumes after a dropped connection from the last event it had', async () => {
    const seen: number[] = []
    let call = 0
    // the first stream breaks after one event (a clean end would mean the job is over), the second attempt cannot connect
    const fetch = vi.fn(async (_url: string, init?: RequestInit) => {
      call += 1
      const headers = init?.headers as Record<string, string>
      if (call === 1) {
        let pulls = 0
        return new Response(new ReadableStream<Uint8Array>({
          pull(controller) {   // the event arrives, then the connection breaks on the next read
            if (pulls++ === 0) {
              controller.enqueue(new TextEncoder().encode(frame({ seq: 5, type: 'run_progress', done: 0, total: 1 } as JobEvent)))
            } else {
              controller.error(new TypeError('connection reset'))
            }
          },
        }))
      }
      if (call === 2) throw new TypeError('network')
      expect(headers['Last-Event-ID']).toBe('5')
      return streamOf(frame({ seq: 6, type: 'state', state: 'failed' } as JobEvent))
    })

    const ended = await followJob('j', { token: () => null, onEvent: (e) => seen.push(e.seq), fetch: fetch as never,
                                         retryMs: 1 })

    expect(ended).toBe(true)
    expect(seen).toEqual([5, 6])
    expect(call).toBe(3)
  })

  it('treats a clean end of the stream as the job being over', async () => {
    const fetch = vi.fn(async () => streamOf())
    expect(await followJob('j', { token: () => null, onEvent: () => {}, fetch: fetch as never })).toBe(true)
    expect(fetch).toHaveBeenCalledOnce()
  })

  it('gives up on a 401 at once, and after the retries on a service that is down', async () => {
    const unauthorised = vi.fn(async () => new Response(null, { status: 401, statusText: 'Unauthorized' }))
    await expect(followJob('j', { token: () => null, onEvent: () => {}, fetch: unauthorised as never })).rejects.toThrow('401')
    expect(unauthorised).toHaveBeenCalledOnce()

    const down = vi.fn(async () => new Response(null, { status: 503, statusText: 'Service Unavailable' }))
    await expect(followJob('j', { token: () => null, onEvent: () => {}, fetch: down as never, retryMs: 1, maxRetries: 2 }))
      .rejects.toThrow('503')
    expect(down).toHaveBeenCalledTimes(3)
  })

  it('stops when told to', async () => {
    const controller = new AbortController()
    const fetch = vi.fn(async () => {
      controller.abort()
      throw new DOMException('aborted', 'AbortError')
    })
    expect(await followJob('j', { token: () => null, onEvent: () => {}, fetch: fetch as never, signal: controller.signal }))
      .toBe(false)
  })
})
