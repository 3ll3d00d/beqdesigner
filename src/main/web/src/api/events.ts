// Following a job's events (GET /v1/jobs/{id}/events, Server-Sent Events). EventSource cannot send the Authorization header,
// so the stream is read with fetch; a dropped connection is resumed with Last-Event-ID until the job's final state event.
import type { Schemas } from './client'

export type JobEvent = Schemas['JobEvent']

const FINISHED = new Set(['succeeded', 'failed', 'cancelled', 'interrupted'])

export interface SseMessage {
  id?: string
  event?: string
  data: string
}

/** An incremental text/event-stream parser: push text as it arrives, get back the messages it completes. */
export class SseParser {
  private buffer = ''
  private id: string | undefined
  private event: string | undefined
  private data: string[] = []

  push(text: string): SseMessage[] {
    this.buffer += text
    const messages: SseMessage[] = []
    let newline: number
    while ((newline = this.buffer.search(/\r\n|\r|\n/)) >= 0) {
      const line = this.buffer.slice(0, newline)
      const width = this.buffer.startsWith('\r\n', newline) ? 2 : 1
      this.buffer = this.buffer.slice(newline + width)
      if (line === '') {
        if (this.data.length) messages.push({ id: this.id, event: this.event, data: this.data.join('\n') })
        this.event = undefined
        this.data = []
        continue
      }
      if (line.startsWith(':')) continue
      const colon = line.indexOf(':')
      const field = colon < 0 ? line : line.slice(0, colon)
      const value = colon < 0 ? '' : line.slice(colon + 1).replace(/^ /, '')
      if (field === 'data') this.data.push(value)
      else if (field === 'event') this.event = value
      else if (field === 'id') this.id = value
    }
    return messages
  }
}

export function isFinal(event: JobEvent): boolean {
  return event.type === 'state' && FINISHED.has(event.state)
}

export interface FollowOptions {
  token: () => string | null
  onEvent: (event: JobEvent) => void
  signal?: AbortSignal
  after?: number
  retryMs?: number
  maxRetries?: number
  fetch?: typeof globalThis.fetch
  baseUrl?: string
}

/**
 * Reads the job's events until its final state event or a clean end of the stream (resolves true: the service ends a
 * stream only when the job is over, and keeps a live one open with comments every 15 s), the signal aborts (false), or
 * the stream cannot be resumed after `maxRetries` failures in a row (rejects). A 401 or 404 is not retried.
 */
export async function followJob(jobId: string, options: FollowOptions): Promise<boolean> {
  const { token, onEvent, signal, retryMs = 2000, maxRetries = 5 } = options
  const fetcher = options.fetch ?? globalThis.fetch
  const url = `${options.baseUrl ?? ''}/v1/jobs/${encodeURIComponent(jobId)}/events`
  let last = options.after ?? 0
  let failures = 0
  while (!signal?.aborted) {
    try {
      const headers: Record<string, string> = { Accept: 'text/event-stream' }
      const value = token()
      if (value) headers.Authorization = `Bearer ${value}`
      if (last) headers['Last-Event-ID'] = String(last)
      const response = await fetcher(url, { headers, signal })
      if (response.status === 401 || response.status === 404) {
        throw Object.assign(new Error(`${response.status} ${response.statusText}`), { fatal: true })
      }
      if (!response.ok || !response.body) throw new Error(`${response.status} ${response.statusText}`)
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
      const parser = new SseParser()
      for (;;) {
        const { value: chunk, done } = await reader.read()
        if (done) return true   // a finished job whose final event has left the buffer ends like this
        for (const message of parser.push(chunk)) {
          const event = JSON.parse(message.data) as JobEvent
          last = Math.max(last, event.seq)
          failures = 0
          onEvent(event)
          if (isFinal(event)) return true
        }
      }
    } catch (error) {
      if (signal?.aborted) return false
      if ((error as { fatal?: boolean }).fatal || ++failures > maxRetries) throw error
    }
    if (signal?.aborted) return false
    await new Promise((resolve) => setTimeout(resolve, retryMs))
  }
  return false
}
