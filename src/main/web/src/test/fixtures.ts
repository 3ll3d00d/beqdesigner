// What the service answers, shaped by the generated types, so a fixture that no longer fits the interface fails typecheck.
import type { Schemas } from '../api/client'
import type { JobEvent } from '../api/events'

export const NOW = '2026-10-10T12:00:00Z'

export function index(overrides: Partial<Schemas['IndexStatus']> = {}): Schemas['IndexStatus'] {
  return {
    generation: 3, last_scan_at: NOW, titles: 120, new: 4, flags: {},
    counts: { attention: 2, extract: 30, design: 10, review: 7, publish: 3, commit: 1, done: 67 },
    sources: [{ name: 'films', kind: 'jriver', last_scanned: NOW, last_ok: NOW, last_error: '', item_count: 100 },
              { name: 'tv', kind: 'filesystem', last_scanned: NOW, last_ok: null, last_error: 'share not mounted', item_count: 20 }],
    ...overrides,
  }
}

export function runResult(overrides: Partial<Schemas['RunResult']> = {}): Schemas['RunResult'] {
  return {
    through: 'design', selected: 3, extracted: ['a', 'b'], cached: [], designed: ['a'], design_cached: [],
    failed: [{ id: 'b', message: 'designer said no' }], failed_earlier: [], unavailable: [], meta_unresolved: [],
    project_edit_preserved: [], seasons: {}, published: [], publish_errors: [], committed: null, commit_error: '',
    skipped: [{ id: 'c', title: 'Cube', reason: 'waiting for review' }], cancelled: false, stopped: '', attempted: ['a', 'b'],
    not_run: [], counts: {},
    ...overrides,
  }
}

export function runJob(overrides: Partial<Schemas['RunJob']> = {}): Schemas['RunJob'] {
  return {
    kind: 'run', id: 'job-1', origin: 'api', state: 'succeeded', submitted_at: NOW, started_at: NOW, finished_at: NOW,
    progress: null, error: null, joined_to: null,
    request: { filter: { needs: ['extract', 'design'], new_since_scan: false, kind: 'movie' }, through: 'design',
               scan_first: true, retry_failed: false },
    result: runResult(),
    ...overrides,
  }
}

export function scanJob(overrides: Partial<Schemas['ScanJob']> = {}): Schemas['ScanJob'] {
  return {
    kind: 'scan', id: 'scan-1', origin: 'schedule', state: 'failed', submitted_at: NOW, started_at: NOW, finished_at: NOW,
    progress: null, error: 'profile unreadable', joined_to: null, request: { sources: [], allow_empty: false }, result: null,
    ...overrides,
  }
}

export function schedule(overrides: Partial<Schemas['Schedule']> = {}): Schemas['Schedule'] {
  return { enabled: true, interval_minutes: 60, filter: { new_since_scan: false, kind: 'movie' }, through: 'design',
           retry_failed: false, next_run_at: NOW, last_run: null, last_skip: null, ...overrides }
}

export function event(seq: number, fields: Record<string, unknown>): JobEvent {
  return { seq, at: NOW, text: `event ${seq}`, ...fields } as JobEvent
}

/** A text/event-stream body of these events, then the end. */
export function eventStream(events: JobEvent[]): ReadableStream<Uint8Array> {
  const text = events.map((e) => `id: ${e.seq}\nevent: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join('')
  return new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text))
      controller.close()
    },
  })
}
