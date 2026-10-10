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
           retry_failed: false, next_run_at: NOW, last_run: null, last_skip: null, ended: null, ...overrides }
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

export function candidate(index: number, overrides: Partial<Schemas['CandidateView']> = {}): Schemas['CandidateView'] {
  return { index, rejected: false, method: 'fitted', confidence: 0.9 - index * 0.3, mv_adjust_db: 4 - index,
           gain_reduction_db: -1, residual_db: null, residual_band_hz: null,
           commentary: index === 0 ? { note: 'top pick', found: ['shelf at 20 Hz', 'roll-off'] } : { note: 'alternative' },
           rejection_reasons: [], filters: { filters: [] }, ...overrides }
}

export function review(overrides: Partial<Schemas['Review']> = {}): Schemas['Review'] {
  return {
    id: 'a', title: 'Alien', year: '1979', status: 'pending', status_text: 'Waiting for a decision', digest: 'd1',
    candidates: [candidate(0), candidate(1)],
    rejected: [candidate(2, { rejected: true, method: 'non_parametric',
                             rejection_reasons: ['introduces a cliff of 53 dB/oct at 17 Hz'] })],
    chosen_index: null, declined: null, metadata: { title: 'Alien', year: '1979', audio_types: ['DTS-HD MA 5.1'] },
    metadata_problems: [], blocked: { accept: '', reject: '' }, in_flight: false,
    playback: 'none sent: the designer assumed its own playback chain', designer: 'rolloff', designer_build: 'beqforge 0.2.0',
    needs: 'review', detail: 'designed', reviewer_note: null,
    ...overrides,
  }
}

export function chart(candidateIndex: number | null): Schemas['Chart'] {
  const x = [0, 10, 20, 40, 80]
  const base = [{ name: 'Average audio track (all channels mixed)', kind: 'average' as const, filtered: false, x, y: [0, 0, 0, 0, 0] },
                { name: 'Peak audio track (all channels mixed)', kind: 'peak' as const, filtered: false, x, y: [10, 10, 10, 10, 10] }]
  const after = candidateIndex === null ? [] : [
    { name: 'Filtered average audio track (all channels mixed)', kind: 'average' as const, filtered: true, x, y: [4, 4, 3, 1, 0] }]
  return { candidate: candidateIndex, series: [...base, ...after] }
}

export function title(id: string, overrides: Partial<Schemas['Title']> = {}): Schemas['Title'] {
  return { id, title: id.toUpperCase(), display_name: id, year: '2001', kind: 'movie', source: 'films', path: `/m/${id}.mkv`,
           season: '', episodes: [], external_ids: {}, needs: 'review', tier: 'human', detail: 'designed', flags: [],
           extract_state: 'done', design_state: 'done', review_state: 'pending', publish_state: '', commit_state: '',
           confidence: 0.8, candidate_count: 2, failure: '', is_new: false, state_since: NOW, last_seen: NOW, ...overrides }
}
