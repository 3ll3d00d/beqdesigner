// How the app says times, durations and the pipeline's words.

export const NEEDS_ORDER = ['attention', 'extract', 'design', 'review', 'publish', 'commit', 'done'] as const
export type NeedsWord = (typeof NEEDS_ORDER)[number]

export const NEEDS_WORDS: Record<NeedsWord, string> = {
  attention: 'Attention',
  extract: 'Extract',
  design: 'Design',
  review: 'Review',
  publish: 'Publish',
  commit: 'Commit',
  done: 'Done',
}

/**
 * The status strip's words: each count is how many titles are waiting for that stage, so it says so ("12 to extract"),
 * rather than the bare stage, which beside a number read as how many had been done.
 */
export const STRIP_WORDS: Record<NeedsWord, string> = {
  attention: 'Need attention',
  extract: 'To extract',
  design: 'To design',
  review: 'To review',
  publish: 'To publish',
  commit: 'To commit',
  done: 'Done',
}

/** An execution's states. */
export const STATE_WORDS: Record<string, string> = {
  queued: 'Queued',
  running: 'Running',
  succeeded: 'Succeeded',
  failed: 'Failed',
  cancelled: 'Cancelled',
  stopped: 'Stopped',
  interrupted: 'Interrupted',
}

/** A job's states. */
export const JOB_STATE_WORDS: Record<string, string> = {
  active: 'Active',
  paused: 'Paused',
  done: 'Done',
  removed: 'Removed',
}

export const STOPPED_WORDS: Record<string, string> = {
  window: 'its window closed',
  paused: 'its job was paused',
  removed: 'its job was removed',
}

export const FINISHED_STATES = new Set(['succeeded', 'failed', 'cancelled', 'stopped', 'interrupted'])

export function when(value: string | null | undefined, now: Date = new Date()): string {
  if (!value) return '—'
  const at = new Date(value)
  if (Number.isNaN(at.getTime())) return value
  const sameDay = at.toDateString() === now.toDateString()
  return sameDay
    ? at.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    : at.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (h) return `${h} h ${m} min`
  if (m) return `${m} min ${s % 60} s`
  return `${s} s`
}

export function between(start: string | null | undefined, end: string | null | undefined): string {
  if (!start) return '—'
  const finish = end ? new Date(end) : new Date()
  return duration((finish.getTime() - new Date(start).getTime()) / 1000)
}

export function count(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`
}

/** A metadata value as words: a list joined, and a TMDB-style `{ name }` object (a genre, say) as its name. */
export function metadataText(value: unknown): string {
  if (Array.isArray(value)) return value.map(metadataText).filter(Boolean).join(', ')
  if (value && typeof value === 'object') {
    const named = value as { name?: unknown }
    return named.name !== undefined ? String(named.name) : Object.values(value).map(metadataText).filter(Boolean).join(' ')
  }
  return String(value ?? '')
}

/**
 * Why a designer endpoint is down, in a few words, from the service's detail (which carries the whole connection
 * error): the OS's reason, a timeout, an HTTP status, or the start of the text.
 */
export function briefly(detail: string | null | undefined): string {
  const text = detail ?? ''
  const errno = /\[Errno -?\d+\] ([^"')\]]+)/.exec(text)
  if (errno) return errno[1]!.trim().replace(/^\w/, (c) => c.toLowerCase())
  if (/timed? ?out|ConnectTimeout|ReadTimeout/i.test(text)) return 'timed out'
  if (/connection refused/i.test(text)) return 'connection refused'
  if (/Name or service not known|nodename nor servname|getaddrinfo/i.test(text)) return 'unknown host'
  const status = /\bHTTP (\d{3})\b/.exec(text)
  if (status) return `HTTP ${status[1]}`
  return text.length > 60 ? `${text.slice(0, 57)}…` : text
}
