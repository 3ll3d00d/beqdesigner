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

export const STATE_WORDS: Record<string, string> = {
  queued: 'Queued',
  running: 'Running',
  succeeded: 'Succeeded',
  failed: 'Failed',
  cancelled: 'Cancelled',
  interrupted: 'Interrupted',
}

export const FINISHED_STATES = new Set(['succeeded', 'failed', 'cancelled', 'interrupted'])

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
