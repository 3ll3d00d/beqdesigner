// A job in a line, its state as a badge, and its progress, live while it runs.
import type { JobEvent } from '../api/events'
import { describeFilter } from '../api/filter'
import type { Job } from '../api/hooks'
import { duration, STATE_WORDS } from '../format'

export function describeJob(job: Job): string {
  switch (job.kind) {
    case 'scan':
      return job.request.sources?.length ? `Scan ${job.request.sources.join(', ')}` : 'Scan every source'
    case 'run':
      return `Run through ${job.request.through}: ${describeFilter(job.request.filter ?? {})}`
    case 'accept':
      return `${job.request.dry_run ? 'Dry run of bulk accept' : 'Bulk accept'}: ${describeFilter(job.request.filter ?? {})}`
  }
}

export function StateBadge({ state }: { state: string }) {
  return <span className={`badge state-${state}`}>{STATE_WORDS[state] ?? state}</span>
}

/** The newest ffmpeg report in the events, if its title is the one the run is on. */
function extraction(events: JobEvent[], titleId: string | undefined) {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]!
    if (event.type === 'extract_progress') return event.title_id === titleId || !titleId ? event : null
    if (event.type === 'run_progress') return null
  }
  return null
}

export function JobProgress({ job, events = [] }: { job: Job; events?: JobEvent[] }) {
  const progress = job.progress
  if (!progress || job.state !== 'running') return null
  const fraction = progress.total ? progress.done / progress.total : 0
  const ffmpeg = extraction(events, progress.id || undefined)
  return (
    <div className="progress">
      <progress value={progress.done} max={Math.max(progress.total, 1)} aria-label="Progress" />
      <p>
        {progress.done.toLocaleString()} of {progress.total.toLocaleString()} done ({Math.round(fraction * 100)}%)
        {progress.stage && progress.title && <> — {progress.stage} <strong>{progress.title}</strong></>}
        {ffmpeg?.percent !== undefined && ffmpeg.percent !== null && <> ({ffmpeg.percent}% extracted)</>}
      </p>
      {progress.remaining_seconds !== null && progress.remaining_seconds !== undefined && (
        <p className="muted">About {duration(progress.remaining_seconds)} to go
          {progress.per_hour ? `, ${progress.per_hour} an hour` : ''}.</p>
      )}
    </div>
  )
}
