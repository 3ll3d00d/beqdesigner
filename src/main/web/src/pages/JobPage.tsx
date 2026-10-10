// One job: what was asked, how it went, its log (live while it runs), and Cancel.
import { Link, useParams } from 'react-router'

import type { Schemas } from '../api/client'
import type { JobEvent } from '../api/events'
import { useCancelJob, useJob, useJobEvents, useJobLog, type Job } from '../api/hooks'
import { describeJob, JobProgress, StateBadge } from '../components/JobBits'
import { between, count, FINISHED_STATES, when } from '../format'

function Titles({ heading, items }: { heading: string; items: { id: string; message?: string; reason?: string; title?: string }[] }) {
  if (!items.length) return null
  return (
    <details open={items.length <= 10}>
      <summary>{heading} ({items.length.toLocaleString()})</summary>
      <ul className="titles">
        {items.map((item) => (
          <li key={item.id}>
            <Link to={`/titles/${encodeURIComponent(item.id)}`}>{item.title || item.id}</Link>
            {(item.message || item.reason) && <span className="muted"> — {item.message || item.reason}</span>}
          </li>
        ))}
      </ul>
    </details>
  )
}

const ids = (values: string[]) => values.map((id) => ({ id }))

function ScanOutcome({ result }: { result: Schemas['ScanResult'] }) {
  const errors = Object.entries(result.errors)
  return (
    <>
      <p>{count(result.titles, 'title')}: {result.new.length.toLocaleString()} new, {result.gone.length.toLocaleString()} gone.</p>
      {errors.map(([source, why]) => <p key={source} className="problem">{source}: {why}</p>)}
      <Titles heading="New" items={ids(result.new)} />
    </>
  )
}

function RunOutcome({ result }: { result: Schemas['RunResult'] }) {
  return (
    <>
      {result.scan && <ScanOutcome result={result.scan} />}
      <p>
        {count(result.selected, 'title')} selected: {result.extracted.length.toLocaleString()} extracted
        ({result.cached.length.toLocaleString()} already), {result.designed.length.toLocaleString()} designed
        ({result.design_cached.length.toLocaleString()} already)
        {result.published.length ? `, ${result.published.length.toLocaleString()} published` : ''}.
      </p>
      {result.cancelled && <p className="problem">Cancelled: {count(result.not_run.length, 'title')} not run.</p>}
      {result.stopped && <p className="problem">{result.stopped}</p>}
      {result.commit_error && <p className="problem">Commit: {result.commit_error}</p>}
      {result.committed && <p>Committed to the catalogue.</p>}
      <Titles heading="Failed" items={result.failed} />
      <Titles heading="Not done: something it needs was unavailable" items={result.unavailable ?? []} />
      <Titles heading="Not tried: failed before with the same source and settings" items={result.failed_earlier} />
      <Titles heading="Not published" items={result.publish_errors.map((e) => ({ id: e.id, message: e.message || e.error }))} />
      <Titles heading="Designed" items={ids(result.designed)} />
      <Titles heading="Skipped" items={result.skipped} />
    </>
  )
}

function AcceptOutcome({ result }: { result: Schemas['AcceptResult'] }) {
  return (
    <>
      <p>{result.dry_run ? 'Would accept' : 'Accepted'} {count(result.accepted.length, 'title')} at confidence {result.threshold}
        {' '}or more; {result.below_threshold.toLocaleString()} below it.</p>
      {result.note && <p className="muted">{result.note}</p>}
      <Titles heading={result.dry_run ? 'Would accept' : 'Accepted'} items={ids(result.accepted)} />
      <Titles heading="Left for a person" items={result.excluded} />
    </>
  )
}

function Outcome({ job }: { job: Job }) {
  if (job.error) return <p role="alert" className="problem">{job.error}</p>
  if (!job.result) return null
  switch (job.kind) {
    case 'scan': return <ScanOutcome result={job.result} />
    case 'run': return <RunOutcome result={job.result} />
    case 'accept': return <AcceptOutcome result={job.result} />
  }
}

function Log({ events }: { events: JobEvent[] }) {
  if (!events.length) return <p className="muted">Nothing reported yet.</p>
  return (
    <ol className="log" aria-label="Log">
      {events.map((event) => (
        <li key={event.seq} className={`event-${event.type}`}>
          <time dateTime={event.at}>{when(event.at)}</time> {event.text}
        </li>
      ))}
    </ol>
  )
}

export function JobPage() {
  const { jobId = '' } = useParams()
  const job = useJob(jobId)
  const running = job.data ? !FINISHED_STATES.has(job.data.state) : false
  const live = useJobEvents(running ? jobId : null)
  const history = useJobLog(jobId, !!job.data && !running)
  const cancel = useCancelJob()

  if (job.isPending) return <p className="muted">Loading…</p>
  if (job.isError) return <p role="alert" className="problem">{job.error.message}</p>
  const data = job.data
  const events = running ? live.events : history.data ?? []
  return (
    <section>
      <p><Link to="/jobs">← Jobs</Link></p>
      <div className="title-row">
        <h1>{describeJob(data)}</h1>
        {running && (
          <button type="button" className="danger" disabled={cancel.isPending} onClick={() => cancel.mutate(data.id)}>
            {data.state === 'queued' ? 'Drop' : 'Cancel'}
          </button>
        )}
      </div>
      {cancel.error && <p role="alert" className="problem">{cancel.error.message}</p>}
      {cancel.isSuccess && data.state === 'running' && <p className="muted">Stopping after the title in hand…</p>}
      <dl className="facts">
        <dt>State</dt><dd><StateBadge state={data.state} /></dd>
        <dt>From</dt><dd>{data.origin === 'schedule' ? 'the schedule' : 'a request'}</dd>
        <dt>Submitted</dt><dd>{when(data.submitted_at)}</dd>
        <dt>Started</dt><dd>{when(data.started_at)}</dd>
        <dt>Took</dt><dd>{data.started_at ? between(data.started_at, data.finished_at) : '—'}</dd>
        {data.joined_to && <><dt>Joined</dt><dd><Link to={`/jobs/${data.joined_to}`}>the run in progress</Link></dd></>}
      </dl>
      <JobProgress job={data} events={live.events} />
      <Outcome job={data} />
      <h2>Log</h2>
      {live.problem && <p role="alert" className="problem">Lost the live log: {live.problem}</p>}
      <Log events={events} />
    </section>
  )
}
