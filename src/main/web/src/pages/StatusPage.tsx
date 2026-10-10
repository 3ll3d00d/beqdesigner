// What the service is doing (design/web-app.md §4): the pipeline strip, the job running now followed live, the designer,
// the schedule and the sources.
import { Link } from 'react-router'

import type { Schemas } from '../api/client'
import { filterToQuery } from '../api/filter'
import { useJobEvents, useSaveSchedule, useSchedule, useStatus, useTriggerSchedule, type Job } from '../api/hooks'
import { describeJob, JobProgress, StateBadge } from '../components/JobBits'
import { count, NEEDS_ORDER, NEEDS_WORDS, when } from '../format'

function Strip({ counts }: { counts: Record<string, number> }) {
  return (
    <ol className="strip" aria-label="What titles need">
      {NEEDS_ORDER.map((need) => (
        <li key={need} className={`tier-${need}`}>
          <Link to={`/titles?${filterToQuery({ needs: [need] })}`}>
            <span className="n">{(counts[need] ?? 0).toLocaleString()}</span>
            <span className="label">{NEEDS_WORDS[need]}</span>
          </Link>
        </li>
      ))}
    </ol>
  )
}

function CurrentJob({ job, queued }: { job: Job | null; queued: number }) {
  const { events } = useJobEvents(job?.state === 'running' ? job.id : null)
  return (
    <section className="card" aria-labelledby="current-job">
      <h2 id="current-job">Job</h2>
      {job ? (
        <>
          <p><StateBadge state={job.state} /> <Link to={`/jobs/${job.id}`}>{describeJob(job)}</Link></p>
          <JobProgress job={job} events={events} />
        </>
      ) : <p className="muted">None running.</p>}
      <p className="muted">{queued ? `${count(queued, 'job')} waiting.` : 'Nothing waiting.'}{' '}
        <Link to="/jobs/new">New job…</Link></p>
    </section>
  )
}

function Designer({ designer }: { designer: Schemas['DesignerStatus'] | null | undefined }) {
  if (!designer) return null
  const state = designer.reachable === null ? 'not asked' : designer.reachable ? 'answering' : 'not answering'
  return (
    <section className="card" aria-labelledby="designer">
      <h2 id="designer">Designer</h2>
      <p><strong>{designer.name || '—'}</strong>: <span className={designer.reachable === false ? 'problem' : ''}>{state}</span></p>
      {designer.detail && <p className="muted">{designer.detail}</p>}
    </section>
  )
}

function ScheduleCard() {
  const schedule = useSchedule()
  const save = useSaveSchedule()
  const trigger = useTriggerSchedule()
  const data = schedule.data
  if (!data) return null
  const update = { enabled: data.enabled, interval_minutes: data.interval_minutes, filter: data.filter,
                   through: data.through, retry_failed: data.retry_failed }
  const problem = save.error?.message ?? trigger.error?.message
  return (
    <section className="card" aria-labelledby="schedule">
      <h2 id="schedule">Schedule</h2>
      <p>
        {data.enabled ? `Every ${data.interval_minutes} min, through ${data.through}.` : 'Paused.'}
        {data.enabled && data.next_run_at && <> Next at {when(data.next_run_at)}.</>}
      </p>
      {data.last_run && <p className="muted">Last: <StateBadge state={data.last_run.state} />{' '}
        <Link to={`/jobs/${data.last_run.job_id}`}>{when(data.last_run.finished_at)}</Link></p>}
      {data.last_skip && <p className="muted">Last skipped: {data.last_skip}</p>}
      <div className="actions">
        <button type="button" disabled={save.isPending} onClick={() => save.mutate({ ...update, enabled: !data.enabled })}>
          {data.enabled ? 'Pause' : 'Resume'}
        </button>
        <button type="button" className="secondary" disabled={trigger.isPending} onClick={() => trigger.mutate()}>
          Run now
        </button>
      </div>
      {problem && <p role="alert" className="problem">{problem}</p>}
    </section>
  )
}

function Sources({ sources }: { sources: Schemas['SourceStatus'][] }) {
  return (
    <section className="card wide" aria-labelledby="sources">
      <h2 id="sources">Sources</h2>
      <table>
        <thead><tr><th>Source</th><th>Kind</th><th>Titles</th><th>Last listed</th><th>Problem</th></tr></thead>
        <tbody>
          {sources.map((s) => (
            <tr key={s.name}>
              <td>{s.name}</td><td>{s.kind}</td><td>{s.item_count.toLocaleString()}</td>
              <td>{when(s.last_ok)}</td><td className={s.last_error ? 'problem' : ''}>{s.last_error || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

export function StatusPage() {
  const status = useStatus()
  return (
    <section>
      <h1>Status</h1>
      {status.isPending && <p className="muted">Asking the service…</p>}
      {status.isError && <p role="alert" className="problem">{status.error.message}</p>}
      {status.data && (
        <>
          {status.data.index ? (
            <>
              <Strip counts={status.data.index.counts} />
              <p className="muted">
                {count(status.data.index.titles, 'title')}, {status.data.index.new.toLocaleString()} new; last scanned{' '}
                {when(status.data.index.last_scan_at)}.
              </p>
            </>
          ) : <p>Not scanned yet. <Link to="/jobs/new?kind=scan">Scan the sources…</Link></p>}
          <p className="muted">Service <span>{status.data.version}</span>.</p>
          <div className="cards">
            <CurrentJob job={status.data.current_job as Job | null} queued={status.data.queued} />
            <Designer designer={status.data.designer} />
            <ScheduleCard />
            {status.data.index && <Sources sources={status.data.index.sources} />}
          </div>
          {status.data.tmdb === false && <p className="muted">No TMDB key is set: titles are designed with the library's own metadata.</p>}
        </>
      )}
    </section>
  )
}
