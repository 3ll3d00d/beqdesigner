// What the service is doing (design/web-app.md §4): the pipeline strip, the job running now followed live, the designer,
// the schedule and the sources.
import { Link } from 'react-router'

import type { Schemas } from '../api/client'
import { filterToQuery } from '../api/filter'
import { useJobEvents, useSaveSchedule, useSchedule, useStatus, useTriggerSchedule, type Job } from '../api/hooks'
import type { JobEvent } from '../api/events'
import { describeJob, extraction, StateBadge } from '../components/JobBits'
import { ScanSources } from '../components/ScanSources'
import { count, duration, NEEDS_ORDER, STRIP_WORDS, when } from '../format'

function Strip({ counts }: { counts: Record<string, number> }) {
  return (
    <ol className="strip" aria-label="What titles need">
      {NEEDS_ORDER.map((need) => (
        <li key={need} className={`tier-${need}`}>
          <Link to={`/titles?${filterToQuery({ needs: [need] })}`}>
            <span className="n">{(counts[need] ?? 0).toLocaleString()}</span>
            <span className="label">{STRIP_WORDS[need]}</span>
          </Link>
        </li>
      ))}
    </ol>
  )
}

type Designs = Schemas['EndpointDesign'][]

/** What the job is doing now: the title in hand (with ffmpeg's progress on it), and the designs out on the designer. */
function Activity({ job, events, designing }: { job: Job; events: JobEvent[]; designing: Designs }) {
  const progress = job.progress
  const ffmpeg = extraction(events, progress?.id || undefined)
  const inHand = progress?.stage && progress.title && !(progress.stage === 'design' && designing.length)
  if (!inHand && !designing.length) return <>—</>
  return (
    <>
      {inHand && <>{progress!.stage} <strong>{progress!.title}</strong>
        {ffmpeg?.percent !== undefined && ffmpeg.percent !== null && <> ({ffmpeg.percent}% extracted)</>}</>}
      {inHand && designing.length > 0 && '; '}
      {designing.length > 0 && <>designing {designing.map((d, i) => (
        <span key={d.title_id}>{i ? ', ' : ''}<Link to={`/titles/${d.title_id}`}>{d.title || d.title_id}</Link></span>
      ))}</>}
    </>
  )
}

function CurrentJob({ job, queued, designing }: { job: Job | null; queued: number; designing: Designs }) {
  const { events } = useJobEvents(job?.state === 'running' ? job.id : null)
  const progress = job?.state === 'running' ? job.progress : null
  return (
    <section className="card" aria-labelledby="current-job">
      <h2 id="current-job">Job</h2>
      {job ? (
        <dl className="facts" aria-label="Current job">
          <dt>Job</dt><dd><StateBadge state={job.state} /> <Link to={`/jobs/${job.id}`}>{describeJob(job)}</Link></dd>
          <dt>Started</dt><dd>{when(job.started_at)}</dd>
          <dt>Estimated end</dt>
          <dd>{progress?.estimated_finish ? when(progress.estimated_finish) : '—'}
            {progress?.remaining_seconds != null && <span className="muted"> (about {duration(progress.remaining_seconds)} to go)</span>}</dd>
          <dt>Progress</dt>
          <dd>{progress ? <>{progress.done.toLocaleString()} of {progress.total.toLocaleString()}
            {progress.per_hour ? <span className="muted">, {progress.per_hour} an hour</span> : null}</> : '—'}</dd>
          <dt>Activity</dt><dd><Activity job={job} events={events} designing={job.state === 'running' ? designing : []} /></dd>
        </dl>
      ) : <p className="muted">None running.</p>}
      {progress && <progress value={progress.done} max={Math.max(progress.total, 1)} aria-label="Progress" />}
      <p className="muted">{queued ? `${count(queued, 'job')} waiting.` : 'Nothing waiting.'}{' '}
        <Link to="/jobs/new">New job…</Link></p>
    </section>
  )
}

function Designer({ designer }: { designer: Schemas['DesignerStatus'] | null | undefined }) {
  if (!designer) return null
  const endpoints = designer.endpoints ?? []
  return (
    <section className="card wide" aria-labelledby="designer">
      <h2 id="designer">Designer{designer.name ? `: ${designer.name}` : ''}</h2>
      {(designer.design_disagreement?.length ?? 0) > 0 && (
        <div className="problem" role="alert" aria-label="Designers disagree">
          <p>The designers that are up would not design alike, so a redesign answers differently by endpoint:</p>
          <ul>{designer.design_disagreement!.map((d) => <li key={d}>{d}</li>)}</ul>
        </div>
      )}
      {endpoints.length ? (
        <table aria-label="Designers">
          <thead><tr><th>URL</th><th>Version</th><th>Status</th><th className="num">Parallel</th><th>Activity</th></tr></thead>
          <tbody>
            {endpoints.map((e) => (
              <tr key={e.url}>
                <td>{e.url}</td>
                <td>{e.design_version != null ? `design ${e.design_version}` : '—'}
                  {e.designer_build && <div className="muted">{e.designer_build}</div>}</td>
                <td><span className={e.reachable ? '' : 'problem'}>{e.reachable ? 'up' : 'down'}</span>
                  <span className="muted"> · {when(e.checked_at)}</span>
                  {!e.reachable && <div className="muted">{e.detail}</div>}</td>
                <td className="num">{e.parallel}</td>
                <td>{e.designing?.length ? e.designing.map((d) => (
                  <div key={d.title_id}><Link to={`/titles/${d.title_id}`}>{d.title || d.title_id}</Link>
                    <span className="muted"> since {when(d.started_at)}</span></div>
                )) : <span className="muted">{e.reachable ? 'idle' : '—'}</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className={designer.reachable === false ? 'problem' : 'muted'}>{designer.detail}</p>}
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
      {data.ended && <p className="muted">Ended: {data.ended}</p>}
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
      <div className="title-row">
        <h1>Status</h1>
        {status.data && <ScanSources scanned={!!status.data.index} />}
      </div>
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
            <CurrentJob job={status.data.current_job as Job | null} queued={status.data.queued}
                        designing={(status.data.designer?.endpoints ?? []).flatMap((e) => e.designing ?? [])} />
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
