// What the service is doing (design/web-app.md §4): the pipeline strip, the jobs (the one running now followed live, the
// schedule and the sources it scans) and the designers.
import { Link } from 'react-router'

import type { Schemas } from '../api/client'
import { describeFilter, filterToQuery } from '../api/filter'
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

type Schedule = Schemas['Schedule']

/** When the schedule runs next and what it does: each tick lists the sources again, then runs what the filter selects. */
function describeSchedule(schedule: Schedule): string {
  if (!schedule.enabled) return 'Paused.'
  return `Every ${schedule.interval_minutes} min: scan the sources, then run ${describeFilter(schedule.filter ?? {})} through ` +
    `${schedule.through}.`
}

/**
 * The work, from both ends: the job running now (followed live), what is waiting, the schedule that queues the next one
 * and the sources it lists first -- with every way to start or stop one.
 */
function Jobs({ job, queued, designing, sources, scanned }:
              { job: Job | null; queued: number; designing: Designs; sources: Schemas['SourceStatus'][]; scanned: boolean }) {
  const { events } = useJobEvents(job?.state === 'running' ? job.id : null)
  const progress = job?.state === 'running' ? job.progress : null
  const schedule = useSchedule()
  const save = useSaveSchedule()
  const trigger = useTriggerSchedule()
  const data = schedule.data
  const problem = save.error?.message ?? trigger.error?.message
  return (
    <section className="card wide" aria-labelledby="jobs">
      <h2 id="jobs">Jobs</h2>
      <dl className="facts" aria-label="Jobs">
        <dt>Job</dt>
        <dd>{job ? <><StateBadge state={job.state} /> <Link to={`/jobs/${job.id}`}>{describeJob(job)}</Link></>
          : <span className="muted">None running</span>}</dd>
        {job && <>
          <dt>Started</dt><dd>{when(job.started_at)}</dd>
          <dt>Estimated end</dt>
          <dd>{progress?.estimated_finish ? when(progress.estimated_finish) : '—'}
            {progress?.remaining_seconds != null && <span className="muted"> (about {duration(progress.remaining_seconds)} to go)</span>}</dd>
          <dt>Progress</dt>
          <dd>{progress ? <>{progress.done.toLocaleString()} of {progress.total.toLocaleString()}
            {progress.per_hour ? <span className="muted">, {progress.per_hour} an hour</span> : null}</> : '—'}</dd>
          <dt>Activity</dt><dd><Activity job={job} events={events} designing={job.state === 'running' ? designing : []} /></dd>
        </>}
        <dt>Waiting</dt><dd>{queued ? count(queued, 'job') : 'Nothing'}</dd>
        {data && <>
          <dt>Schedule</dt><dd>{describeSchedule(data)}</dd>
          {data.enabled && data.next_run_at && <><dt>Next run</dt><dd>{when(data.next_run_at)}</dd></>}
          {data.last_run && <><dt>Last run</dt><dd><StateBadge state={data.last_run.state} />{' '}
            <Link to={`/jobs/${data.last_run.job_id}`}>{when(data.last_run.finished_at)}</Link></dd></>}
          {data.last_skip && <><dt>Last skipped</dt><dd>{data.last_skip}</dd></>}
          {data.ended && <><dt>Ended</dt><dd>{data.ended}</dd></>}
        </>}
      </dl>
      {progress && <progress value={progress.done} max={Math.max(progress.total, 1)} aria-label="Progress" />}
      <div className="actions">
        {data && <>
          <button type="button" disabled={save.isPending}
                  onClick={() => save.mutate({ enabled: !data.enabled, interval_minutes: data.interval_minutes,
                                               filter: data.filter, through: data.through, retry_failed: data.retry_failed })}>
            {data.enabled ? 'Pause' : 'Resume'}
          </button>
          <button type="button" className="secondary" disabled={trigger.isPending} onClick={() => trigger.mutate()}>
            Run now
          </button>
        </>}
        <ScanSources scanned={scanned} />
        <Link to="/jobs/new">New job…</Link>
      </div>
      {problem && <p role="alert" className="problem">{problem}</p>}
      {sources.length > 0 && (
        <table aria-label="Sources">
          <thead><tr><th>Source</th><th>Kind</th><th className="num">Titles</th><th>Last listed</th><th>Problem</th></tr></thead>
          <tbody>
            {sources.map((s) => (
              <tr key={s.name}>
                <td>{s.name}</td><td>{s.kind}</td><td className="num">{s.item_count.toLocaleString()}</td>
                <td>{when(s.last_ok)}</td><td className={s.last_error ? 'problem' : ''}>{s.last_error || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

/** Every endpoint of the configured designer, one row each; a designer with none (manual) is one row saying why. */
function Designers({ designer }: { designer: Schemas['DesignerStatus'] | null | undefined }) {
  if (!designer) return null
  const endpoints = designer.endpoints ?? []
  return (
    <section className="card wide" aria-labelledby="designers">
      <h2 id="designers">Designers</h2>
      {(designer.design_disagreement?.length ?? 0) > 0 && (
        <div className="problem" role="alert" aria-label="Designers disagree">
          <p>The designers that are up would not design alike, so a redesign answers differently by endpoint:</p>
          <ul>{designer.design_disagreement!.map((d) => <li key={d}>{d}</li>)}</ul>
        </div>
      )}
      <table aria-label="Designers">
        <thead><tr><th>Designer</th><th>URL</th><th>Status</th><th className="num">Protocol</th><th>Version</th>
          <th className="num">Max jobs</th><th>Activity</th></tr></thead>
        <tbody>
          {endpoints.map((e) => (
            <tr key={e.url}>
              <td>{designer.name}</td>
              <td>{e.url}</td>
              <td><span className={e.reachable ? '' : 'problem'}>{e.reachable ? 'up' : 'down'}</span>
                <span className="muted"> · {when(e.checked_at)}</span>
                {!e.reachable && <div className="muted">{e.detail}</div>}</td>
              <td className="num">{e.design_version ?? '—'}</td>
              <td title={e.designer_build ?? undefined}>{e.designer_release ?? '—'}</td>
              <td className="num">{e.parallel}</td>
              <td>{e.designing?.length ? e.designing.map((d) => (
                <div key={d.title_id}><Link to={`/titles/${d.title_id}`}>{d.title || d.title_id}</Link>
                  <span className="muted"> since {when(d.started_at)}</span></div>
              )) : <span className="muted">{e.reachable ? 'idle' : '—'}</span>}</td>
            </tr>
          ))}
          {!endpoints.length && (
            <tr>
              <td>{designer.name || '—'}</td>
              <td colSpan={6} className={designer.reachable === false ? 'problem' : 'muted'}>{designer.detail}</td>
            </tr>
          )}
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
            <Jobs job={status.data.current_job as Job | null} queued={status.data.queued}
                  designing={(status.data.designer?.endpoints ?? []).flatMap((e) => e.designing ?? [])}
                  sources={status.data.index?.sources ?? []} scanned={!!status.data.index} />
            <Designers designer={status.data.designer} />
          </div>
          {status.data.tmdb === false && <p className="muted">No TMDB key is set: titles are designed with the library's own metadata.</p>}
        </>
      )}
    </section>
  )
}
