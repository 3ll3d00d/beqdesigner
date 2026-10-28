// What the service is doing (design/web-app.md §4, design/jobs-and-executions.md §5): the pipeline strip, the jobs (each
// one, with the execution running now followed live, and the sources a scan lists) and the designers.
import { Link } from 'react-router'

import type { Schemas } from '../api/client'
import { filterToQuery } from '../api/filter'
import { useExecutionEvents, useJobAction, useJobs, useStatus, type Execution, type Job } from '../api/hooks'
import type { JobEvent } from '../api/events'
import { describeEvery, describeExecution, describeRequest, extraction, JobStateBadge, StateBadge } from '../components/JobBits'
import { ScanSources } from '../components/ScanSources'
import { DefaultWindow, describeRanges } from '../components/WindowField'
import { briefly, count, duration, NEEDS_ORDER, STRIP_WORDS, when } from '../format'

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
function Activity({ job, events, designing }: { job: Execution; events: JobEvent[]; designing: Designs }) {
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

/** A job in a row: what it does, when, its state and next run, the execution running now, and pause, resume, run now. */
function JobRow({ job }: { job: Job }) {
  const act = useJobAction()
  const running = job.running_execution as Execution | null | undefined
  return (
    <tr>
      <td><Link to={`/jobs/${job.id}`}>{job.name}</Link><div className="muted">{describeRequest(job)}</div></td>
      <td>{describeEvery(job)}{job.window !== null && job.window !== undefined &&
        <div className="muted">{describeRanges(job.window)}</div>}</td>
      <td><JobStateBadge state={job.state} />{job.last_skip && <div className="muted">skipped: {job.last_skip}</div>}</td>
      <td>{running ? <><StateBadge state={running.state} />{' '}
        <Link to={`/executions/${running.id}`}>{running.progress && running.state === 'running'
          ? `${running.progress.done.toLocaleString()} of ${running.progress.total.toLocaleString()}` : 'open'}</Link></>
        : job.next_at ? when(job.next_at) : '—'}</td>
      <td className="actions">
        {job.state === 'active'
          ? <button type="button" className="secondary" disabled={act.isPending}
                    onClick={() => act.mutate({ id: job.id, action: 'pause' })}>Pause</button>
          : job.state === 'paused' && <button type="button" className="secondary" disabled={act.isPending}
                                              onClick={() => act.mutate({ id: job.id, action: 'resume' })}>Resume</button>}
        {job.every && !running && (job.state === 'active' || job.state === 'paused') &&
          <button type="button" className="secondary" disabled={act.isPending}
                  onClick={() => act.mutate({ id: job.id, action: 'run' })}>Run now</button>}
        <Link to={`/jobs/${job.id}`}>Edit…</Link>
        {act.error && <span role="alert" className="problem"> {act.error.message}</span>}
      </td>
    </tr>
  )
}

/**
 * The work, from both ends: the execution running now (followed live), what waits, and every job that is active or paused
 * -- with the ways to start, pause or change one -- and the sources a scan lists.
 */
function Jobs({ running, waiting, designing, sources, scanned }:
              { running: Execution | null; waiting: number; designing: Designs; sources: Schemas['SourceStatus'][];
                scanned: boolean }) {
  const { events } = useExecutionEvents(running?.state === 'running' ? running.id : null)
  const progress = running?.state === 'running' ? running.progress : null
  const jobs = useJobs()
  const listed = (jobs.data ?? []).filter((j) => j.state === 'active' || j.state === 'paused')
  return (
    <section className="card wide" aria-labelledby="jobs">
      <h2 id="jobs">Jobs</h2>
      <dl className="facts" aria-label="Running">
        <dt>Running</dt>
        <dd>{running ? <><StateBadge state={running.state} /> <Link to={`/executions/${running.id}`}>{describeExecution(running)}</Link></>
          : <span className="muted">Nothing</span>}</dd>
        {running && <>
          <dt>Started</dt><dd>{when(running.started_at)}</dd>
          <dt>Estimated end</dt>
          <dd>{progress?.estimated_finish ? when(progress.estimated_finish) : '—'}
            {progress?.remaining_seconds != null && <span className="muted"> (about {duration(progress.remaining_seconds)} to go)</span>}</dd>
          <dt>Progress</dt>
          <dd>{progress ? <>{progress.done.toLocaleString()} of {progress.total.toLocaleString()}
            {progress.per_hour ? <span className="muted">, {progress.per_hour} an hour</span> : null}</> : '—'}</dd>
          <dt>Activity</dt><dd><Activity job={running} events={events} designing={running.state === 'running' ? designing : []} /></dd>
        </>}
        <dt>Waiting</dt><dd>{waiting ? count(waiting, 'execution') : 'Nothing'}</dd>
        <dt>Jobs run</dt><dd><DefaultWindow /></dd>
      </dl>
      {progress && <progress value={progress.done} max={Math.max(progress.total, 1)} aria-label="Progress" />}
      {jobs.isError && <p role="alert" className="problem">{jobs.error.message}</p>}
      {listed.length > 0 ? (
        <table className="list" aria-label="Jobs">
          <thead><tr><th>Job</th><th>When</th><th>State</th><th>Next</th><th></th></tr></thead>
          <tbody>{listed.map((job) => <JobRow key={job.id} job={job} />)}</tbody>
        </table>
      ) : jobs.data && <p className="muted">No jobs are active or paused.</p>}
      <div className="actions">
        <ScanSources scanned={scanned} />
        <Link to="/jobs/new">New job…</Link>
        <Link to="/executions">Every execution</Link>
      </div>
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

/** Why an endpoint is down in a few words; the service's whole message when opened (it can be a paragraph). */
function Why({ detail }: { detail: string }) {
  const short = briefly(detail)
  if (short === detail) return <div className="muted">{detail}</div>
  return (
    <details className="why">
      <summary className="muted">{short}</summary>
      <p className="muted">{detail}</p>
    </details>
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
                {!e.reachable && <Why detail={e.detail} />}</td>
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
            <Jobs running={(status.data.running_execution ?? null) as Execution | null} waiting={status.data.waiting ?? 0}
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
