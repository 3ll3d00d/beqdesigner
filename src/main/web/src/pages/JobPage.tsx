// One job (design/jobs-and-executions.md §5): what it does and when, its execution running now (followed live), the ones
// before, and pause, resume, run now, edit and remove.
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import type { Schemas } from '../api/client'
import { cleanFilter, type Needs, type TitleFilter } from '../api/filter'
import { useCancelExecution, useDefaultWindow, useEditJob, useExecutionEvents, useExecutions, useJob, useJobAction,
  useStatus, type Execution, type Job } from '../api/hooks'
import { EveryField, type Every } from '../components/EveryField'
import { OUTCOME_WORDS } from '../components/ExecutionStatistics'
import { FilterForm } from '../components/FilterForm'
import { describeEvery, describeExecution, describeRequest, ExecutionProgress, JobStateBadge, StateBadge, stoppedBecause }
  from '../components/JobBits'
import { describeWindow, WindowField, type Window } from '../components/WindowField'
import { between, count, when } from '../format'

const RECURRING_NEEDS: readonly Needs[] = ['extract', 'design']

/** Its name and window; a recurring job's interval and run too. A one-off's request is fixed once made. */
function EditJob({ job, done }: { job: Job; done: () => void }) {
  const edit = useEditJob()
  const status = useStatus(0)
  const sources = status.data?.index?.sources.map((s) => s.name) ?? []
  const [name, setName] = useState(job.name)
  const [window, setWindow] = useState<Window>(job.window ?? null)
  const [every, setEvery] = useState<Every>(job.every ?? null)
  const run = job.kind === 'run' ? job.request : null
  const [filter, setFilter] = useState<TitleFilter>(run?.filter ?? {})
  const [through, setThrough] = useState<'extract' | 'design'>(run?.through === 'extract' ? 'extract' : 'design')
  const [retryFailed, setRetryFailed] = useState(!!run?.retry_failed)
  return (
    <form aria-label="Edit job" onSubmit={(e) => {
      e.preventDefault()
      const body: Schemas['JobUpdate'] & { id: string } = { id: job.id, name, window }
      if (job.every && run) {
        Object.assign(body, { every: every ?? job.every, request: {
          ...run, filter: cleanFilter({ ...filter, needs: ['extract', 'design'] }), through, retry_failed: retryFailed } })
      }
      edit.mutate(body, { onSuccess: done })
    }}>
      <label className="inline">Name <input value={name} onChange={(e) => setName(e.target.value)} /></label>
      <WindowField value={window} onChange={setWindow} />
      {job.every && run && <>
        <EveryField value={every} onChange={setEvery} allowed={true} />
        <FilterForm value={filter} onChange={setFilter} sources={sources} needs={RECURRING_NEEDS} />
        <fieldset>
          <legend>Through</legend>
          {(['extract', 'design'] as const).map((stage) => (
            <label key={stage} className="check">
              <input type="radio" name="through" checked={through === stage} onChange={() => setThrough(stage)} />{stage}
            </label>
          ))}
          <label className="check">
            <input type="checkbox" checked={retryFailed} onChange={(e) => setRetryFailed(e.target.checked)} />
            Retry titles that failed before
          </label>
        </fieldset>
      </>}
      <div className="actions">
        <button type="submit" disabled={edit.isPending}>Save</button>
        <button type="button" className="secondary" onClick={done}>Cancel</button>
      </div>
      {edit.error && <p role="alert" className="problem">{edit.error.message}</p>}
    </form>
  )
}

/** The execution running now (or waiting), followed live, with Cancel. */
function Running({ execution, oneOff }: { execution: Execution; oneOff: boolean }) {
  const live = useExecutionEvents(execution.state === 'running' ? execution.id : null)
  const cancel = useCancelExecution()
  return (
    <section aria-label="Running">
      <h2>Now</h2>
      <p><StateBadge state={execution.state} />{' '}
        <Link to={`/executions/${execution.id}`}>{describeExecution(execution)}</Link>
        {execution.waiting_until && <span className="muted"> — waits for {when(execution.waiting_until)}</span>}
        {' '}<button type="button" className="danger" disabled={cancel.isPending} onClick={() => cancel.mutate(execution.id)}>
          Cancel</button></p>
      {oneOff && <p className="muted">Cancelling it removes this job.</p>}
      {cancel.error && <p role="alert" className="problem">{cancel.error.message}</p>}
      <ExecutionProgress execution={execution} events={live.events} />
    </section>
  )
}

export function JobPage() {
  const { jobId = '' } = useParams()
  const job = useJob(jobId)
  const executions = useExecutions(jobId)
  const defaults = useDefaultWindow()
  const act = useJobAction()
  const navigate = useNavigate()
  const [editing, setEditing] = useState(false)
  const [removing, setRemoving] = useState(false)

  if (job.isPending) return <p className="muted">Loading…</p>
  if (job.isError) return <p role="alert" className="problem">{job.error.message}</p>
  const data = job.data
  const live = data.state === 'active' || data.state === 'paused'
  const running = data.running_execution as Execution | null | undefined
  const history = ((executions.data?.executions ?? []) as Execution[]).filter((e) => e.id !== running?.id)
  return (
    <section>
      <p><Link to="/">← Status</Link></p>
      <div className="title-row">
        <h1>{data.name}</h1>
        {live && (
          <div className="actions">
            {data.state === 'active'
              ? <button type="button" disabled={act.isPending} onClick={() => act.mutate({ id: data.id, action: 'pause' })}>Pause</button>
              : <button type="button" disabled={act.isPending} onClick={() => act.mutate({ id: data.id, action: 'resume' })}>Resume</button>}
            {data.every && !running && (
              <button type="button" className="secondary" disabled={act.isPending}
                      onClick={() => act.mutate({ id: data.id, action: 'run' })}>Run now</button>
            )}
            <button type="button" className="secondary" onClick={() => setEditing(true)}>Edit…</button>
            <button type="button" className="danger" onClick={() => setRemoving(true)}>Remove…</button>
          </div>
        )}
      </div>
      {act.error && <p role="alert" className="problem">{act.error.message}</p>}
      {removing && (
        <div className="confirm" role="group" aria-label="Confirm remove">
          <p>Remove this job? Nothing more of it runs{running ? ', and the run going stops before its next title' : ''}.</p>
          <div className="actions">
            <button type="button" className="danger" disabled={act.isPending}
                    onClick={() => act.mutate({ id: data.id, action: 'remove' },
                                              { onSuccess: () => { setRemoving(false); navigate('/') } })}>Remove</button>
            <button type="button" className="secondary" onClick={() => setRemoving(false)}>Back</button>
          </div>
        </div>
      )}
      {editing ? <EditJob job={data} done={() => setEditing(false)} /> : (
        <dl className="facts">
          <dt>State</dt><dd><JobStateBadge state={data.state} />{data.ended && <span className="muted"> — {data.ended}</span>}</dd>
          <dt>Does</dt><dd>{describeRequest(data)}</dd>
          <dt>When</dt><dd>{describeEvery(data)}</dd>
          <dt>Runs</dt><dd>{describeWindow(data.window, defaults.data)}</dd>
          <dt>Next</dt><dd>{data.next_at ? when(data.next_at) : '—'}</dd>
          {data.progress && <><dt>Progress</dt><dd>{data.progress.done.toLocaleString()}
            {data.progress.total != null && <> of {data.progress.total.toLocaleString()}</>} done
            {data.remaining != null && data.state !== 'done' && <span className="muted"> ({count(data.remaining, 'title')} left)</span>}</dd></>}
          {data.last_skip && <><dt>Last skipped</dt><dd>{data.last_skip}</dd></>}
          {data.outcomes && Object.keys(data.outcomes).length > 0 && <><dt>Titles</dt>
            <dd>{Object.entries(data.outcomes).map(([o, n]) => `${(OUTCOME_WORDS[o] ?? o).toLowerCase()} ${n.toLocaleString()}`)
              .join(', ')} <span className="muted">(over the executions kept)</span></dd></>}
          <dt>Made</dt><dd>{when(data.created_at)}</dd>
        </dl>
      )}
      {running && <Running execution={running} oneOff={!data.every} />}
      <h2>Executions</h2>
      {history.length ? (
        <table className="list" aria-label="Its executions">
          <thead><tr><th>State</th><th>Run</th><th>Started</th><th>Took</th></tr></thead>
          <tbody>
            {history.map((e) => (
              <tr key={e.id}>
                <td><StateBadge state={e.state} />{stoppedBecause(e) && <span className="muted"> {stoppedBecause(e)}</span>}</td>
                <td><Link to={`/executions/${e.id}`}>{e.number ? `Run ${e.number}` : describeExecution(e)}</Link></td>
                <td>{when(e.started_at ?? e.submitted_at)}</td>
                <td>{e.started_at ? between(e.started_at, e.finished_at) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="muted">{running ? 'None before this one.' : 'None yet.'}</p>}
    </section>
  )
}
