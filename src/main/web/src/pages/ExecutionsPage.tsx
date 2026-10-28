// The log of executions (design/jobs-and-executions.md §5): the one running, each active job's next, and every one kept,
// newest first, by job and state.
import { Link, useSearchParams } from 'react-router'

import { useExecutions, useJobs, type Execution, type ExecutionState } from '../api/hooks'
import { describeExecution, StateBadge, stoppedBecause } from '../components/JobBits'
import { between, STATE_WORDS, when } from '../format'

const STATES = Object.keys(STATE_WORDS) as ExecutionState[]

/** What a run designed, declined and failed, in a few words; '' for a scan or an accept. */
function counts(execution: Execution): string {
  if (execution.kind !== 'run' || !execution.result) return ''
  const result = execution.result
  return [`${result.designed.length.toLocaleString()} designed`,
    result.failed.length ? `${result.failed.length.toLocaleString()} failed` : '',
    result.unavailable?.length ? `${result.unavailable.length.toLocaleString()} unavailable` : ''].filter(Boolean).join(', ')
}

export function ExecutionsPage() {
  const [params, setParams] = useSearchParams()
  const state = (STATES as string[]).includes(params.get('state') ?? '') ? (params.get('state') as ExecutionState) : null
  const job = params.get('job')
  const log = useExecutions(job, state)
  const jobs = useJobs()
  const choose = (name: string, value: string) => {
    const next = new URLSearchParams(params)
    if (value) next.set(name, value)
    else next.delete(name)
    setParams(next)
  }
  const executions = (log.data?.executions ?? []) as Execution[]
  return (
    <section>
      <div className="title-row">
        <h1>Executions</h1>
        <Link className="button" to="/jobs/new">New job…</Link>
      </div>
      <div className="filters">
        <label className="inline">
          Job{' '}
          <select value={job ?? ''} onChange={(e) => choose('job', e.target.value)}>
            <option value="">Any</option>
            {(jobs.data ?? []).map((j) => <option key={j.id} value={j.id}>{j.name}</option>)}
          </select>
        </label>
        <label className="inline">
          State{' '}
          <select value={state ?? ''} onChange={(e) => choose('state', e.target.value)}>
            <option value="">Any</option>
            {STATES.map((s) => <option key={s} value={s}>{STATE_WORDS[s]}</option>)}
          </select>
        </label>
      </div>
      {log.isError && <p role="alert" className="problem">{log.error.message}</p>}
      {(log.data?.next?.length ?? 0) > 0 && (
        <table className="list" aria-label="Next">
          <thead><tr><th>Next</th><th>Job</th></tr></thead>
          <tbody>
            {log.data!.next!.map((n) => (
              <tr key={n.job_id}>
                <td>{n.at ? when(n.at) : 'after the one running'}</td>
                <td><Link to={`/jobs/${n.job_id}`}>{n.job_name}</Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {log.data && (executions.length ? (
        <table className="list" aria-label="Executions">
          <thead><tr><th>State</th><th>Execution</th><th>From</th><th>Started</th><th>Took</th><th>Titles</th></tr></thead>
          <tbody>
            {executions.map((execution) => (
              <tr key={execution.id}>
                <td><StateBadge state={execution.state} />
                  {stoppedBecause(execution) && <span className="muted"> {stoppedBecause(execution)}</span>}
                  {execution.waiting_until && <span className="muted"> waits for {when(execution.waiting_until)}</span>}</td>
                <td><Link to={`/executions/${execution.id}`}>{describeExecution(execution)}</Link></td>
                <td>{execution.origin === 'schedule' ? 'recurring' : 'once'}</td>
                <td>{when(execution.started_at ?? execution.submitted_at)}</td>
                <td>{execution.started_at ? between(execution.started_at, execution.finished_at) : '—'}</td>
                <td>{counts(execution) || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="muted">No executions{state ? ` ${STATE_WORDS[state]!.toLowerCase()}` : ''}.</p>)}
    </section>
  )
}
