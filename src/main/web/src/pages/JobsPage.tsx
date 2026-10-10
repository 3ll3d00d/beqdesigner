// The jobs, newest first, by state.
import { Link, useSearchParams } from 'react-router'

import { useJobs, type Job, type JobState } from '../api/hooks'
import { describeJob, StateBadge } from '../components/JobBits'
import { between, STATE_WORDS, when } from '../format'

const STATES = Object.keys(STATE_WORDS) as JobState[]

export function JobsPage() {
  const [params, setParams] = useSearchParams()
  const state = (STATES as string[]).includes(params.get('state') ?? '') ? (params.get('state') as JobState) : null
  const jobs = useJobs(state)
  return (
    <section>
      <div className="title-row">
        <h1>Jobs</h1>
        <Link className="button" to="/jobs/new">New job…</Link>
      </div>
      <label className="inline">
        State{' '}
        <select value={state ?? ''} onChange={(e) => setParams(e.target.value ? { state: e.target.value } : {})}>
          <option value="">Any</option>
          {STATES.map((s) => <option key={s} value={s}>{STATE_WORDS[s]}</option>)}
        </select>
      </label>
      {jobs.isError && <p role="alert" className="problem">{jobs.error.message}</p>}
      {jobs.data && (jobs.data.jobs.length ? (
        <table className="list">
          <thead><tr><th>State</th><th>Job</th><th>From</th><th>Submitted</th><th>Took</th></tr></thead>
          <tbody>
            {(jobs.data.jobs as Job[]).map((job) => (
              <tr key={job.id}>
                <td><StateBadge state={job.state} /></td>
                <td><Link to={`/jobs/${job.id}`}>{describeJob(job)}</Link></td>
                <td>{job.origin === 'schedule' ? 'schedule' : 'asked'}</td>
                <td>{when(job.submitted_at)}</td>
                <td>{job.started_at ? between(job.started_at, job.finished_at) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="muted">No jobs{state ? ` ${STATE_WORDS[state]!.toLowerCase()}` : ''}.</p>)}
    </section>
  )
}
