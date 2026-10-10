// Starting work (design/web-review.md §4): a scan, or a run of a filtered selection through a stage, previewed first with
// POST /v1/plan. Publish and commit are offered only when the service allows them (allow_repository_writes, a repository).
import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router'

import type { Schemas } from '../api/client'
import { cleanFilter, describeFilter, filterFromQuery, type Needs, type TitleFilter } from '../api/filter'
import { usePlan, useStatus, useSubmitRun, useSubmitScan } from '../api/hooks'
import { FilterForm } from '../components/FilterForm'
import { count } from '../format'

type Through = Schemas['Through']
const RUN_NEEDS: readonly Needs[] = ['attention', 'extract', 'design', 'review', 'publish', 'commit']

export function repositoryRefusal(status: Schemas['ServiceStatus'] | undefined): string {
  if (!status) return 'Checking what the service allows…'
  if (!status.repository_writes) return 'The service does not allow publish or commit (allow_repository_writes in service.yaml).'
  if (!status.repositories_configured) return 'The profile names no filter repository to publish to.'
  return ''
}

function ScanForm({ sources }: { sources: string[] }) {
  const navigate = useNavigate()
  const scan = useSubmitScan()
  const [chosen, setChosen] = useState<string[]>([])
  return (
    <form onSubmit={(e) => {
      e.preventDefault()
      scan.mutate({ sources: chosen, allow_empty: false }, { onSuccess: (job) => navigate(`/jobs/${job.id}`) })
    }}>
      <fieldset>
        <legend>Sources</legend>
        {sources.length ? sources.map((name) => (
          <label key={name} className="check">
            <input type="checkbox" checked={chosen.includes(name)}
                   onChange={(e) => setChosen(e.target.checked ? [...chosen, name] : chosen.filter((n) => n !== name))} />
            {name}
          </label>
        )) : null}
        <p className="muted">{chosen.length ? `List ${chosen.join(', ')} again.` : 'Every source is listed again.'}</p>
      </fieldset>
      {scan.error && <p role="alert" className="problem">{scan.error.message}</p>}
      <button type="submit" disabled={scan.isPending}>Scan</button>
    </form>
  )
}

function RunForm({ sources, initial, status }: { sources: string[]; initial: TitleFilter; status?: Schemas['ServiceStatus'] }) {
  const navigate = useNavigate()
  const plan = usePlan()
  const run = useSubmitRun()
  const [filter, setFilter] = useState<TitleFilter>(initial)
  const [through, setThrough] = useState<Through>('design')
  const [scanFirst, setScanFirst] = useState(true)
  const [retryFailed, setRetryFailed] = useState(false)
  const refusal = repositoryRefusal(status)
  const body = (): Schemas['RunJobRequest'] => ({ filter: cleanFilter(filter), through, scan_first: scanFirst,
                                                  retry_failed: retryFailed })
  const preview = plan.data
  const stale = plan.variables && JSON.stringify(plan.variables) !== JSON.stringify(body())

  return (
    <form onSubmit={(e) => {
      e.preventDefault()
      run.mutate(body(), { onSuccess: (job) => navigate(`/jobs/${job.id}`) })
    }}>
      <FilterForm value={filter} onChange={setFilter} sources={sources} needs={RUN_NEEDS} />
      <fieldset>
        <legend>Through</legend>
        {(['extract', 'design', 'publish', 'commit'] as Through[]).map((stage) => {
          const refused = (stage === 'publish' || stage === 'commit') && !!refusal
          return (
            <label key={stage} className="check" title={refused ? refusal : undefined}>
              <input type="radio" name="through" value={stage} checked={through === stage} disabled={refused}
                     onChange={() => setThrough(stage)} />
              {stage}
            </label>
          )
        })}
        {refusal && <p className="muted">Publish and commit: {refusal}</p>}
        <p className="muted">Each title is taken only as far as it needs; nothing goes past design until a person accepts it.</p>
        <label className="check">
          <input type="checkbox" checked={scanFirst} onChange={(e) => setScanFirst(e.target.checked)} />
          List the sources first (so new titles are seen)
        </label>
        <label className="check">
          <input type="checkbox" checked={retryFailed} onChange={(e) => setRetryFailed(e.target.checked)} />
          Retry titles that failed before
        </label>
      </fieldset>
      <div className="actions">
        <button type="button" className="secondary" disabled={plan.isPending} onClick={() => plan.mutate(body())}>
          Preview
        </button>
        <button type="submit" disabled={run.isPending}>Run</button>
      </div>
      {(plan.error || run.error) && <p role="alert" className="problem">{(plan.error ?? run.error)!.message}</p>}
      {preview && (
        <section className={stale ? 'preview stale' : 'preview'} aria-label="Preview">
          <h2>{preview.label}</h2>
          {stale && <p className="muted">The choices changed since this preview: preview again.</p>}
          <p>{count(preview.planned.length, 'title')} would run{preview.skipped.length
            ? `, ${count(preview.skipped.length, 'title')} skipped` : ''} ({describeFilter(cleanFilter(filter))}).
            {scanFirst && ' The scan first may add new titles.'}</p>
          {preview.planned.length > 0 && (
            <details>
              <summary>Would run</summary>
              <ul className="titles">
                {preview.planned.slice(0, 200).map((p) => <li key={p.id}>{p.title} <span className="muted">— {p.stages.join(', ')}</span></li>)}
                {preview.planned.length > 200 && <li className="muted">…and {preview.planned.length - 200} more</li>}
              </ul>
            </details>
          )}
          {preview.skipped.length > 0 && (
            <details>
              <summary>Skipped</summary>
              <ul className="titles">
                {preview.skipped.slice(0, 200).map((s) => <li key={s.id}>{s.title} <span className="muted">— {s.reason}</span></li>)}
              </ul>
            </details>
          )}
        </section>
      )}
    </form>
  )
}

export function NewJobPage() {
  const [params, setParams] = useSearchParams()
  const status = useStatus(0)
  const kind = params.get('kind') === 'scan' ? 'scan' : 'run'
  const sources = status.data?.index?.sources.map((s) => s.name) ?? []
  return (
    <section>
      <p><Link to="/jobs">← Jobs</Link></p>
      <h1>New job</h1>
      <div className="tabs" role="tablist">
        {(['run', 'scan'] as const).map((k) => (
          <button key={k} type="button" role="tab" aria-selected={kind === k} className={kind === k ? 'tab on' : 'tab'}
                  onClick={() => setParams(k === 'scan' ? { kind: 'scan' } : {})}>
            {k === 'run' ? 'Extract & design' : 'Scan'}
          </button>
        ))}
      </div>
      {kind === 'scan' ? <ScanForm sources={sources} />
        : <RunForm sources={sources} initial={filterFromQuery(params)} status={status.data} />}
    </section>
  )
}
