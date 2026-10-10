// Titles (design/web-app.md §4): the work list's table over the filter in the address bar, "review the next waiting
// title", and publishing/committing the accepted titles in it, offered only when the service allows them.
import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router'

import { cleanFilter, describeFilter, filterFromQuery, filterToQuery, type TitleFilter } from '../api/filter'
import { PAGE_SIZE, useNextTitle, usePlan, useStatus, useSubmitRun, useTitles } from '../api/hooks'
import { FilterForm } from '../components/FilterForm'
import { count, NEEDS_WORDS, type NeedsWord } from '../format'
import { repositoryRefusal } from './NewJobPage'

/** The filter and paging the page is showing, read from and written to the address bar. */
export function useListParams() {
  const [params, setParams] = useSearchParams()
  const filter = filterFromQuery(params)
  const includeDone = params.get('include_done') === 'true'
  const offset = Math.max(0, Number(params.get('offset')) || 0)
  const update = (next: TitleFilter, done = includeDone, at = 0) => {
    const query = filterToQuery(next)
    if (done) query.set('include_done', 'true')
    if (at) query.set('offset', String(at))
    setParams(query, { replace: true })
  }
  return { filter, includeDone, offset, update, query: filterToQuery(filter).toString() }
}

function Repositories({ filter }: { filter: TitleFilter }) {
  const status = useStatus()
  const plan = usePlan()
  const run = useSubmitRun()
  const navigate = useNavigate()
  const [asking, setAsking] = useState<'publish' | 'commit' | null>(null)
  const refusal = repositoryRefusal(status.data)
  const request = (through: 'publish' | 'commit') => ({
    filter: cleanFilter({ ...filter, needs: [through] }), through, scan_first: false, retry_failed: false,
  })

  function ask(through: 'publish' | 'commit') {
    setAsking(through)
    plan.mutate(request(through))
  }

  const planned = plan.data?.planned.length ?? 0
  return (
    <div className="repositories">
      <div className="actions">
        <button type="button" className="secondary" disabled={!!refusal || plan.isPending} title={refusal || undefined}
                onClick={() => ask('publish')}>Publish accepted…</button>
        <button type="button" className="secondary" disabled={!!refusal || plan.isPending} title={refusal || undefined}
                onClick={() => ask('commit')}>Commit published…</button>
      </div>
      {refusal && <p className="muted">{refusal}</p>}
      {asking && plan.data && (
        <div className="confirm" role="group" aria-label="Confirm">
          {planned ? (
            <>
              <p>{asking === 'publish'
                ? `Publish ${count(planned, 'accepted title')} to the catalogue's filter repository?`
                : `Commit ${count(planned, 'published title')} to the catalogue repositories (and push, if the profile says so)?`}
              </p>
              <div className="actions">
                <button type="button" disabled={run.isPending} onClick={() => run.mutate(request(asking), {
                  onSuccess: (job) => navigate(`/jobs/${job.id}`) })}>
                  {asking === 'publish' ? 'Publish' : 'Commit'}
                </button>
                <button type="button" className="secondary" onClick={() => setAsking(null)}>Back</button>
              </div>
            </>
          ) : <p className="muted">Nothing in this list is waiting to be {asking === 'publish' ? 'published' : 'committed'}.</p>}
        </div>
      )}
      {(plan.error || run.error) && <p role="alert" className="problem">{(plan.error ?? run.error)!.message}</p>}
    </div>
  )
}

export function TitlesPage() {
  const { filter, includeDone, offset, update, query } = useListParams()
  const status = useStatus()
  const titles = useTitles(filter, includeDone, offset)
  const next = useNextTitle()
  const navigate = useNavigate()
  const [nothing, setNothing] = useState('')
  const sources = status.data?.index?.sources.map((s) => s.name) ?? []
  const link = (id: string) => `/titles/${encodeURIComponent(id)}${query ? `?${query}` : ''}`

  async function reviewNext() {
    setNothing('')
    try {
      const found = await next(filter)
      if (found) navigate(link(found.id))
      else setNothing('No title in this list is waiting for a decision.')
    } catch (error) {
      setNothing(error instanceof Error ? error.message : String(error))
    }
  }

  const page = titles.data
  return (
    <section>
      <div className="title-row">
        <h1>Titles</h1>
        <button type="button" onClick={reviewNext}>Review next waiting</button>
      </div>
      {nothing && <p role="status" className="muted">{nothing}</p>}
      <FilterForm value={filter} onChange={(f) => update(f)} sources={sources} />
      <label className="check">
        <input type="checkbox" checked={includeDone} onChange={(e) => update(filter, e.target.checked)} />
        Include titles that need nothing
      </label>
      <Repositories filter={filter} />
      {titles.isError && <p role="alert" className="problem">{titles.error.message}</p>}
      {page && (
        <>
          <p className="muted">{count(page.total, 'title')} ({describeFilter(filter)}).</p>
          <div className="table-wrap">
            <table className="list">
              <thead>
                <tr><th>Title</th><th>Year</th><th>Needs</th><th>Detail</th><th className="num">Confidence</th>
                  <th className="num">Designs</th><th>Flags</th></tr>
              </thead>
              <tbody>
                {page.titles.map((t) => (
                  <tr key={t.id} className={`tier-${t.tier}`}>
                    <td><Link to={link(t.id)}>{t.title || t.display_name || t.id}</Link>{t.is_new && <span className="badge">new</span>}</td>
                    <td>{t.year}</td>
                    <td>{NEEDS_WORDS[t.needs as NeedsWord] ?? t.needs}</td>
                    <td className="detail">{t.detail}</td>
                    <td className="num">{t.confidence === null ? '—' : t.confidence.toFixed(2)}</td>
                    <td className="num">{t.candidate_count || '—'}</td>
                    <td>{t.flags.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {page.total > PAGE_SIZE && (
            <div className="actions pager">
              <button type="button" className="secondary" disabled={offset === 0}
                      onClick={() => update(filter, includeDone, Math.max(0, offset - PAGE_SIZE))}>Previous</button>
              <span className="muted">{offset + 1}–{Math.min(offset + PAGE_SIZE, page.total)} of {page.total.toLocaleString()}</span>
              <button type="button" className="secondary" disabled={offset + PAGE_SIZE >= page.total}
                      onClick={() => update(filter, includeDone, offset + PAGE_SIZE)}>Next</button>
            </div>
          )}
        </>
      )}
    </section>
  )
}
