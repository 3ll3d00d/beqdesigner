// Titles (design/web-app.md §4): the work list's table over the filter in the address bar, "review the next waiting
// title", and publishing/committing the accepted titles in it, offered only when the service allows them. Each column
// sorts from its heading and filters from the row under it; both run on the service, so they hold across pages.
import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router'

import {
  cleanFilter, describeFilter, filterFromQuery, filterToQuery, FLAGS, sortFromQuery, sortToQuery, type Flag, type Kind,
  type Sort, type TitleFilter, type TitleSort,
} from '../api/filter'
import type { Schemas } from '../api/client'
import { MAX_REVISED, PAGE_SIZE, useMatchingIds, useNextTitle, usePlan, useStatus, useSubmitRun, useTitles } from '../api/hooks'
import { ImpactPanel } from '../components/ImpactPanel'
import { ReviseDialog, reviseSummary } from '../components/ReviseDialog'
import { ScanSources } from '../components/ScanSources'
import { count, NEEDS_ORDER, NEEDS_WORDS, type NeedsWord } from '../format'
import { repositoryRefusal } from './NewJobPage'

/** A new filter, or how to change the one in the address bar now. */
type FilterChange = TitleFilter | ((current: TitleFilter) => TitleFilter)

/**
 * The filter, sort and paging the page is showing, read from and written to the address bar. A change given as a function
 * is applied to the address as last written, not as last rendered: two changes before the page draws again (a typed
 * filter's pause ending as a list is picked) would otherwise each start from the same filter, and the second undo the first.
 */
export function useListParams() {
  const [params, setParams] = useSearchParams()
  const written = useRef(params)
  useEffect(() => { written.current = params }, [params])
  const filter = filterFromQuery(params)
  const order = sortFromQuery(params)
  const includeDone = params.get('include_done') === 'true'
  const offset = Math.max(0, Number(params.get('offset')) || 0)
  const update = (next: FilterChange, done = includeDone, at = 0, by: Sort = order) => {
    const chosen = typeof next === 'function' ? next(filterFromQuery(written.current)) : next
    const query = sortToQuery(by, filterToQuery(chosen))
    if (done) query.set('include_done', 'true')
    if (at) query.set('offset', String(at))
    written.current = query
    setParams(query, { replace: true })
  }
  return { filter, order, includeDone, offset, update, query: sortToQuery(order, filterToQuery(filter)).toString() }
}

const TYPING_MS = 300

/** A text filter that follows the address bar but writes to it only once typing pauses (and untrimmed while typing). */
function FilterText({ label, value, placeholder, onChange }: {
  label: string; value: string; placeholder?: string; onChange: (value: string | null) => void
}) {
  const [text, setText] = useState(value)
  const latest = useRef(onChange)
  useEffect(() => { latest.current = onChange })
  const [seen, setSeen] = useState(value)
  if (seen !== value) {   // the address bar changed under it (Back, a link): show that, unless it is what was typed
    setSeen(value)
    if (text.trim() !== value) setText(value)
  }
  useEffect(() => {
    if (text.trim() === value) return
    const timer = setTimeout(() => latest.current(text.trim() || null), TYPING_MS)
    return () => clearTimeout(timer)
  }, [text, value])
  return <input type="text" aria-label={label} value={text} placeholder={placeholder} onChange={(e) => setText(e.target.value)} />
}

/** Which needs to show, as a drop-down of checkboxes in the Needs column. */
function NeedsPicker({ value, onChange }: { value: TitleFilter['needs']; onChange: (needs: TitleFilter['needs']) => void }) {
  const chosen = new Set(value ?? [])
  const toggle = (need: NeedsWord) => {
    const next = new Set(chosen)
    if (next.has(need)) next.delete(need)
    else next.add(need)
    onChange(NEEDS_ORDER.filter((n) => next.has(n)))
  }
  return (
    <details className="pick">
      <summary aria-label="Filter needs">{chosen.size ? [...chosen].map((n) => NEEDS_WORDS[n as NeedsWord]).join(', ') : 'Any'}</summary>
      <div className="pick-menu" role="group" aria-label="Needs">
        {NEEDS_ORDER.map((need) => (
          <label key={need} className="check">
            <input type="checkbox" checked={chosen.has(need)} onChange={() => toggle(need)} />
            {NEEDS_WORDS[need]}
          </label>
        ))}
      </div>
    </details>
  )
}

const COLUMNS: { sort: TitleSort; label: string; num?: boolean }[] = [
  { sort: 'title', label: 'Title' }, { sort: 'year', label: 'Year' }, { sort: 'kind', label: 'Kind' },
  { sort: 'source', label: 'Source' }, { sort: 'needs', label: 'Needs' }, { sort: 'detail', label: 'Detail' },
  { sort: 'confidence', label: 'Confidence', num: true }, { sort: 'candidate_count', label: 'Designs', num: true },
  { sort: 'flags', label: 'Flags' },
]

/** A heading that sorts by its column: first click ascending, then descending, then back to work-list order. */
function SortHeading({ column, order, onSort }: {
  column: (typeof COLUMNS)[number]; order: Sort; onSort: (order: Sort) => void
}) {
  const on = order.sort === column.sort
  const next: Sort = !on ? { sort: column.sort, descending: false }
    : !order.descending ? { sort: column.sort, descending: true } : { sort: null, descending: false }
  return (
    <th className={column.num ? 'num' : undefined} aria-sort={on ? (order.descending ? 'descending' : 'ascending') : 'none'}>
      <button type="button" className="sort" onClick={() => onSort(next)}>
        {column.label}<span aria-hidden="true">{on ? (order.descending ? ' ▼' : ' ▲') : ''}</span>
      </button>
    </th>
  )
}

/** The row of filters under the headings, one per column that can be filtered. */
function FilterRow({ filter, sources, onChange }: {
  filter: TitleFilter; sources: string[]; onChange: (change: (current: TitleFilter) => TitleFilter) => void
}) {
  const set = (patch: Partial<TitleFilter>) => onChange((current) => ({ ...current, ...patch }))
  return (
    <tr className="filters">
      <td />
      <td><FilterText label="Filter title" value={filter.match ?? ''} placeholder="title, name, id or path"
                      onChange={(match) => set({ match })} /></td>
      <td><FilterText label="Filter year" value={filter.year ?? ''} placeholder=">=2020" onChange={(year) => set({ year })} /></td>
      <td>
        <select aria-label="Filter kind" value={filter.kind ?? ''}
                onChange={(e) => set({ kind: (e.target.value || null) as Kind | null })}>
          <option value="">Any</option>
          <option value="movie">Movies</option>
          <option value="tv">TV</option>
        </select>
      </td>
      <td>
        <select aria-label="Filter source" value={filter.source ?? ''} onChange={(e) => set({ source: e.target.value || null })}>
          <option value="">Any</option>
          {sources.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
      </td>
      <td><NeedsPicker value={filter.needs} onChange={(needs) => set({ needs })} /></td>
      <td><FilterText label="Filter detail" value={filter.detail ?? ''} onChange={(detail) => set({ detail })} /></td>
      <td />
      <td />
      <td>
        <select aria-label="Filter flags" value={filter.flags?.[0] ?? ''}
                onChange={(e) => set({ flags: e.target.value ? [e.target.value as Flag] : [] })}>
          <option value="">Any</option>
          {FLAGS.map((flag) => <option key={flag} value={flag}>{flag}</option>)}
        </select>
      </td>
    </tr>
  )
}

function Repositories({ filter }: { filter: TitleFilter }) {
  const status = useStatus()
  const plan = usePlan()
  const run = useSubmitRun()
  const navigate = useNavigate()
  const [asking, setAsking] = useState<'publish' | 'commit' | null>(null)
  const refusal = repositoryRefusal(status.data)
  const request = (through: 'publish' | 'commit') => ({
    filter: cleanFilter({ ...filter, needs: [through] }), through, scan_first: false, retry_failed: false, redesign: false, revise_to: 'design' as const,
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

/** A filter that is true, false or either, as a drop-down. */
function Tristate({ label, value, onChange, yes, no }: {
  label: string; value: boolean | null | undefined; onChange: (value: boolean | null) => void; yes: string; no: string
}) {
  return (
    <label className="inline">
      {label}
      <select aria-label={label} value={value === undefined || value === null ? '' : String(value)}
              onChange={(e) => onChange(e.target.value === '' ? null : e.target.value === 'true')}>
        <option value="">Any</option>
        <option value="true">{yes}</option>
        <option value="false">{no}</option>
      </select>
    </label>
  )
}

/** Sends every title the list shows back for redesign, by design version: one job, the whole designer pool at once. */
function RedesignOutOfDate({ filter, total }: { filter: TitleFilter, total: number }) {
  const run = useSubmitRun()
  const navigate = useNavigate()
  const [asking, setAsking] = useState(false)
  const request = (): Schemas['RunJobRequest'] => ({
    filter: cleanFilter({ ...filter, ids: [] }), through: 'design', redesign: true, revise_to: 'design', scan_first: false,
    retry_failed: false,
  })
  return (
    <div className="redesign">
      <div className="actions">
        <button type="button" className="secondary" onClick={() => setAsking(true)}>Redesign out of date…</button>
      </div>
      {asking && (
        <div className="confirm" role="group" aria-label="Confirm redesign">
          <p>Send {count(total, 'title')} back to the designer? Each must be reviewed again. Titles a person accepted or
            published are left as they are.</p>
          <div className="actions">
            <button type="button" disabled={run.isPending}
                    onClick={() => run.mutate(request(), { onSuccess: (job) => navigate(`/jobs/${job.id}`) })}>Redesign</button>
            <button type="button" className="secondary" onClick={() => setAsking(false)}>Back</button>
          </div>
        </div>
      )}
      {run.error && <p role="alert" className="problem">{run.error.message}</p>}
    </div>
  )
}

export function TitlesPage() {
  const { filter, order, includeDone, offset, update, query } = useListParams()
  const status = useStatus()
  const titles = useTitles(filter, includeDone, offset, order)
  const next = useNextTitle()
  const navigate = useNavigate()
  const [nothing, setNothing] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [revising, setRevising] = useState(false)
  const [revised, setRevised] = useState('')
  const [selecting, setSelecting] = useState(false)
  const matching = useMatchingIds()
  const [listed, setListed] = useState(query)
  if (listed !== query) {   // another list: what was selected in the last one is not in it
    setListed(query)
    setSelected(new Set())
    setRevising(false)
  }
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

  async function selectEveryMatching() {
    setSelecting(true)
    try {
      const found = await matching(filter, includeDone)
      setSelected(new Set(found.ids))
      setRevised(found.total > found.ids.length
        ? `Selected the first ${found.ids.length.toLocaleString()} of ${found.total.toLocaleString()}: a revise takes at most ${MAX_REVISED}.`
        : '')
    } catch (error) {
      setRevised(error instanceof Error ? error.message : String(error))
    } finally {
      setSelecting(false)
    }
  }

  const page = titles.data
  const onPage = page?.titles.map((t) => t.id) ?? []
  const allOnPage = onPage.length > 0 && onPage.every((id) => selected.has(id))
  const select = (ids: string[], on: boolean) => setSelected((was) => {
    const next = new Set(was)
    for (const id of ids) {
      if (on) next.add(id)
      else next.delete(id)
    }
    return next
  })
  return (
    <section>
      <div className="title-row">
        <h1>Titles</h1>
        <div className="actions">
          {status.data && <ScanSources scanned={!!status.data.index} />}
          <button type="button" onClick={reviewNext}>Review next waiting</button>
        </div>
      </div>
      {nothing && <p role="status" className="muted">{nothing}</p>}
      <div className="actions">
        <label className="check">
          <input type="checkbox" checked={!!filter.new_since_scan}
                 onChange={(e) => update((f) => ({ ...f, new_since_scan: e.target.checked }))} />
          New since the last scan
        </label>
        <label className="check">
          <input type="checkbox" checked={includeDone} onChange={(e) => update(filter, e.target.checked)} />
          Include titles that need nothing
        </label>
        <label className="inline">
          Designer
          <select aria-label="Designer" value={filter.declined === undefined || filter.declined === null ? '' : String(filter.declined)}
                  onChange={(e) => update((f) => ({ ...f, declined: e.target.value === '' ? null : e.target.value === 'true' }))}>
            <option value="">Any</option>
            <option value="true">Declined</option>
            <option value="false">Not declined</option>
          </select>
        </label>
        <label className="inline">
          Design version
          <select aria-label="Design version" value={filter.out_of_date === undefined || filter.out_of_date === null ? '' : String(filter.out_of_date)}
                  onChange={(e) => update((f) => ({ ...f, out_of_date: e.target.value === '' ? null : e.target.value === 'true' }))}>
            <option value="">Any</option>
            <option value="true">Out of date</option>
            <option value="false">Not out of date</option>
          </select>
        </label>
        <Tristate label="Adjustments" value={filter.adjusted} onChange={(adjusted) => update((f) => ({ ...f, adjusted }))}
                  yes="Asked for some" no="None asked" />
        <Tristate label="Answers" value={filter.unmet} onChange={(unmet) => update((f) => ({ ...f, unmet }))}
                  yes="Some not honoured" no="All honoured" />
      </div>
      {filter.out_of_date === true && page && page.total > 0 && <>
        <RedesignOutOfDate filter={filter} total={page.total} />
        <ImpactPanel link={link} />
      </>}
      <Repositories filter={filter} />
      {selected.size > 0 && (
        <div className="actions" role="group" aria-label="Selected titles">
          <span>{count(selected.size, 'title')} selected</span>
          {allOnPage && page && page.total > selected.size && selected.size < MAX_REVISED && (
            <button type="button" className="link" disabled={selecting} onClick={() => void selectEveryMatching()}>
              {selecting ? 'Selecting…' : `Select all ${page.total.toLocaleString()} matching${page.total > MAX_REVISED
                ? ` (the first ${MAX_REVISED})` : ''}`}</button>
          )}
          <button type="button" className="secondary" onClick={() => setRevising(true)}>Revise selected…</button>
          <button type="button" className="secondary" onClick={() => setSelected(new Set())}>Clear selection</button>
        </div>
      )}
      {revising && selected.size > 0 && (
        <ReviseDialog ids={[...selected]} name={count(selected.size, 'title')}
                      published={page?.titles.some((t) => selected.has(t.id) && t.publish_state !== '') ?? false}
                      onClose={() => setRevising(false)}
                      onDone={(outcome) => {
                        setRevising(false)
                        setSelected(new Set())
                        setRevised(reviseSummary(outcome))
                      }} />
      )}
      {revised && <p role="status">{revised}</p>}
      {titles.isError && <p role="alert" className="problem">{titles.error.message}</p>}
      <p className="muted">{page ? `${count(page.total, 'title')} (${describeFilter(filter)}).` : ' '}</p>
      <div className="table-wrap">
        <table className="list">
          <thead>
            <tr><th><input type="checkbox" aria-label="Select every title on this page" checked={allOnPage}
                           onChange={(e) => select(onPage, e.target.checked)} /></th>
              {COLUMNS.map((c) => <SortHeading key={c.sort} column={c} order={order} onSort={(by) => update(filter, includeDone, 0, by)} />)}</tr>
            <FilterRow filter={filter} sources={sources} onChange={(change) => update(change)} />
          </thead>
          <tbody>
            {page?.titles.map((t) => (
              <tr key={t.id} className={`tier-${t.tier}`}>
                <td><input type="checkbox" aria-label={`Select ${t.title || t.display_name || t.id}`} checked={selected.has(t.id)}
                           onChange={(e) => select([t.id], e.target.checked)} /></td>
                <td><Link to={link(t.id)}>{t.title || t.display_name || t.id}</Link>{t.is_new && <span className="badge">new</span>}
                  {t.adjusted && <span className={`badge${t.unmet_adjustments ? ' warn' : ''}`}>
                    adjusted{t.unmet_adjustments ? `, ${t.unmet_adjustments} not honoured` : ''}</span>}</td>
                <td>{t.year}</td>
                <td>{t.kind === 'tv' ? 'TV' : 'Movie'}</td>
                <td>{t.source}</td>
                <td>{NEEDS_WORDS[t.needs as NeedsWord] ?? t.needs}</td>
                <td className="detail">{t.detail}</td>
                <td className="num">{t.confidence === null ? '—' : t.confidence.toFixed(2)}</td>
                <td className="num">{(t.design_choices || t.candidate_count) || '—'}</td>
                <td>{t.flags.join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {page && page.total > PAGE_SIZE && (
            <div className="actions pager">
              <button type="button" className="secondary" disabled={offset === 0}
                      onClick={() => update(filter, includeDone, Math.max(0, offset - PAGE_SIZE))}>Previous</button>
              <span className="muted">{offset + 1}–{Math.min(offset + PAGE_SIZE, page.total)} of {page.total.toLocaleString()}</span>
              <button type="button" className="secondary" disabled={offset + PAGE_SIZE >= page.total}
                      onClick={() => update(filter, includeDone, offset + PAGE_SIZE)}>Next</button>
            </div>
      )}
    </section>
  )
}
