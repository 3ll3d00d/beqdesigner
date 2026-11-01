// The impact check's titles as a table (contract 1.6, design/redesign-scope.md step 5): a filter at the top of each
// column, "select all shown" to take everything the filters leave, and a checkbox per row to add or drop any by hand.
import { useMemo, useState } from 'react'
import { Link } from 'react-router'

import type { Schemas } from '../api/client'
import { count } from '../format'

type ImpactTitle = Schemas['ImpactTitle']
export type Tier = ImpactTitle['tier']

export const TIER_WORDS: { tier: Tier; label: string }[] = [
  { tier: 'will_move', label: 'Will move' },
  { tier: 'unknown', label: 'Unknown: read as will move' },
  { tier: 'likely', label: 'Likely to move' },
  { tier: 'possible', label: 'Might move' },
  { tier: 'unaffected', label: 'Unaffected' },
]
const TIER_LABEL = new Map(TIER_WORDS.map(({ tier, label }) => [tier, label]))

export interface ImpactFilters {
  title: string
  year: string
  tier: Tier | ''
  minScore: string      // as typed: '' is no bound
  from: string          // a design version, as typed in the select; '' is any
  reason: string
}

export const NO_FILTERS: ImpactFilters = { title: '', year: '', tier: '', minScore: '', from: '', reason: '' }

const has = (text: string, part: string) => text.toLowerCase().includes(part.trim().toLowerCase())

/** Whether `t` is shown under `f`. An unknown has no score and is read as will move, so it passes any minimum. */
export function matches(t: ImpactTitle, f: ImpactFilters): boolean {
  const min = f.minScore.trim() === '' ? null : Number(f.minScore)
  return (!f.title.trim() || has(t.title || t.id, f.title))
    && (!f.year.trim() || has(t.year, f.year))
    && (!f.tier || t.tier === f.tier)
    && (min === null || Number.isNaN(min) || (t.score ?? 1) >= min)
    && (!f.from || String(t.from_design_version ?? '') === f.from)
    && (!f.reason.trim() || t.reasons.some((r) => has(`${r.feature ?? ''} ${r.detail}`, f.reason)))
}

interface Props {
  titles: ImpactTitle[]
  selected: Set<string>
  done: Map<string, string>
  onSelect: (ids: string[], on: boolean) => void
  link: (id: string) => string
}

export function ImpactTable({ titles, selected, done, onSelect, link }: Props) {
  const [filters, setFilters] = useState<ImpactFilters>(NO_FILTERS)
  const set = (change: Partial<ImpactFilters>) => setFilters((f) => ({ ...f, ...change }))
  const shown = useMemo(() => titles.filter((t) => matches(t, filters)), [titles, filters])
  const choosable = shown.filter((t) => !done.has(t.id)).map((t) => t.id)
  const versions = useMemo(() => [...new Set(titles.map((t) => t.from_design_version).filter((v) => v !== null))]
    .sort((a, b) => a! - b!), [titles])
  const filtered = JSON.stringify(filters) !== JSON.stringify(NO_FILTERS)

  return (
    <>
      <div className="actions" role="group" aria-label="Shown titles">
        <span>{shown.length === titles.length ? `${count(titles.length, 'title')}` : `${shown.length.toLocaleString()} of ${count(titles.length, 'title')} shown`}</span>
        <button type="button" className="link" disabled={!choosable.length} onClick={() => onSelect(choosable, true)}>
          Select all {choosable.length.toLocaleString()} shown</button>
        <button type="button" className="link" disabled={!choosable.some((id) => selected.has(id))}
                onClick={() => onSelect(choosable, false)}>Deselect shown</button>
        {filtered && <button type="button" className="link" onClick={() => setFilters(NO_FILTERS)}>Clear filters</button>}
      </div>
      <div className="impact-table">
        <table aria-label="Titles a redesign could move">
          <thead>
            <tr>
              <th aria-label="Chosen" />
              <th>Title</th>
              <th>Year</th>
              <th>Tier</th>
              <th className="num">Probability</th>
              <th className="num">From</th>
              <th>Reasons</th>
            </tr>
            <tr className="filters">
              <th />
              <th><input aria-label="Filter title" placeholder="contains" value={filters.title}
                         onChange={(e) => set({ title: e.target.value })} /></th>
              <th><input aria-label="Filter year" placeholder="contains" size={5} value={filters.year}
                         onChange={(e) => set({ year: e.target.value })} /></th>
              <th>
                <select aria-label="Filter tier" value={filters.tier} onChange={(e) => set({ tier: e.target.value as Tier | '' })}>
                  <option value="">Any</option>
                  {TIER_WORDS.map(({ tier, label }) => <option key={tier} value={tier}>{label}</option>)}
                </select>
              </th>
              <th className="num">
                <label className="inline" title="An unknown has no probability and is read as will move: it is always shown">
                  ≥ <input aria-label="Minimum probability" type="number" min={0} max={1} step={0.05} placeholder="0"
                           value={filters.minScore} onChange={(e) => set({ minScore: e.target.value })} />
                </label>
              </th>
              <th className="num">
                <select aria-label="Filter design version" value={filters.from} onChange={(e) => set({ from: e.target.value })}>
                  <option value="">Any</option>
                  {versions.map((v) => <option key={v} value={String(v)}>v{v}</option>)}
                </select>
              </th>
              <th><input aria-label="Filter reasons" placeholder="contains" value={filters.reason}
                         onChange={(e) => set({ reason: e.target.value })} /></th>
            </tr>
          </thead>
          <tbody>
            {shown.map((t) => (
              <tr key={t.id} className={selected.has(t.id) ? 'chosen' : undefined}>
                <td>
                  <input type="checkbox" aria-label={`Choose ${t.title || t.id}`} disabled={done.has(t.id)}
                         checked={selected.has(t.id)} onChange={(e) => onSelect([t.id], e.target.checked)} />
                </td>
                <td>
                  <Link to={link(t.id)}>{t.title || t.id}</Link>
                  {done.has(t.id) && <> <span className="badge">{done.get(t.id)}</span></>}
                </td>
                <td>{t.year}</td>
                <td>{TIER_LABEL.get(t.tier) ?? t.tier}</td>
                <td className="num">{t.score === null ? '—' : t.score.toFixed(2)}</td>
                <td className="num">{t.from_design_version === null ? '' : `v${t.from_design_version}`}</td>
                <td className="detail">
                  {t.reasons.map((r, i) => <div key={i}>{r.feature ? `${r.feature}: ` : ''}{r.detail}</div>)}
                </td>
              </tr>
            ))}
            {!shown.length && <tr><td colSpan={7} className="muted">No title matches these filters.</td></tr>}
          </tbody>
        </table>
      </div>
    </>
  )
}
