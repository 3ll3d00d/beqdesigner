// Which titles: the work list's filters (needs, source, kind, year, text, new) as one form, for runs and the titles list.
import { useId } from 'react'

import type { Kind, Needs, TitleFilter } from '../api/filter'
import { NEEDS_ORDER, NEEDS_WORDS } from '../format'

export interface FilterFormProps {
  value: TitleFilter
  onChange: (filter: TitleFilter) => void
  sources: string[]
  /** The needs offered as choices (a run never takes `done`, say). */
  needs?: readonly Needs[]
}

export function FilterForm({ value, onChange, sources, needs = NEEDS_ORDER }: FilterFormProps) {
  const id = useId()
  const chosen = new Set(value.needs ?? [])
  const set = (patch: Partial<TitleFilter>) => onChange({ ...value, ...patch })

  function toggle(need: Needs) {
    const next = new Set(chosen)
    if (next.has(need)) next.delete(need)
    else next.add(need)
    set({ needs: needs.filter((n) => next.has(n)) })
  }

  return (
    <fieldset className="filter">
      <legend>Titles</legend>
      <div className="chips" role="group" aria-label="Needs">
        {needs.map((need) => (
          <label key={need} className={chosen.has(need) ? 'chip on' : 'chip'}>
            <input type="checkbox" checked={chosen.has(need)} onChange={() => toggle(need)} />
            {NEEDS_WORDS[need]}
          </label>
        ))}
      </div>
      <div className="fields">
        <label htmlFor={`${id}-match`}>
          Search
          <input id={`${id}-match`} type="text" value={value.match ?? ''} placeholder="title, name, id or path"
                 onChange={(e) => set({ match: e.target.value || null })} />
        </label>
        <label htmlFor={`${id}-source`}>
          Source
          <select id={`${id}-source`} value={value.source ?? ''} onChange={(e) => set({ source: e.target.value || null })}>
            <option value="">Any</option>
            {sources.map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
        <label htmlFor={`${id}-kind`}>
          Kind
          <select id={`${id}-kind`} value={value.kind ?? ''}
                  onChange={(e) => set({ kind: (e.target.value || null) as Kind | null })}>
            <option value="">Any</option>
            <option value="movie">Movies</option>
            <option value="tv">TV</option>
          </select>
        </label>
        <label htmlFor={`${id}-year`}>
          Year
          <input id={`${id}-year`} type="text" value={value.year ?? ''} placeholder="2026, >=2020, 1990-1999"
                 onChange={(e) => set({ year: e.target.value || null })} />
        </label>
        <label className="check">
          <input type="checkbox" checked={!!value.new_since_scan}
                 onChange={(e) => set({ new_since_scan: e.target.checked })} />
          New since the last scan
        </label>
      </div>
    </fieldset>
  )
}
