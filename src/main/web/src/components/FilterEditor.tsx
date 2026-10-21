// Editing a design's filters in the browser (design/web-parity.md WP8): a table of sections -- type, frequency, Q, gain,
// how many stacked -- with the edit's response and the track under it drawn live beside the design it started from,
// computed here as model/iir.py does (src/chart/biquad.ts). *Save as a new design* makes it the title's current design run;
// the one it started from is kept.
import { useState } from 'react'

import type { Schemas } from '../api/client'
import { useSaveEdited } from '../api/hooks'
import { MAX_SECTIONS, responseDb, rowProblem, rowsOf, sections, TYPE_NAMES, type FilterRow, type FilterType } from '../chart/biquad'
import type { ChartView, Series } from '../chart/view'
import { MagnitudeChart } from './MagnitudeChart'

const TYPES = Object.keys(TYPE_NAMES) as FilterType[]

interface Props {
  review: Schemas['Review']
  source: number
  /** The chart route's series for the highlighted design: the measured curves, and after it. */
  series: Series[]
  view: ChartView
  onView: (view: ChartView) => void
  onClose: () => void
  /** A .beq project of the title was edited by hand since its design: that, not this edit, is what gets published. */
  projectEdited?: boolean
}

/** The edit drawn: its response, and the measured average and peak under it. */
export function editedSeries(measured: Series[], rows: FilterRow[], fs: number): Series[] {
  const grid = measured.find((s) => s.kind === 'average' && !s.filtered)?.x ?? []
  if (!grid.length) return []
  const response = responseDb(rows, fs, grid)
  const under = measured.filter((s) => !s.filtered && s.kind !== 'filter').map((s) => {
    const r = s.x === grid ? response : responseDb(rows, fs, s.x)
    return { ...s, name: `${s.name} under the edit`, filtered: true, candidate: -1, y: s.y.map((v, i) => v + r[i]!) }
  })
  return [...under, { name: 'The edit', kind: 'filter' as const, filtered: true, candidate: -1, x: grid, y: response }]
}

export function FilterEditor({ review, source, series, view, onView, onClose, projectEdited = false }: Props) {
  const design = [...review.candidates, ...review.rejected][source]
  const start = rowsOf(design?.filters)
  const [rows, setRows] = useState<FilterRow[]>(start?.rows ?? [])
  const save = useSaveEdited(review.id)
  const fs = start?.fs ?? 96000
  const set = (i: number, change: Partial<FilterRow>) => setRows(rows.map((r, j) => (j === i ? { ...r, ...change } : r)))
  const move = (i: number, by: number) => {
    const next = [...rows]
    const [taken] = next.splice(i, 1)
    next.splice(i + by, 0, taken!)
    setRows(next)
  }
  const problems = rows.map((r) => rowProblem(r, fs))
  const total = sections(rows)
  const problem = !rows.length ? 'A design needs at least one section'
    : total > MAX_SECTIONS ? `${total} sections: at most ${MAX_SECTIONS}` : problems.find(Boolean) ?? ''
  const measured = series.filter((s) => !s.filtered)
  const original = series.filter((s) => s.filtered && s.kind !== 'filter')
  const drawn = problem ? [...measured, ...original] : [...measured, ...original, ...editedSeries(measured, rows, fs)]
  if (!design) return null
  if (!start) {
    return <p className="muted">This design has a filter the editor cannot show. <button type="button" className="link"
      onClick={onClose}>Close</button></p>
  }
  const number = (value: number, onChange: (v: number) => void, label: string) => (
    <input aria-label={label} inputMode="decimal" className="number" defaultValue={String(value)}
           onChange={(e) => { const v = Number(e.target.value); if (e.target.value.trim() !== '' && !Number.isNaN(v)) onChange(v) }} />
  )
  return (
    <section className="card filter-editor" aria-labelledby="editor-heading">
      <h2 id="editor-heading">Edit design {source + 1}</h2>
      <p className="muted">Drawn live beside the design it started from (as model/iir.py computes it). Saving makes it the
        title's current design run, to review and accept; the design it started from is kept.</p>
      {projectEdited && <p className="warn" role="note">A project of this title was edited by hand since its design: what
        is saved in the project, not this edit, is what gets published. Use one or the other.</p>}
      <table className="list">
        <thead><tr><th>Type</th><th>Frequency (Hz)</th><th>Q</th><th>Gain (dB)</th><th>Stacked</th><th /></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${i}-${rows.length}`}>
              <td><select aria-label={`Section ${i + 1} type`} value={r.type}
                          onChange={(e) => set(i, { type: e.target.value as FilterType })}>
                {TYPES.map((t) => <option key={t} value={t}>{TYPE_NAMES[t]}</option>)}
              </select></td>
              <td>{number(r.freq_hz, (v) => set(i, { freq_hz: v }), `Section ${i + 1} frequency`)}</td>
              <td>{number(r.q, (v) => set(i, { q: v }), `Section ${i + 1} Q`)}</td>
              <td>{number(r.gain_db, (v) => set(i, { gain_db: v }), `Section ${i + 1} gain`)}</td>
              <td>{number(r.count, (v) => set(i, { count: v }), `Section ${i + 1} stacked`)}</td>
              <td className="row-actions">
                <button type="button" className="secondary small" disabled={i === 0} onClick={() => move(i, -1)}
                        aria-label={`Move section ${i + 1} up`}>↑</button>
                <button type="button" className="secondary small" disabled={i === rows.length - 1} onClick={() => move(i, 1)}
                        aria-label={`Move section ${i + 1} down`}>↓</button>
                <button type="button" className="secondary small" onClick={() => setRows(rows.filter((_, j) => j !== i))}
                        aria-label={`Remove section ${i + 1}`}>Remove</button>
                {problems[i] && <span className="problem">{problems[i]}</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="actions">
        <button type="button" className="secondary small"
                onClick={() => setRows([...rows, { type: 'low_shelf', freq_hz: 20, q: 0.7, gain_db: 3, count: 1 }])}>
          Add a low shelf (more bass)</button>
        <button type="button" className="secondary small"
                onClick={() => setRows([...rows, { type: 'peaking_eq', freq_hz: 30, q: 1, gain_db: 0, count: 1 }])}>
          Add a peak</button>
        <button type="button" className="secondary small" onClick={() => setRows(start.rows)}>Back to the design</button>
      </div>
      <MagnitudeChart series={drawn} view={{ ...view, responses: true }} highlighted={-1}
                      onZoom={(x, y) => onView({ ...view, x, y })} onReset={() => onView({ ...view, x: [1, 160], y: null })} />
      {problem && <p className="problem" role="alert">{problem}</p>}
      <div className="actions">
        <button type="button" disabled={!!problem || save.isPending}
                onClick={() => save.mutate({ digest: review.digest, source, filters: rows }, { onSuccess: onClose })}>
          Save as a new design</button>
        <button type="button" className="secondary" onClick={onClose}>Close</button>
      </div>
      {save.error && <p role="alert" className="problem">{save.error.message}</p>}
    </section>
  )
}
