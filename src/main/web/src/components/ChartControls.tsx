// The review chart's controls (design/web-parity.md WP1): range presets and limits, a log frequency axis, smoothing, which
// curves are shown, every design at once, normalising to one curve, and going back to the defaults. They change a
// `ChartView`, which the page keeps in this browser.
import { useState } from 'react'

import { DEFAULT_VIEW, highestFrequency, PRESETS, type ChartView, type Range, type Series, SMOOTHINGS, smoothingLabel,
  visibleSeries } from '../chart/view'

interface Props {
  view: ChartView
  series: Series[]
  onChange: (view: ChartView) => void
}

/** A number typed into a limit box; blank is `null`. */
function parseLimit(text: string): number | null | undefined {
  if (!text.trim()) return null
  const value = Number(text)
  return Number.isFinite(value) ? value : undefined
}

function LimitPair({ label, unit, value, auto, onCommit }: {
  label: string; unit: string; value: Range | null; auto?: boolean; onCommit: (range: Range | null) => string
}) {
  const shown = (v: number | undefined) => (v === undefined ? '' : String(Number(v.toFixed(2))))
  const [low, setLow] = useState(shown(value?.[0]))
  const [high, setHigh] = useState(shown(value?.[1]))
  const [problem, setProblem] = useState('')
  const [seen, setSeen] = useState(value)
  if (seen !== value) {   // changed elsewhere (a preset, a zoom): show it
    setSeen(value)
    setLow(shown(value?.[0]))
    setHigh(shown(value?.[1]))
    setProblem('')
  }
  function commit() {
    const a = parseLimit(low), b = parseLimit(high)
    if (a === undefined || b === undefined) return setProblem('Numbers only')
    if (a === null && b === null && auto) return setProblem(onCommit(null))
    if (a === null || b === null) return setProblem(auto ? 'Both, or neither for auto' : 'Both are needed')
    if (a >= b) return setProblem('The first must be the lower')
    setProblem(onCommit([a, b]))
  }
  const key = (event: React.KeyboardEvent) => { if (event.key === 'Enter') commit() }
  return (
    <span className="limits" role="group" aria-label={label}>
      <span>{label}</span>
      <input aria-label={`${label} from`} inputMode="decimal" value={low} placeholder={auto ? 'auto' : ''}
             onChange={(e) => setLow(e.target.value)} onBlur={commit} onKeyDown={key} />
      <span aria-hidden>–</span>
      <input aria-label={`${label} to`} inputMode="decimal" value={high} placeholder={auto ? 'auto' : ''}
             onChange={(e) => setHigh(e.target.value)} onBlur={commit} onKeyDown={key} />
      <span className="muted">{unit}</span>
      {problem && <span className="problem" role="alert">{problem}</span>}
    </span>
  )
}

export function ChartControls({ view, series, onChange }: Props) {
  const set = (change: Partial<ChartView>) => onChange({ ...view, ...change })
  const toggle = (key: 'average' | 'peak' | 'before' | 'after' | 'responses' | 'allDesigns' | 'logX', label: string) => (
    <label className="inline">
      <input type="checkbox" checked={view[key]} onChange={(e) => set({ [key]: e.target.checked })} /> {label}
    </label>
  )
  const references = visibleSeries(series, view).filter((s) => s.kind !== 'filter').map((s) => s.name)
  return (
    <div className="chart-controls" aria-label="Chart controls">
      <div className="row">
        <span role="group" aria-label="Range">
          {PRESETS.map((p) => (
            <button key={p.id} type="button" className="small"
                    onClick={() => set({ x: p.x === 'full' ? [1, highestFrequency(series)] : p.x, y: null })}>
              {p.label}
            </button>
          ))}
        </span>
        <LimitPair label="Frequency" unit="Hz" value={view.x}
                   onCommit={(range) => {
                     if (!range) return 'Both are needed'
                     if (view.logX && range[0] <= 0) return 'A log axis starts above 0 Hz'
                     set({ x: range })
                     return ''
                   }} />
        <LimitPair label="Level" unit="dB" value={view.y} auto onCommit={(range) => { set({ y: range }); return '' }} />
        {toggle('logX', 'Log frequency')}
      </div>
      <div className="row">
        <label className="inline">Smoothing
          <select value={view.smoothing} onChange={(e) => set({ smoothing: e.target.value as ChartView['smoothing'] })}>
            {SMOOTHINGS.map((s) => <option key={s} value={s}>{smoothingLabel(s)}</option>)}
          </select>
        </label>
        {toggle('average', 'Average')}
        {toggle('peak', 'Peak')}
        {toggle('before', 'Before design')}
        {toggle('after', 'After design')}
        {toggle('responses', 'Filter response')}
        {toggle('allDesigns', 'All designs')}
        <label className="inline">Relative to
          <select value={view.reference ?? ''} onChange={(e) => set({ reference: e.target.value || null })}>
            <option value="">Nothing (absolute)</option>
            {view.reference !== null && !references.includes(view.reference)
              && <option value={view.reference}>{view.reference} (not shown)</option>}
            {references.map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
        </label>
        <button type="button" className="small" onClick={() => onChange(DEFAULT_VIEW)}>Reset view</button>
      </div>
    </div>
  )
}
