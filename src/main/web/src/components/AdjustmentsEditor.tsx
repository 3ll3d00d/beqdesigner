// What a redesign asks the designer to change (contract 1.5, design/web-parity.md WP7): a list of adjustments, each one of
// the kinds the designer lists on its /health (the service passes them on in /v1/status), with the parameters it says,
// and a note in words. The designer decides what to do with each and says so; the review shows its answers.
import type { Schemas } from '../api/client'
import type { Range } from '../chart/view'

export type Ask = Schemas['AdjustmentAsk']
type ParamSpec = { type?: string; unit?: string; minimum?: number; maximum?: number; default?: unknown; description?: string }
export type KindSpec = { description: string; params?: Record<string, ParamSpec> }

/** Common asks, as kinds and parameters: offered only when the designer lists the kind. */
export const PRESETS: { label: string; ask: (range: Range) => Ask }[] = [
  { label: 'Ignore the chart\'s range', ask: ([low, high]) => ({ kind: 'exclude_band', params: { low_hz: round(low), high_hz: round(high) } }) },
  { label: 'Bass boost +1 dB/oct', ask: () => ({ kind: 'goal_tilt', params: { db_per_octave: 1 } }) },
  { label: 'Cap the boost at 15 dB', ask: () => ({ kind: 'max_boost', params: { db: 15 } }) },
  { label: 'Nothing below 10 Hz', ask: () => ({ kind: 'floor', params: { hz: 10 } }) },
]

const round = (hz: number) => Math.round(hz * 10) / 10

/** A new ask of `kind`, its parameters at the designer's defaults. */
export function blankAsk(kind: string, spec: KindSpec | undefined): Ask {
  const params: Record<string, unknown> = {}
  for (const [name, p] of Object.entries(spec?.params ?? {})) if (p.default !== undefined) params[name] = p.default
  return { kind, params, text: null }
}

/** Why an ask cannot be sent as it is, in words; '' when it can. The designer still judges it. */
export function askProblem(ask: Ask, spec: KindSpec | undefined): string {
  if (ask.kind === 'note') return ask.text?.trim() ? '' : 'Say what you want in the note'
  for (const [name, p] of Object.entries(spec?.params ?? {})) {
    const value = ask.params?.[name]
    if (value === undefined || value === null || value === '') return `${name} is needed`
    if ((p.type === 'number' || p.type === 'integer') && typeof value === 'number') {
      if (p.minimum !== undefined && value < p.minimum) return `${name} is at least ${p.minimum}`
      if (p.maximum !== undefined && value > p.maximum) return `${name} is at most ${p.maximum}`
    }
  }
  if (ask.kind === 'exclude_band' && Number(ask.params?.low_hz) >= Number(ask.params?.high_hz)) {
    return 'The band\'s low end must be below its high end'
  }
  return ''
}

/** An ask as one line of words. */
export function describeAsk(ask: { kind: string; params?: Record<string, unknown>; text?: string | null }): string {
  const p = ask.params ?? {}
  switch (ask.kind) {
    case 'exclude_band': return `ignore ${p.low_hz}-${p.high_hz} Hz`
    case 'floor': return `nothing below ${p.hz} Hz`
    case 'max_boost': return `boost at most ${p.db} dB`
    case 'goal_tilt': return `tilt the goal ${Number(p.db_per_octave) > 0 ? '+' : ''}${p.db_per_octave} dB/oct`
    case 'note': return `note: "${ask.text ?? ''}"`
    default: {
      const said = Object.entries(p).map(([k, v]) => `${k} ${String(v)}`).join(', ')
      return `${ask.kind}${said ? ` (${said})` : ''}${ask.text ? `: "${ask.text}"` : ''}`
    }
  }
}

interface Props {
  kinds: Record<string, KindSpec>
  value: Ask[]
  onChange: (asks: Ask[]) => void
  /** The magnitude chart's frequency range, for "ignore the chart's range". */
  chartRange: Range
}

export function AdjustmentsEditor({ kinds, value, onChange, chartRange }: Props) {
  const names = Object.keys(kinds)
  const set = (i: number, ask: Ask) => onChange(value.map((a, j) => (j === i ? ask : a)))
  return (
    <div className="adjustments" role="group" aria-label="Adjustments">
      {value.length === 0 && <p className="muted">Nothing asked: the designer works as it would.</p>}
      <ol>
        {value.map((ask, i) => {
          const spec = kinds[ask.kind]
          const problem = askProblem(ask, spec)
          return (
            <li key={i}>
              <label className="inline">Kind
                <select value={ask.kind} onChange={(e) => set(i, blankAsk(e.target.value, kinds[e.target.value]))}>
                  {[...new Set([...names, ask.kind])].map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
              </label>
              {spec && <span className="muted">{spec.description}</span>}
              {Object.entries(spec?.params ?? {}).map(([name, p]) => (
                <label key={name} className="inline" title={p.description}>{name}{p.unit ? ` (${p.unit})` : ''}
                  {p.type === 'boolean'
                    ? <input type="checkbox" checked={!!ask.params?.[name]}
                             onChange={(e) => set(i, { ...ask, params: { ...ask.params, [name]: e.target.checked } })} />
                    : <input inputMode={p.type === 'string' ? 'text' : 'decimal'} className="number"
                             value={ask.params?.[name] === undefined ? '' : String(ask.params[name])}
                             onChange={(e) => {
                               const text = e.target.value
                               const parsed = p.type === 'string' || text.trim() === '' || Number.isNaN(Number(text))
                                 ? text : Number(text)
                               set(i, { ...ask, params: { ...ask.params, [name]: parsed } })
                             }} />}
                </label>
              ))}
              <label className="inline">{ask.kind === 'note' ? 'Note' : 'Why (optional)'}
                <input value={ask.text ?? ''} onChange={(e) => set(i, { ...ask, text: e.target.value || null })} />
              </label>
              <button type="button" className="secondary small" onClick={() => onChange(value.filter((_, j) => j !== i))}>
                Remove</button>
              {problem && <span className="problem" role="alert">{problem}</span>}
            </li>
          )
        })}
      </ol>
      <div className="actions">
        {names.length > 0 && (
          <button type="button" className="secondary small" onClick={() => onChange([...value, blankAsk(names[0]!, kinds[names[0]!])])}>
            Add an adjustment</button>
        )}
        {PRESETS.map((preset) => {
          const ask = preset.ask(chartRange)
          return kinds[ask.kind] && (
            <button key={preset.label} type="button" className="secondary small" onClick={() => onChange([...value, ask])}>
              {preset.label}</button>
          )
        })}
      </div>
    </div>
  )
}
