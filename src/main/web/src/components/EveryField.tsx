// How often a job runs (design/jobs-and-executions.md §2.1): once, or every N minutes from the end of the last execution.
import { useState } from 'react'

export const MIN_EVERY = 5

/** null: once; else minutes between executions. */
export type Every = number | null

export function EveryField({ value, onChange, allowed = true, why = '' }:
                           { value: Every; onChange: (every: Every) => void; allowed?: boolean; why?: string }) {
  const [minutes, setMinutes] = useState(String(value ?? 60))
  const typed = (text: string) => {
    setMinutes(text)
    const n = Number(text)
    if (Number.isInteger(n) && n >= MIN_EVERY) onChange(n)
  }
  return (
    <fieldset>
      <legend>When</legend>
      <label className="check">
        <input type="radio" name="every" checked={value === null} onChange={() => onChange(null)} />
        Once
      </label>
      <label className="check" title={allowed ? undefined : why}>
        <input type="radio" name="every" checked={value !== null} disabled={!allowed}
               onChange={() => typed(minutes)} />
        Every
      </label>
      {value !== null && (
        <label className="inline">
          <input type="number" aria-label="Minutes" min={MIN_EVERY} step={1} value={minutes}
                 onChange={(e) => typed(e.target.value)} /> minutes, from the end of the last run
        </label>
      )}
      {!allowed && why && <p className="muted">{why}</p>}
    </fieldset>
  )
}
