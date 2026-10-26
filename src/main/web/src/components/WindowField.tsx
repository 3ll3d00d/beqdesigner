// When a job may run (design/job-windows.md §5): the default window, any time, or daily ranges typed a line each, in the
// service's time zone -- which the field names, rather than converting, since that is the clock the service keeps.
import { useState } from 'react'

import type { Schemas } from '../api/client'
import { useDefaultWindow, useSaveDefaultWindow } from '../api/hooks'
import { when } from '../format'

/** null: the default; []: any time; else the ranges. */
export type Window = string[] | null

/** The ranges typed, a line (or comma) each, blanks dropped; the service checks and normalises them. */
export function parseRanges(text: string): string[] {
  return text.split(/[\n,]/).map((line) => line.trim()).filter(Boolean)
}

export function describeRanges(ranges: string[]): string {
  return ranges.length ? ranges.join(', ') : 'any time'
}

/** A job's window in words, the default's spelled out when the job follows it. */
export function describeWindow(window: Window | undefined, defaults?: Schemas['DefaultWindow']): string {
  if (window === null || window === undefined) {
    return defaults ? `the default (${describeRanges(defaults.window)}, ${defaults.time_zone})` : 'the default'
  }
  return defaults ? `${describeRanges(window)} (${defaults.time_zone})` : describeRanges(window)
}

type Mode = 'default' | 'any' | 'ranges'

const modeOf = (window: Window): Mode => (window === null ? 'default' : window.length ? 'ranges' : 'any')

export function WindowField({ value, onChange, legend = 'Runs' }:
                            { value: Window; onChange: (window: Window) => void; legend?: string }) {
  const defaults = useDefaultWindow()
  const [mode, setMode] = useState<Mode>(modeOf(value))
  const [text, setText] = useState((value ?? []).join('\n'))
  const choose = (next: Mode, typed = text) => {
    setMode(next)
    onChange(next === 'default' ? null : next === 'any' ? [] : parseRanges(typed))
  }
  return (
    <fieldset>
      <legend>{legend}</legend>
      <label className="check">
        <input type="radio" name="window" checked={mode === 'default'} onChange={() => choose('default')} />
        When the default allows{defaults.data ? ` (${describeRanges(defaults.data.window)})` : ''}
      </label>
      <label className="check">
        <input type="radio" name="window" checked={mode === 'any'} onChange={() => choose('any')} />
        Any time
      </label>
      <label className="check">
        <input type="radio" name="window" checked={mode === 'ranges'} onChange={() => choose('ranges')} />
        Only between
      </label>
      {mode === 'ranges' && (
        <label className="inline">
          Times{defaults.data ? ` (${defaults.data.time_zone})` : ''}, one range a line
          <textarea aria-label="Times" rows={2} value={text} placeholder="01:00-07:00"
                    onChange={(e) => { setText(e.target.value); choose('ranges', e.target.value) }} />
        </label>
      )}
      <p className="muted">Outside these times it waits; a run that is going pauses between titles and carries on when
        they next come round.</p>
    </fieldset>
  )
}

/** The window every job given none follows, and when it next opens or closes; editable. */
export function DefaultWindow() {
  const defaults = useDefaultWindow()
  const save = useSaveDefaultWindow()
  const [editing, setEditing] = useState(false)
  const [text, setText] = useState('')
  const data = defaults.data
  if (!data) return defaults.isError ? <span className="problem">{defaults.error.message}</span> : <span className="muted">…</span>
  const change = data.next_change ? ` ${data.open_now ? 'until' : 'from'} ${when(data.next_change)}` : ''
  return (
    <>
      {data.window.length ? `${describeRanges(data.window)} (${data.time_zone})` : 'Any time'}
      <span className="muted"> — {data.open_now ? 'open now' : 'closed now'}{change}</span>{' '}
      {editing ? (
        <form className="inline" onSubmit={(e) => {
          e.preventDefault()
          save.mutate({ window: parseRanges(text) }, { onSuccess: () => setEditing(false) })
        }}>
          <input aria-label="Default times" value={text} placeholder="01:00-07:00, or blank for any time"
                 onChange={(e) => setText(e.target.value)} />
          <button type="submit" disabled={save.isPending}>Save</button>
          <button type="button" className="secondary" onClick={() => setEditing(false)}>Cancel</button>
          {save.error && <span role="alert" className="problem"> {save.error.message}</span>}
        </form>
      ) : (
        <button type="button" className="secondary" onClick={() => { setText(data.window.join(', ')); setEditing(true) }}>
          Change…
        </button>
      )}
    </>
  )
}
