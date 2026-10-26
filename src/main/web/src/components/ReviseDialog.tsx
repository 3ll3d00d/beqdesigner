// Sending titles back (design/web-parity.md WP5): to review, design or extract, in the desktop's words
// (model/worklist_revise.py CHOICES), now in one run job or for the next run. One title from its review, or the titles
// selected on the Titles page.
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import type { Schemas } from '../api/client'
import { useRevise, useStatus } from '../api/hooks'
import { WindowField, type Window } from './WindowField'
import { FREQUENCY_RANGE, type Range } from '../chart/view'
import { count } from '../format'
import { AdjustmentsEditor, askProblem, bandAsk, bandsOf, type Ask, type KindSpec } from './AdjustmentsEditor'

const NO_BANDS: Range[] = []

type To = Schemas['ReviseRequest']['to']

export const REVISE_CHOICES: { to: To; label: string; what: string }[] = [
  { to: 'review', label: 'Reopen for review',
    what: 'Back to "Waiting for review". Its candidates, metadata, artwork and projects are kept and nothing is redesigned; '
      + 'you pick again.' },
  { to: 'design', label: 'Redesign',
    what: 'As reopening, and the design is marked out of date so Extract & design can design it again. What you set is '
      + 'kept: metadata, artwork, your note and any edit you made to a project.' },
  { to: 'extract', label: 'Re-extract and redesign',
    what: 'As redesigning, and the extracted audio is forgotten, so the next run extracts it again with ffmpeg.' },
]

interface Props {
  ids: string[]
  /** What the titles are called: one title's name, else how many. */
  name: string
  /** One of them is published: its uncommitted catalogue files are put back, committed ones stay for a revision. */
  published?: boolean
  onClose: () => void
  /** Sent back now (no job): what came of it. */
  onDone?: (outcome: Schemas['ReviseOutcome']) => void
  /** What the next design of the title is asked now (one title), to start from. */
  asked?: Ask[]
  /** The chart's frequency range, for an adjustment that ignores it. */
  chartRange?: Range
  /** Hands the page what to do with the next drag across the chart (null: stop waiting for one). */
  onPick?: (take: ((range: Range) => void) | null) => void
  /** A drag on the chart is awaited. */
  picking?: boolean
  /** The bands the asks ignore, for the page to shade on the chart. */
  onBands?: (bands: Range[]) => void
}

export function ReviseDialog({ ids, name, published = false, onClose, onDone, asked = [], chartRange = FREQUENCY_RANGE,
                              onPick, picking = false, onBands }: Props) {
  const revise = useRevise()
  const status = useStatus()
  const navigate = useNavigate()
  const [to, setTo] = useState<To>('design')
  const [now, setNow] = useState(true)
  const [reason, setReason] = useState('')
  const [jobWindow, setJobWindow] = useState<Window>(null)
  const [adjusting, setAdjusting] = useState(false)
  const [asks, setAsks] = useState<Ask[]>(() => asked.map((a) => ({ kind: a.kind, params: a.params ?? {}, text: a.text ?? null })))
  const kinds = (status.data?.designer?.adjustment_kinds ?? {}) as Record<string, KindSpec>
  const takes = Object.keys(kinds).length > 0
  const chosen = REVISE_CHOICES.find((c) => c.to === to)!
  const run = now && to !== 'review'
  const sending = adjusting && to !== 'review'
  const problem = sending ? asks.map((a) => askProblem(a, kinds[a.kind])).find(Boolean) ?? '' : ''
  const shaded = sending ? bandsOf(asks) : NO_BANDS
  const shadedKey = JSON.stringify(shaded)
  const page = useRef({ onBands, onPick })   // the page's latest callbacks, without making each render a change
  useEffect(() => { page.current = { onBands, onPick } })
  useEffect(() => {
    page.current.onBands?.(JSON.parse(shadedKey) as Range[])
  }, [shadedKey])
  useEffect(() => () => {   // closed: nothing shaded, and no drag awaited
    page.current.onBands?.([])
    page.current.onPick?.(null)
  }, [])
  function submit() {
    revise.mutate({ ids, to, reason: reason.trim(), run, ...(sending ? { adjustments: asks } : {}),
                    ...(run && jobWindow !== null ? { window: jobWindow } : {}) }, {
      onSuccess: (outcome) => {
        if (outcome.job) navigate(`/jobs/${outcome.job.id}`)
        else onDone?.(outcome)
      },
    })
  }
  return (
    <div className="confirm" role="group" aria-label="Revise">
      <fieldset>
        <legend>Send {name} back</legend>
        {REVISE_CHOICES.map((c) => (
          <label key={c.to} className="check">
            <input type="radio" name="revise-to" checked={to === c.to} onChange={() => setTo(c.to)} /> {c.label}
          </label>
        ))}
      </fieldset>
      <p>{chosen.what}</p>
      {to !== 'review' && (
        <label className="check">
          <input type="checkbox" checked={now} onChange={(e) => setNow(e.target.checked)} />
          Extract &amp; design now, in one job (otherwise the next run does it)
        </label>
      )}
      {run && <WindowField value={jobWindow} onChange={setJobWindow} legend="The job runs" />}
      <p className="muted">{run
        ? 'Review the new designs when the job finishes; publish and commit remain separate.'
        : 'Nothing is redesigned, extracted or published now: this only changes what each title needs next.'}</p>
      {published && <p>A published title's uncommitted catalogue files are put back as git has them; committed files stay
        until you publish again (a revision) and commit it.</p>}
      {to !== 'review' && (
        <div className="adjusting">
          <label className="check">
            <input type="checkbox" checked={adjusting} disabled={!takes} onChange={(e) => setAdjusting(e.target.checked)} />
            Ask the designer to change something{ids.length > 1 ? ' (the same for every title)' : ''}
          </label>
          {!takes && <p className="muted">The designer lists no adjustments it takes (it is older than contract 1.5, or
            it cannot be reached): it is asked nothing.</p>}
          {adjusting && (
            <AdjustmentsEditor kinds={kinds} value={asks} onChange={setAsks} chartRange={chartRange} picking={picking}
                               onPickBand={onPick && (() => onPick(picking ? null
                                 : (range) => setAsks((was) => [...was, bandAsk(range)])))} />
          )}
          {!adjusting && asked.length > 0 && <p className="muted">The next design is asked what this one was: change it
            above.</p>}
        </div>
      )}
      <label className="inline">Why (kept in the note)
        <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="optional" />
      </label>
      <div className="actions">
        <button type="button" disabled={revise.isPending || !!problem} title={problem || undefined}
                onClick={submit}>{chosen.label}</button>
        <button type="button" className="secondary" disabled={revise.isPending} onClick={onClose}>Back</button>
      </div>
      {revise.error && <p role="alert" className="problem">{revise.error.message}</p>}
    </div>
  )
}

/** What a revise done now came to, in words. */
export function reviseSummary(outcome: Schemas['ReviseOutcome']): string {
  const parts = []
  if (outcome.revised.length) parts.push(`Sent back: ${count(outcome.revised.length, 'title')}.`)
  if (outcome.failed.length) {
    parts.push(`Not sent back: ${outcome.failed.map((f) => `${f.id} (${f.reason})`).join('; ')}.`)
  }
  return parts.join(' ') || 'Nothing was sent back.'
}
