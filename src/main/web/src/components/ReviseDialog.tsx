// Sending titles back (design/web-parity.md WP5): to review, design or extract, in the desktop's words
// (model/worklist_revise.py CHOICES), now in one run job or for the next run. One title from its review, or the titles
// selected on the Titles page.
import { useState } from 'react'
import { useNavigate } from 'react-router'

import type { Schemas } from '../api/client'
import { useRevise } from '../api/hooks'
import { count } from '../format'

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
}

export function ReviseDialog({ ids, name, published = false, onClose, onDone }: Props) {
  const revise = useRevise()
  const navigate = useNavigate()
  const [to, setTo] = useState<To>('design')
  const [now, setNow] = useState(true)
  const [reason, setReason] = useState('')
  const chosen = REVISE_CHOICES.find((c) => c.to === to)!
  const run = now && to !== 'review'
  function submit() {
    revise.mutate({ ids, to, reason: reason.trim(), run }, {
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
      <p className="muted">{run
        ? 'Review the new designs when the job finishes; publish and commit remain separate.'
        : 'Nothing is redesigned, extracted or published now: this only changes what each title needs next.'}</p>
      {published && <p>A published title's uncommitted catalogue files are put back as git has them; committed files stay
        until you publish again (a revision) and commit it.</p>}
      <label className="inline">Why (kept in the note)
        <input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="optional" />
      </label>
      <div className="actions">
        <button type="button" disabled={revise.isPending} onClick={submit}>{chosen.label}</button>
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
