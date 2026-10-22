// A title's design runs (design/web-parity.md WP7): the current one and every earlier one kept, each with what the
// designer was asked to change and what it did with each ask. Their designs are picked from and compared in the review's
// one list of designs; here an earlier run can be made current again (the current one is kept) or discarded.
import { useState } from 'react'

import type { Schemas } from '../api/client'
import { useDiscardVariant, useUseVariant } from '../api/hooks'
import { describeAsk } from './AdjustmentsEditor'

type Review = Schemas['Review']
type Asked = Record<string, unknown>

/** What became of one ask, in words: the designer's answer, or that it gave none. */
export function outcomeText(ask: Asked, outcomes: Asked[]): string {
  const answer = outcomes.find((o) => o.id === ask.id)
  if (!answer) return 'not acknowledged (the designer did not answer it)'
  const reason = answer.reason ? `: ${String(answer.reason)}` : ''
  switch (answer.outcome) {
    case 'honoured': return 'honoured'
    case 'adapted': return `adapted, as ${Object.entries((answer.applied ?? {}) as Asked).map(([k, v]) => `${k} ${String(v)}`)
      .join(', ')}${reason}`
    case 'declined': return `declined${reason}`
    default: return `not understood${reason}`
  }
}

function Asks({ asked, outcomes }: { asked: Asked[]; outcomes: Asked[] }) {
  if (!asked.length) return <p className="muted asked-nothing">asked nothing</p>
  return (
    <ul className="asks">
      {asked.map((a) => (
        <li key={String(a.id)}>
          {describeAsk(a as { kind: string; params?: Record<string, unknown>; text?: string | null })}
          <span className={outcomes.some((o) => o.id === a.id && o.outcome === 'honoured') ? 'muted' : 'warn'}>
            {' '}— {outcomeText(a, outcomes)}</span>
        </li>
      ))}
    </ul>
  )
}

const when = (at?: string | null) => (at ? new Date(at).toLocaleString() : 'when not recorded')

export function VariantsPanel({ review }: { review: Review }) {
  const use = useUseVariant(review.id)
  const discard = useDiscardVariant(review.id)
  const [discarding, setDiscarding] = useState<string | null>(null)
  const decided = review.status === 'accepted' || review.status === 'published'
  const busy = use.isPending || discard.isPending || review.in_flight
  const failed = use.error ?? discard.error
  const pending = review.adjustments.length > 0 && JSON.stringify(review.adjustments) !== JSON.stringify(review.adjusted_with)
  return (
    <section className="card variants" aria-labelledby="variants-heading">
      <h2 id="variants-heading">Design runs <span className="muted">({1 + review.variants.length})</span></h2>
      <div className="variant current">
        <strong>{review.variant_id ?? 'This design'}</strong> <span className="muted">(current) · {when(review.designed_at)}</span>
        <Asks asked={review.adjusted_with as Asked[]} outcomes={review.adjustment_outcomes as Asked[]} />
        {pending && <p className="muted">The next design is asked: {review.adjustments.map((a) => describeAsk(
          a as { kind: string; params?: Record<string, unknown>; text?: string | null })).join('; ')}.</p>}
      </div>
      {review.variants.length === 0 && <p className="muted">No earlier design runs: a redesign keeps this one here.</p>}
      {[...review.variants].reverse().map((v) => (
        <div key={v.variant_id} className="variant">
          <strong>{v.variant_id}</strong>
          <span className="muted"> · {when(v.designed_at)} · {v.declined ? 'declined' : `${v.designs} design${v.designs === 1 ? '' : 's'}`}
            {v.designer_build ? ` · ${v.designer_build}` : ''}</span>
          <Asks asked={v.adjusted_with as Asked[]} outcomes={v.adjustment_outcomes as Asked[]} />
          <div className="actions">
            <button type="button" className="secondary small" disabled={busy || decided}
                    title={decided ? 'Reopen the title for review before using another design run' : undefined}
                    onClick={() => use.mutate({ variant: v.variant_id, digest: review.digest })}>Use this one</button>
            {discarding === v.variant_id ? (
              <>
                <button type="button" className="danger small" disabled={busy}
                        onClick={() => discard.mutate({ variant: v.variant_id, digest: review.digest }, {
                          onSettled: () => setDiscarding(null) })}>Discard {v.variant_id}</button>
                <button type="button" className="secondary small" onClick={() => setDiscarding(null)}>Keep it</button>
              </>
            ) : (
              <button type="button" className="secondary small" disabled={busy} onClick={() => setDiscarding(v.variant_id)}>
                Discard…</button>
            )}
          </div>
        </div>
      ))}
      {failed && <p role="alert" className="problem">{failed.message}</p>}
    </section>
  )
}
