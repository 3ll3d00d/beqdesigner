// Which out-of-date designs to redesign (contract 1.6, design/redesign-scope.md step 5): ask the designer what a
// redesign now would move, read its answer most likely to move first with the reasons, then redesign the titles chosen
// and keep the rest, which then stop showing as out of date. Nothing is redesigned or kept on the answer alone.
import { useState } from 'react'
import { Link } from 'react-router'

import type { Schemas } from '../api/client'
import { MAX_REVISED, useImpact, useKeepReviewed, useRevise, useStartImpact, useStatus } from '../api/hooks'
import { count } from '../format'

type Tier = Schemas['ImpactTitle']['tier']

export const TIER_WORDS: { tier: Tier; label: string }[] = [
  { tier: 'will_move', label: 'Will move' },
  { tier: 'unknown', label: 'Unknown: read as will move' },
  { tier: 'likely', label: 'Likely to move' },
  { tier: 'possible', label: 'Might move' },
  { tier: 'unaffected', label: 'Unaffected' },
]
const MOVING: Tier[] = ['will_move', 'unknown', 'likely']

/** What became of a title chosen here. */
type Done = 'sent back' | 'kept'

export function ImpactPanel({ link }: { link: (id: string) => string }) {
  const impact = useImpact()
  const start = useStartImpact()
  const keep = useKeepReviewed()
  const revise = useRevise()
  const status = useStatus()
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [done, setDone] = useState<Map<string, Done>>(new Map())
  const [said, setSaid] = useState('')

  const check = impact.data
  const running = check?.state === 'running' || start.isPending
  const current = status.data?.designer?.design_version ?? null
  const answer = check?.state === 'succeeded' ? check : null
  const stale = answer !== null && current !== null && answer.design_version !== current
  const open = (answer?.titles ?? []).filter((t) => !done.has(t.id))
  const pick = (tiers: Tier[]) => setSelected(new Set(open.filter((t) => tiers.includes(t.tier)).map((t) => t.id)))
  const toggle = (id: string) => setSelected((s) => {
    const next = new Set(s)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    return next
  })
  const finish = (ids: string[], what: Done, text: string) => {
    setDone((d) => new Map([...d, ...ids.map((id) => [id, what] as const)]))
    setSelected(new Set())
    setSaid(text)
  }
  const ids = [...selected]
  const tooMany = ids.length > MAX_REVISED

  return (
    <section className="impact" aria-label="What a redesign would move">
      <div className="actions">
        <button type="button" className="secondary" disabled={running} onClick={() => {
          setDone(new Map())
          setSelected(new Set())
          setSaid('')
          start.mutate([])
        }}>{running ? 'Checking…' : answer ? 'Check again' : 'Check what a redesign would move'}</button>
        {running && check && <span className="muted" role="status">
          Asked about {check.done.toLocaleString()} of {count(check.total, 'title')}…</span>}
      </div>
      {start.error && <p role="alert" className="problem">{start.error.message}</p>}
      {check?.state === 'failed' && <p role="alert" className="problem">The check failed: {check.error}</p>}
      {stale && <p className="problem">This answer is for design version {answer.design_version}; the designer now
        designs with {current}. Check again.</p>}
      {answer && !stale && (
        <>
          <p className="muted">
            {TIER_WORDS.map(({ tier, label }) => {
              const n = answer.titles.filter((t) => t.tier === tier).length
              return n ? `${label}: ${n.toLocaleString()}` : null
            }).filter(Boolean).join(' · ')}
            {answer.changes.length > 0 && <> — changes: {answer.changes.map((c) => `v${c.design_version} ${c.summary}`).join('; ')}</>}
          </p>
          <div className="actions" role="group" aria-label="Choose titles">
            <button type="button" className="link" onClick={() => pick(MOVING)}>Select those likely to move</button>
            <button type="button" className="link" onClick={() => pick(['possible', 'unaffected'])}>Select the rest</button>
            {selected.size > 0 && <>
              <span>{count(selected.size, 'title')} selected</span>
              <button type="button" disabled={revise.isPending || tooMany} onClick={() => revise.mutate(
                { ids, to: 'design', reason: 'a redesign now could move its design', run: true },
                { onSuccess: () => finish(ids, 'sent back', `Sent ${count(ids.length, 'title')} back to the designer in one run.`) })}>
                Redesign selected</button>
              <button type="button" className="secondary" disabled={keep.isPending || tooMany} onClick={() => keep.mutate(ids, {
                onSuccess: (outcome) => {
                  const kept = outcome.titles.filter((t) => t.reviewed).map((t) => t.id)
                  const refused = outcome.titles.filter((t) => !t.reviewed)
                  finish(kept, 'kept', [`Kept ${count(kept.length, 'design')} at design version ${outcome.design_version}.`,
                    ...(refused.length ? [`Not kept: ${refused.map((t) => `${t.id} (${t.reason})`).join('; ')}.`] : [])].join(' '))
                },
              })}>Keep selected</button>
            </>}
          </div>
          {tooMany && <p className="problem">Choose at most {MAX_REVISED} at a time.</p>}
          {revise.error && <p role="alert" className="problem">{revise.error.message}</p>}
          {keep.error && <p role="alert" className="problem">{keep.error.message}</p>}
          {said && <p role="status">{said}</p>}
          {TIER_WORDS.map(({ tier, label }) => {
            const rows = answer.titles.filter((t) => t.tier === tier)
            if (!rows.length) return null
            return (
              <div key={tier} className="impact-tier">
                <h3>{label} ({rows.length.toLocaleString()})</h3>
                <ul>
                  {rows.map((t) => (
                    <li key={t.id}>
                      <label className="inline">
                        <input type="checkbox" aria-label={`Choose ${t.title || t.id}`} disabled={done.has(t.id)}
                               checked={selected.has(t.id)} onChange={() => toggle(t.id)} />
                      </label>{' '}
                      <Link to={link(t.id)}>{t.title || t.id}</Link>{t.year && ` (${t.year})`}
                      {t.score !== null && <span className="muted"> {t.score.toFixed(2)}</span>}
                      {done.has(t.id) && <span className="badge">{done.get(t.id)}</span>}
                      {t.reasons.length > 0 && (
                        <ul className="muted reasons">
                          {t.reasons.map((r, i) => <li key={i}>{r.feature ? `${r.feature}: ` : ''}{r.detail}</li>)}
                        </ul>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )
          })}
        </>
      )}
    </section>
  )
}
