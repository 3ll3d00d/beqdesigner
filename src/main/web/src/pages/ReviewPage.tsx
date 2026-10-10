// Deciding one title (design/web-app.md §4, design/review-over-http.md): its designs, the chart of the one highlighted,
// commentary and metadata problems, and Accept / Reject / Skip (A, R, S; 1-9 highlight a design). Accepting a design the
// designer rejected asks first. After a decision, or Skip, the next title waiting in the same filtered list opens.
import { useEffect, useState, type ReactNode } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import type { Schemas } from '../api/client'
import { useChart, useDecide, useNextTitle, useReview } from '../api/hooks'
import { MagnitudeChart } from '../components/MagnitudeChart'
import { NEEDS_WORDS, type NeedsWord } from '../format'
import { useListParams } from './TitlesPage'

type Candidate = Schemas['CandidateView']
type Review = Schemas['Review']

const reasons = (candidate: Candidate): string[] => candidate.rejection_reasons ?? []

function number(value: number | null | undefined, unit: string, digits = 1): string {
  return value === null || value === undefined ? '—' : `${value.toFixed(digits)}${unit}`
}

function CandidateLine({ candidate }: { candidate: Candidate }) {
  return (
    <span>
      <strong>{candidate.index + 1}.</strong> {candidate.method}
      {candidate.confidence !== null && <> · confidence {candidate.confidence.toFixed(2)}</>}
      {' '}· MV +{number(candidate.mv_adjust_db, ' dB')}
      {candidate.gain_reduction_db !== null && candidate.gain_reduction_db !== undefined &&
        <> · gain reduction {number(candidate.gain_reduction_db, ' dB')}</>}
      {candidate.residual_db !== null && candidate.residual_db !== undefined && <> · residual {number(candidate.residual_db, ' dB')}</>}
    </span>
  )
}

function Value({ value }: { value: unknown }): ReactNode {
  if (Array.isArray(value)) return <ul>{value.map((v, i) => <li key={i}><Value value={v} /></li>)}</ul>
  if (value && typeof value === 'object') {
    return <dl className="facts">{Object.entries(value).map(([k, v]) => <Entry key={k} name={k} value={v} />)}</dl>
  }
  return <>{String(value ?? '')}</>
}

function Entry({ name, value }: { name: string; value: unknown }) {
  const heading = name.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())
  return <><dt>{heading}</dt><dd><Value value={value} /></dd></>
}

function Commentary({ candidate, review }: { candidate: Candidate | undefined; review: Review }) {
  const commentary = candidate?.commentary
  return (
    <section className="card" aria-labelledby="commentary">
      <h2 id="commentary">Commentary</h2>
      {review.declined && (
        <p>The designer found nothing to correct ({review.declined.reason}): {review.declined.message}</p>
      )}
      {candidate && reasons(candidate).length ? (
        <div className="problem">
          <p>The designer rejected this design:</p>
          <ul>{reasons(candidate).map((r) => <li key={r}>{r}</li>)}</ul>
        </div>
      ) : null}
      {commentary && Object.keys(commentary).length
        ? <dl className="facts">{Object.entries(commentary).map(([k, v]) => <Entry key={k} name={k} value={v} />)}</dl>
        : !review.declined && <p className="muted">None given.</p>}
      <p className="muted">Playback sent to the designer: {review.playback}.
        {review.designer && <> Designed by {review.designer}{review.designer_build ? ` (${review.designer_build})` : ''}.</>}</p>
    </section>
  )
}

function Metadata({ review }: { review: Review }) {
  const meta = review.metadata as Record<string, unknown>
  const shown = ['title', 'year', 'audio_types', 'genres', 'content_type'].filter((k) => meta[k] !== undefined && meta[k] !== '')
  return (
    <section className="card" aria-labelledby="metadata">
      <h2 id="metadata">Metadata</h2>
      {review.metadata_problems.length > 0 && (
        <div className="problem" role="note">
          <p>Not complete, so it cannot be accepted (fix it in the BEQDesigner app):</p>
          <ul>{review.metadata_problems.map((p) => <li key={p}>{p}</li>)}</ul>
        </div>
      )}
      <dl className="facts">
        {shown.map((k) => <Entry key={k} name={k} value={Array.isArray(meta[k]) ? (meta[k] as unknown[]).join(', ') : meta[k]} />)}
      </dl>
    </section>
  )
}

export function ReviewPage() {
  const { titleId = '' } = useParams()
  const { filter, query } = useListParams()
  const navigate = useNavigate()
  const review = useReview(titleId)
  const decide = useDecide(titleId)
  const next = useNextTitle()
  const [pick, setPick] = useState<{ digest: string; index: number } | null>(null)
  const [overriding, setOverriding] = useState(false)
  const [notice, setNotice] = useState('')
  const data = review.data
  const offered = data ? [...data.candidates, ...data.rejected] : []
  const picked = data && pick?.digest === data.digest ? pick.index : (data?.chosen_index ?? 0)
  const candidate = offered[picked]
  const chart = useChart(titleId, offered.length ? picked : null, !!data)
  const list = query ? `/titles?${query}` : '/titles'
  const link = (id: string) => `/titles/${encodeURIComponent(id)}${query ? `?${query}` : ''}`

  async function goNext(said: string) {
    const found = await next(filter, titleId).catch(() => null)
    if (found) {
      setNotice('')
      navigate(link(found.id))
    } else {
      setNotice(`${said}No other title in this list is waiting for a decision.`)
    }
  }

  function choose(index: number) {
    if (data && index >= 0 && index < offered.length) {
      setPick({ digest: data.digest, index })
      setOverriding(false)
    }
  }

  function accept(override = false) {
    if (!data || !candidate) return
    if (candidate.rejected && !override) {
      setOverriding(true)
      return
    }
    setOverriding(false)
    decide.mutate({ decision: 'accept', candidate: picked, digest: data.digest, override_rejection: override },
                  { onSuccess: () => void goNext('Accepted. ') })
  }

  function reject() {
    if (!data) return
    decide.mutate({ decision: 'reject', digest: data.digest, override_rejection: false },
                  { onSuccess: () => void goNext('Rejected. ') })
  }

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target instanceof Element ? event.target : null   // the window itself has no closest()
      const typing = target?.closest('input:not([type="radio"]):not([type="checkbox"]), select, textarea, [contenteditable]')
      if (event.ctrlKey || event.metaKey || event.altKey || typing) return
      if (!data || decide.isPending) return
      const key = event.key.toLowerCase()
      if (key === 'a' && !data.blocked.accept) accept()
      else if (key === 'r' && !data.blocked.reject) reject()
      else if (key === 's') void goNext('')
      else if (/^[1-9]$/.test(key)) choose(Number(key) - 1)
      else return
      event.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  if (review.isPending) return <p className="muted">Loading…</p>
  if (review.isError) {
    return (
      <section>
        <p><Link to={list}>← Titles</Link></p>
        <h1>{titleId}</h1>
        <p role="alert" className="problem">{review.error.message}</p>
        <p><Link to={`/jobs/new?ids=${encodeURIComponent(titleId)}`}>Run it through design…</Link></p>
      </section>
    )
  }
  const decided = data!.status !== 'pending' && data!.status !== 'skipped'
  return (
    <section className="review">
      <p><Link to={list}>← Titles</Link></p>
      <div className="title-row">
        <h1>{data!.year ? `${data!.title} (${data!.year})` : data!.title}</h1>
        <span className={`badge status-${data!.status}`}>{data!.status_text}</span>
      </div>
      <p className="muted">Needs {(NEEDS_WORDS[data!.needs as NeedsWord] ?? data!.needs).toLowerCase()}
        {data!.detail && data!.detail !== data!.needs ? `: ${data!.detail}` : ''}.
        {data!.in_flight && <strong> A run is working on this title.</strong>}</p>
      {notice && <p role="status">{notice}</p>}
      {decide.error && <p role="alert" className="problem">{decide.error.message}</p>}

      <div className="decide">
        <div className="actions">
          <button type="button" disabled={!!data!.blocked.accept || !candidate || decide.isPending}
                  title={data!.blocked.accept || 'A'} onClick={() => accept()}>Accept design {picked + 1}</button>
          <button type="button" className="secondary" disabled={!!data!.blocked.reject || decide.isPending}
                  title={data!.blocked.reject || 'R'} onClick={reject}>Reject</button>
          <button type="button" className="secondary" title="S" onClick={() => void goNext('')}>Skip</button>
        </div>
        {!decided && data!.blocked.accept && <p className="muted">Accept: {data!.blocked.accept}</p>}
        {!decided && data!.blocked.reject && <p className="muted">Reject: {data!.blocked.reject}</p>}
        {overriding && candidate && (
          <div className="confirm problem" role="group" aria-label="Override">
            <p>The designer rejected design {picked + 1} as unfit to publish:</p>
            <ul>{reasons(candidate).map((r) => <li key={r}>{r}</li>)}</ul>
            <p>Accept it anyway, as your override of the designer?</p>
            <div className="actions">
              <button type="button" className="danger" onClick={() => accept(true)}>Accept anyway</button>
              <button type="button" className="secondary" onClick={() => setOverriding(false)}>Back</button>
            </div>
          </div>
        )}
      </div>

      <div className="review-grid">
        <section className="card" aria-labelledby="designs">
          <h2 id="designs">Designs</h2>
          <ul className="candidates" role="radiogroup" aria-label="Designs">
            {data!.candidates.map((c) => (
              <li key={c.index}>
                <label className={c.index === picked ? 'on' : ''}>
                  <input type="radio" name="design" checked={c.index === picked} onChange={() => choose(c.index)} />
                  <CandidateLine candidate={c} />
                  {c.index === data!.chosen_index && <span className="badge status-accepted">chosen</span>}
                </label>
              </li>
            ))}
          </ul>
          {data!.rejected.length > 0 && (
            <>
              <h3>Rejected by the designer</h3>
              <ul className="candidates" aria-label="Rejected designs">
                {data!.rejected.map((c) => (
                  <li key={c.index}>
                    <label className={c.index === picked ? 'on' : ''}>
                      <input type="radio" name="design" checked={c.index === picked} onChange={() => choose(c.index)} />
                      <CandidateLine candidate={c} />
                      <span className="muted"> — {reasons(c).length} reason{reasons(c).length === 1 ? '' : 's'}</span>
                    </label>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
        <section className="card chart-card" aria-labelledby="chart">
          <h2 id="chart">Chart</h2>
          {chart.isError ? <p role="alert" className="problem">{chart.error.message}</p>
            : <MagnitudeChart series={chart.data?.series ?? []} />}
        </section>
        <Commentary candidate={candidate} review={data!} />
        <Metadata review={data!} />
      </div>
    </section>
  )
}
