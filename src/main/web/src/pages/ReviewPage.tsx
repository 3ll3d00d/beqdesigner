// Deciding one title (design/web-app.md §4, design/review-over-http.md): its designs, the chart of the one highlighted,
// commentary, the metadata (one line under the heading) and its problems, and Accept / Reject / Skip (A, R, S; 1-9 highlight a design). Accepting a design the
// designer rejected asks first. After a decision, or Skip, the next title waiting in the same filtered list opens.
import { useEffect, useState, type ReactNode } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import type { Schemas } from '../api/client'
import { useChart, useDecide, useNextTitle, useReview, useStatus, useVariantCharts } from '../api/hooks'
import { DEFAULT_VIEW, loadView, saveView, type ChartView, type Range } from '../chart/view'
import { ChartControls } from '../components/ChartControls'
import { HeatmapView } from '../components/HeatmapView'
import { MagnitudeChart } from '../components/MagnitudeChart'
import { MetadataPanel } from '../components/MetadataPanel'
import { PublishedBeqs } from '../components/PublishedBeqs'
import { ReviseDialog, reviseSummary } from '../components/ReviseDialog'
import { Spectrogram } from '../components/Spectrogram'
import { StreamPicker } from '../components/StreamPicker'
import { FilterEditor } from '../components/FilterEditor'
import { VariantsPanel } from '../components/VariantsPanel'
import type { Ask } from '../components/AdjustmentsEditor'
import { metadataText, NEEDS_WORDS, type NeedsWord } from '../format'
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
    <section className="card commentary" aria-labelledby="commentary">
      <h2 id="commentary">Commentary</h2>
      {review.declined && (
        <p>The designer found nothing to correct ({review.declined.reason}): {review.declined.message}</p>
      )}
      {candidate?.fit && (
        <p className="fit"><strong>Fit</strong>: {candidate.fit}
          <span className="muted"> — how well its filters match the target it was fitted to, not whether that target is right</span></p>
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

/** The title's metadata as one line under its heading (the Metadata panel, below, edits it and says what is missing). */
function Metadata({ review }: { review: Review }) {
  const meta = review.metadata as Record<string, unknown>
  const parts = ['content_type', 'audio_types', 'genres'].map((k) => metadataText(meta[k])).filter(Boolean)
  return (
    <>
      {parts.length > 0 && <p className="metadata" aria-label="Metadata">{parts.join(' · ')}</p>}
      {review.metadata_problems.length > 0 && (
        <p className="problem" role="note">
          Metadata not complete, so it cannot be accepted: <a href="#metadata-heading">fix it in the Metadata panel</a>.
        </p>
      )}
    </>
  )
}

type Visual = 'magnitude' | 'spectrogram' | 'heatmap' | 'published'
const VISUALS: { id: Visual; label: string }[] = [
  { id: 'magnitude', label: 'Magnitude' }, { id: 'spectrogram', label: 'Spectrogram' }, { id: 'heatmap', label: 'Heatmap' },
  { id: 'published', label: 'Published BEQs' },
]

export function ReviewPage() {
  const { titleId = '' } = useParams()
  const { filter, query } = useListParams()
  const navigate = useNavigate()
  const review = useReview(titleId)
  const decide = useDecide(titleId)
  const next = useNextTitle()
  const status = useStatus()
  const [revising, setRevising] = useState(false)
  const [editing, setEditing] = useState<{ title: string; source: number } | null>(null)
  const [pick, setPick] = useState<{ digest: string; index: number } | null>(null)
  const [overriding, setOverriding] = useState(false)
  const [notice, setNotice] = useState('')
  const data = review.data
  const offered = data ? [...data.candidates, ...data.rejected] : []
  const picked = data && pick?.digest === data.digest ? pick.index : (data?.chosen_index ?? 0)
  const candidate = offered[picked]
  const [view, setView] = useState<ChartView>(loadView)
  const [visual, setVisual] = useState<Visual>('magnitude')
  // while *Revise…* waits for a band to be dragged across the chart: what to do with it; and the bands it asks to ignore
  const [pickBand, setPickBand] = useState<{ take: (range: Range) => void } | null>(null)
  const [bands, setBands] = useState<Range[]>([])
  const onPick = (take: ((range: Range) => void) | null) => {
    setPickBand(take ? { take } : null)
    if (take) setVisual('magnitude')
  }
  const [overlaid, setOverlaid] = useState<{ title: string; variants: string[] }>({ title: titleId, variants: [] })
  const shownVariants = overlaid.title === titleId ? overlaid.variants : []
  const variantCharts = useVariantCharts(titleId, shownVariants, view.smoothing)
  const chart = useChart(titleId, { candidate: offered.length ? picked : null, all: view.allDesigns && offered.length > 1,
                                    smoothing: view.smoothing, responses: view.responses }, !!data)
  // the other design runs shown: their curves after each of their designs, drawn dashed under the current one's
  const magnitude = [...(chart.data?.series ?? []),
                     ...variantCharts.flatMap((q) => (q.data?.series ?? []).filter((s) => s.filtered && s.kind !== 'filter'))]
  const changeView = (next: ChartView) => {
    setView(next)
    saveView(next)
  }
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
      if (!data || decide.isPending || revising) return
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
  const reviseBlocked = data!.in_flight ? 'A run is working on this title.'
    : data!.status === 'published' && !status.data?.repository_writes
      ? 'Sending a published title back requires repository writes to be enabled.' : ''
  const decided = data!.status !== 'pending' && data!.status !== 'skipped'
  const settled = data!.status === 'accepted' || data!.status === 'published'   // its catalogue files stand for this design
  return (
    <section className="review">
      <p><Link to={list}>← Titles</Link></p>
      <div className="title-row">
        <h1>{data!.year ? `${data!.title} (${data!.year})` : data!.title}</h1>
        <span className={`badge status-${data!.status}`}>{data!.status_text}</span>
      </div>
      <Metadata review={data!} />
      <p className="muted">Needs {(NEEDS_WORDS[data!.needs as NeedsWord] ?? data!.needs).toLowerCase()}
        {data!.detail && data!.detail !== data!.needs ? `: ${data!.detail}` : ''}.
        {data!.in_flight && <strong> A run is working on this title.</strong>}</p>
      <div className="actions">
        <button type="button" className="secondary" disabled={!!reviseBlocked || decide.isPending}
                title={reviseBlocked || undefined} onClick={() => setRevising(true)}>Revise…</button>
        <button type="button" className="secondary" disabled={!candidate || settled || data!.in_flight}
                title={settled ? 'Reopen the title for review before editing a design' : undefined}
                onClick={() => setEditing({ title: titleId, source: picked })}>Edit filters…</button>
        <StreamPicker titleId={titleId} blocked={data!.in_flight ? 'A run is working on this title.' : reviseBlocked} />
      </div>
      {reviseBlocked && <p className="muted">{reviseBlocked}</p>}
      {revising && (
        <ReviseDialog ids={[titleId]} name={data!.title} published={data!.status === 'published'}
                      asked={data!.adjustments as Ask[]} chartRange={view.x}
                      onPick={onPick} picking={!!pickBand} onBands={setBands}
                      onClose={() => setRevising(false)}
                      onDone={(outcome) => {
                        setRevising(false)
                        setNotice(reviseSummary(outcome))
                      }} />
      )}
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
          <h2 id="chart" className="visually-hidden">Chart</h2>
          <div className="tabs" role="tablist" aria-label="How to look at it">
            {VISUALS.map((v) => (
              <button key={v.id} type="button" role="tab" aria-selected={visual === v.id}
                      className={`tab${visual === v.id ? ' on' : ''}`} onClick={() => setVisual(v.id)}>{v.label}</button>
            ))}
          </div>
          {visual === 'magnitude' && (
            <>
              <ChartControls view={view} series={magnitude} onChange={changeView} />
              {pickBand && <p className="notice" role="status">Drag across the chart to choose the band the redesign
                ignores.</p>}
              {chart.isError ? <p role="alert" className="problem">{chart.error.message}</p>
                : <MagnitudeChart series={magnitude} view={view} highlighted={picked} bands={bands}
                                  onZoom={(x, y) => {
                                    if (pickBand) {
                                      pickBand.take(x)
                                      setPickBand(null)
                                    } else {
                                      changeView({ ...view, x, y })
                                    }
                                  }}
                                  onReset={() => changeView({ ...view, x: DEFAULT_VIEW.x, y: null })} />}
            </>
          )}
          {visual === 'spectrogram' && (offered.length
            ? <Spectrogram titleId={titleId} candidate={picked} view={view} onZoom={(x) => changeView({ ...view, x })}
                           onReset={() => changeView({ ...view, x: DEFAULT_VIEW.x })} />
            : <p className="muted">No design yet: nothing to draw.</p>)}
          {visual === 'published' && (
            <PublishedBeqs titleId={titleId} ours={chart.data?.series ?? []} picked={picked} view={view} onView={changeView} />
          )}
          {visual === 'heatmap' && (offered.length
            ? <HeatmapView titleId={titleId} candidate={picked} title={data!.title} />
            : <p className="muted">No design yet: nothing to draw.</p>)}
        </section>
        <Commentary candidate={candidate} review={data!} />
      </div>
      {editing && editing.title === titleId && (
        <FilterEditor key={`${data!.digest}-${editing.source}`} review={data!} source={editing.source}
                      series={chart.data?.series ?? []} view={view} onView={changeView} onClose={() => setEditing(null)} />
      )}
      <VariantsPanel review={data!} shown={shownVariants}
                     onShow={(variants) => setOverlaid({ title: titleId, variants })} />
      <MetadataPanel review={data!} />
    </section>
  )
}
