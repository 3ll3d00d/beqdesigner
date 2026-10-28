// Deciding one title (design/web-app.md §4, design/review-over-http.md): its designs -- every kept design run's, in one
// list to pick from, the current run's first -- the chart of the one highlighted (and of any others ticked to compare with
// it, from any run), commentary, the metadata (one line under
// the heading) and its problems, and Accept / Reject / Skip (A, R, S; 1-9 highlight a design). Accepting a design the
// designer rejected asks first; accepting one of an earlier run makes that run current. After a decision, or Skip, the
// next title waiting in the same filtered list opens.
import { useEffect, useState, type ReactNode } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import type { Schemas } from '../api/client'
import { useChart, useDecide, useNextTitle, useProjects, useReview, useStatus, useVariantCharts } from '../api/hooks'
import { rowsOf } from '../chart/biquad'
import { DEFAULT_VIEW, loadView, saveView, type ChartView, type Range, type Series } from '../chart/view'
import { ChartControls } from '../components/ChartControls'
import { HeatmapView } from '../components/HeatmapView'
import { MagnitudeChart } from '../components/MagnitudeChart'
import { MetadataPanel } from '../components/MetadataPanel'
import { ProjectsLine } from '../components/ProjectsLine'
import { PublishedBeqs } from '../components/PublishedBeqs'
import { ReviseDialog, reviseSummary } from '../components/ReviseDialog'
import { Spectrogram } from '../components/Spectrogram'
import { StreamPicker } from '../components/StreamPicker'
import { FilterEditor } from '../components/FilterEditor'
import { FilterTable } from '../components/FilterTable'
import { VariantsPanel } from '../components/VariantsPanel'
import type { Ask } from '../components/AdjustmentsEditor'
import { metadataText, NEEDS_WORDS, type NeedsWord } from '../format'
import { useListParams } from './TitlesPage'

type Candidate = Schemas['CandidateView']
type Review = Schemas['Review']

const reasons = (candidate: Candidate): string[] => candidate.rejection_reasons ?? []

/** One design run's designs to pick from: the current run (`run` null) or an earlier one kept. */
interface Run {
  run: string | null
  label: string
  detail: string
  candidates: Candidate[]
  rejected: Candidate[]
}

/** A design in the one list: its run, its index in that run, and its number in the list (what 1-9 and Accept say). */
interface Offer {
  run: string | null
  candidate: Candidate
  number: number
}

const designedWhen = (at?: string | null) => (at ? new Date(at).toLocaleString() : 'when not recorded')

/** The title's design runs, the current one first and then the earlier ones newest first. */
export function runsOf(review: Review): Run[] {
  const describe = (at: string | null | undefined, build: string | null | undefined, declined: boolean) =>
    [designedWhen(at), build, declined ? 'declined' : ''].filter(Boolean).join(' · ')
  return [
    { run: null, label: `${review.variant_id ?? 'This run'} (current)`, candidates: review.candidates, rejected: review.rejected,
      detail: describe(review.designed_at, review.designer_build, !!review.declined) },
    ...[...review.variants].reverse().map((v) => ({
      run: v.variant_id, label: v.variant_id, candidates: v.candidates ?? [], rejected: v.rejected ?? [],
      detail: describe(v.designed_at, v.designer_build, v.declined) })),
  ]
}

/**
 * The series of one run's chart with each design's `candidate` its number in the one list less one, so designs of different
 * runs drawn together take colours of their own (and the highlighted one is found by its number).
 */
export function numbered(series: Series[], run: string | null, offers: Offer[]): Series[] {
  return series.map((s) => {
    if (s.candidate === null || s.candidate === undefined) return s
    const o = offers.find((f) => f.run === run && f.candidate.index === s.candidate)
    return o ? { ...s, candidate: o.number - 1 } : s
  })
}

/** Every run's designs as one list, numbered: each run's candidates, then the ones its designer rejected. */
export function offersOf(runs: Run[]): Offer[] {
  let number = 0
  return runs.flatMap((r) => [...r.candidates, ...r.rejected].map((candidate) => ({ run: r.run, candidate, number: ++number })))
}

function number(value: number | null | undefined, unit: string, digits = 1): string {
  return value === null || value === undefined ? '—' : `${value.toFixed(digits)}${unit}`
}

function CandidateLine({ candidate, position }: { candidate: Candidate; position: number }) {
  return (
    <span>
      <strong>{position}.</strong> {candidate.method}
      {candidate.confidence !== null && <> · confidence {candidate.confidence.toFixed(2)}</>}
      {' '}· MV +{number(candidate.mv_adjust_db, ' dB')}
      {candidate.gain_reduction_db !== null && candidate.gain_reduction_db !== undefined &&
        <> · gain reduction {number(candidate.gain_reduction_db, ' dB')}</>}
      {candidate.residual_db !== null && candidate.residual_db !== undefined && <> · residual {number(candidate.residual_db, ' dB')}</>}
    </span>
  )
}

/** The highlighted design's filters, read-only (editing them is *Edit filters…*). */
function DesignFilters({ candidate, number }: { candidate: Candidate; number: number }) {
  const stored = rowsOf(candidate.filters)
  return (
    <div className="design-filters">
      <h3>Design {number}'s filters</h3>
      {stored ? <FilterTable rows={stored.rows} label={`Design ${number}'s filters`} />
        : <p className="muted">It has a filter this page cannot show.</p>}
    </div>
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
  const [pick, setPick] = useState<{ digest: string; number: number } | null>(null)
  // the designs ticked to draw beside the highlighted one, by number, for what was read
  const [compare, setCompare] = useState<{ digest: string; numbers: number[] } | null>(null)
  const [overriding, setOverriding] = useState(false)
  const [notice, setNotice] = useState('')
  const data = review.data
  const runs = data ? runsOf(data) : []
  const offers = offersOf(runs)
  const chosen = offers.find((o) => o.run === null && o.candidate.index === (data?.chosen_index ?? 0)) ?? offers[0]
  const offer = (data && pick?.digest === data.digest ? offers.find((o) => o.number === pick.number) : undefined) ?? chosen
  const candidate = offer?.candidate
  const picked = candidate?.index ?? 0   // into its own run's designs
  const pickedRun = offer?.run ?? null
  const runOffers = offers.filter((o) => o.run === pickedRun).length
  const [view, setView] = useState<ChartView>(loadView)
  const [visual, setVisual] = useState<Visual>('magnitude')
  const projects = useProjects(titleId, data?.digest)
  // while *Revise…* waits for a band to be dragged across the chart: what to do with it; and the bands it asks to ignore
  const [pickBand, setPickBand] = useState<{ take: (range: Range) => void } | null>(null)
  const [bands, setBands] = useState<Range[]>([])
  const onPick = (take: ((range: Range) => void) | null) => {
    setPickBand(take ? { take } : null)
    if (take) setVisual('magnitude')
  }
  // drawn: the highlighted design and those ticked to compare with it, or, with *All designs*, every one of every run
  const compared = data && compare?.digest === data.digest ? compare.numbers : []
  const drawnOffers = view.allDesigns ? offers : offers.filter((o) => o === offer || compared.includes(o.number))
  const designsOf = (run: string | null) => drawnOffers.filter((o) => o.run === run).map((o) => o.candidate.index)
  const otherRuns = runs.filter((r) => r.run !== pickedRun && designsOf(r.run).length > 0)
    .map((r) => ({ run: r.run, variant: r.run ?? data?.variant_id ?? 'v1', candidates: designsOf(r.run) }))
  const variantCharts = useVariantCharts(titleId, otherRuns.map((r) => (
    { variant: r.variant, candidates: view.allDesigns ? 'all' as const : r.candidates })), view.smoothing, view.responses)
  const chart = useChart(titleId, { candidate: offers.length ? picked : null, all: view.allDesigns && runOffers > 1,
                                    candidates: [picked, ...designsOf(pickedRun).filter((i) => i !== picked)],
                                    smoothing: view.smoothing, responses: view.responses, variant: pickedRun }, !!data)
  const magnitude = [...numbered(chart.data?.series ?? [], pickedRun, offers),
                     ...variantCharts.flatMap((q, i) => numbered((q.data?.series ?? []).filter((s) => s.filtered),
                                                                 otherRuns[i]!.run, offers))]
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

  function choose(number: number) {
    if (data && offers.some((o) => o.number === number)) {
      // while comparing, the design highlighted until now stays on the chart
      if (offer && number !== offer.number && compared.some((n) => n !== offer.number) && !compared.includes(offer.number)) {
        setCompare({ digest: data.digest, numbers: [...compared, offer.number] })
      }
      setPick({ digest: data.digest, number })
      setOverriding(false)
    }
  }

  function toggleCompare(number: number) {
    if (!data) return
    setCompare({ digest: data.digest,
                 numbers: compared.includes(number) ? compared.filter((n) => n !== number) : [...compared, number] })
    setVisual('magnitude')   // the only view that draws several
  }

  function accept(override = false) {
    if (!data || !candidate) return
    if (candidate.rejected && !override) {
      setOverriding(true)
      return
    }
    setOverriding(false)
    decide.mutate({ decision: 'accept', candidate: picked, digest: data.digest, override_rejection: override,
                    ...(pickedRun ? { variant: pickedRun } : {}) },
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
      else if (/^[1-9]$/.test(key)) choose(Number(key))
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
      <ProjectsLine projects={projects.data} />
      <div className="actions">
        <button type="button" className="secondary" disabled={!!reviseBlocked || decide.isPending}
                title={reviseBlocked || undefined} onClick={() => setRevising(true)}>Revise…</button>
        <button type="button" className="secondary" disabled={!candidate || settled || data!.in_flight || pickedRun !== null}
                title={settled ? 'Reopen the title for review before editing a design'
                  : pickedRun !== null ? 'Edit a design of the current run (use this run first to edit one of its designs)'
                    : undefined}
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
                  title={data!.blocked.accept || (pickedRun ? `A — makes ${pickedRun} the current run` : 'A')}
                  onClick={() => accept()}>Accept design {offer?.number ?? 1}{pickedRun ? ` (${pickedRun})` : ''}</button>
          <button type="button" className="secondary" disabled={!!data!.blocked.reject || decide.isPending}
                  title={data!.blocked.reject || 'R'} onClick={reject}>Reject</button>
          <button type="button" className="secondary" title="S" onClick={() => void goNext('')}>Skip</button>
        </div>
        {!decided && data!.blocked.accept && <p className="muted">Accept: {data!.blocked.accept}</p>}
        {!decided && data!.blocked.reject && <p className="muted">Reject: {data!.blocked.reject}</p>}
        {overriding && candidate && (
          <div className="confirm problem" role="group" aria-label="Override">
            <p>The designer rejected design {offer?.number ?? 1} as unfit to publish:</p>
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
          {!view.allDesigns && compared.some((n) => n !== offer?.number) && (
            <p className="muted">Comparing {drawnOffers.length} designs on the chart.{' '}
              <button type="button" className="secondary small" onClick={() => setCompare(null)}>Stop comparing</button></p>
          )}
          {runs.map((r) => {
            const of = (c: Candidate) => offers.find((o) => o.run === r.run && o.candidate.index === c.index)!
            const named = r.run === null ? '' : ` of ${r.run}`
            const line = (c: Candidate, rejected: boolean) => {
              const o = of(c)
              return (
                <li key={c.index}>
                  <label className={o.number === offer?.number ? 'on' : ''}>
                    <input type="radio" name="design" checked={o.number === offer?.number} onChange={() => choose(o.number)} />
                    <CandidateLine candidate={c} position={o.number} />
                    {rejected && <span className="muted"> — {reasons(c).length} reason{reasons(c).length === 1 ? '' : 's'}</span>}
                    {!rejected && r.run === null && c.index === data!.chosen_index &&
                      <span className="badge status-accepted">chosen</span>}
                  </label>
                  {offers.length > 1 && (
                    <label className="compare muted"
                           title={view.allDesigns ? 'All designs are drawn' : o.number === offer?.number
                             ? 'The highlighted design is always drawn' : 'Draw it on the chart with the highlighted design'}>
                      <input type="checkbox" aria-label={`Compare design ${o.number}`}
                             checked={drawnOffers.includes(o)} disabled={view.allDesigns || o.number === offer?.number}
                             onChange={() => toggleCompare(o.number)} /> compare
                    </label>
                  )}
                </li>
              )
            }
            return (
              <div key={r.run ?? 'current'} className="run">
                {runs.length > 1 && <h3>{r.label} <span className="muted">· {r.detail}</span></h3>}
                {r.candidates.length > 0 && (
                  <ul className="candidates" role="radiogroup" aria-label={`Designs${named}`}>
                    {r.candidates.map((c) => line(c, false))}
                  </ul>
                )}
                {r.rejected.length > 0 && (
                  <>
                    <h4>Rejected by the designer</h4>
                    <ul className="candidates" aria-label={`Rejected designs${named}`}>{r.rejected.map((c) => line(c, true))}</ul>
                  </>
                )}
                {!r.candidates.length && !r.rejected.length && <p className="muted">No designs in this run.</p>}
              </div>
            )
          })}
          {candidate && <DesignFilters candidate={candidate} number={offer!.number} />}
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
                : <MagnitudeChart series={magnitude} view={view} highlighted={offer ? offer.number - 1 : null} bands={bands}
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
          {visual === 'spectrogram' && (offers.length
            ? <Spectrogram titleId={titleId} candidate={picked} variant={pickedRun} view={view}
                           onZoom={(x) => changeView({ ...view, x })}
                           onReset={() => changeView({ ...view, x: DEFAULT_VIEW.x })} />
            : <p className="muted">No design yet: nothing to draw.</p>)}
          {visual === 'published' && (
            <PublishedBeqs titleId={titleId} ours={chart.data?.series ?? []} picked={picked} view={view} onView={changeView} />
          )}
          {visual === 'heatmap' && (offers.length
            ? <HeatmapView titleId={titleId} candidate={picked} variant={pickedRun} title={data!.title} />
            : <p className="muted">No design yet: nothing to draw.</p>)}
        </section>
        <Commentary candidate={candidate} review={data!} />
      </div>
      {editing && editing.title === titleId && (
        <FilterEditor key={`${data!.digest}-${editing.source}`} review={data!} source={editing.source}
                      projectEdited={!!projects.data?.projects.some((p) => p.edited)}
                      series={chart.data?.series ?? []} view={view} onView={changeView} onClose={() => setEditing(null)} />
      )}
      <VariantsPanel review={data!} />
      <MetadataPanel review={data!} />
    </section>
  )
}
