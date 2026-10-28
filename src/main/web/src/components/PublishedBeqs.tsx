// Published BEQs for the title (design/web-parity.md WP3): what the catalogue has for it, read on demand by the service,
// and any of them on the workbench beside our design -- their response, and our track under their BEQ -- with how far
// apart the two are. The measured curves and our highlighted design come from the chart route, as on the Magnitude tab.
import { useState } from 'react'

import type { Schemas } from '../api/client'
import { useCatalogue, usePublishedCharts } from '../api/hooks'
import { resample, type ChartView, type Range, type Series } from '../chart/view'
import { ChartControls } from './ChartControls'
import { FilterTable } from './FilterTable'
import { CATALOGUE_BASE, MagnitudeChart } from './MagnitudeChart'

const COMPARED: Range = [5, 40]

/** Their BEQ's peak boost and where; and the RMS of (ours − theirs) on the filtered average over 5-40 Hz, when both are there. */
export function comparison(theirs: Series[], ours: Series | undefined) {
  const response = theirs.find((s) => s.kind === 'filter')
  let peak: { db: number; hz: number } | null = null
  if (response) {
    response.y.forEach((db, i) => {
      if (!peak || db > peak.db) peak = { db, hz: response.x[i]! }
    })
  }
  const under = theirs.find((s) => s.kind === 'average')
  let rms: number | null = null
  if (ours && under) {
    const at = ours.x.filter((hz) => hz >= COMPARED[0] && hz <= COMPARED[1])
    if (at.length) {
      const a = resample(ours.x, ours.y, at), b = resample(under.x, under.y, at)
      rms = Math.sqrt(a.reduce((sum, v, i) => sum + (v - b[i]!) ** 2, 0) / at.length)
    }
  }
  return { peak: peak as { db: number; hz: number } | null, rms }
}

interface Props {
  titleId: string
  ours: Series[]
  picked: number
  view: ChartView
  onView: (view: ChartView) => void
}

export function PublishedBeqs({ titleId, ours, picked, view, onView }: Props) {
  const catalogue = useCatalogue(titleId, true)
  const [others, setOthers] = useState(false)
  const [chosen, setChosen] = useState<number[]>([])
  const entries = catalogue.data?.entries ?? []
  const anySame = entries.some((e) => e.same_track)
  const listed = others || !anySame ? entries : entries.filter((e) => e.same_track)
  const charts = usePublishedCharts(titleId, chosen, view.smoothing)
  const theirs = (index: number): Series[] => {
    const at = chosen.indexOf(index)
    const entry = entries.find((e) => e.index === index)
    return (charts[at]?.data?.series ?? []).map((s) => ({
      ...s, candidate: CATALOGUE_BASE + index,
      name: s.kind === 'filter' ? s.name : `${s.name.split(' under ')[0]} under ${entry?.author ?? 'their'}'s BEQ`,
    }))
  }
  const ourAverage = ours.find((s) => s.kind === 'average' && s.filtered && s.candidate === picked)
  const series = [...ours, ...chosen.flatMap(theirs)]
  const toggle = (index: number, on: boolean) => setChosen((was) => (on ? [...was, index] : was.filter((i) => i !== index)))

  if (catalogue.isPending) return <p className="muted" role="status">Asking the catalogue…</p>
  if (catalogue.isError) return <p role="alert" className="problem">{catalogue.error.message}</p>
  const data = catalogue.data!
  return (
    <div className="published">
      {data.message && <p className="muted">{data.message}</p>}
      {entries.length > 0 && (
        <>
          <p className="muted">
            Matched by {data.matched_by === 'tmdb' ? 'TMDB id' : 'title and year'}
            {data.compare_url && <> · <a href={data.compare_url} target="_blank" rel="noreferrer">Compare every author in the catalogue</a></>}
          </p>
          <label className="check">
            <input type="checkbox" checked={others} onChange={(e) => setOthers(e.target.checked)} disabled={!anySame} />
            Include other tracks and editions
          </label>
          {!anySame && <p className="muted">None is for this title's track, as far as is known: every one is listed.</p>}
          <ul className="published-list" aria-label="Published BEQs">
            {listed.map((e: Schemas['PublishedBeq']) => {
              const at = chosen.indexOf(e.index)
              const summary = at >= 0 && charts[at]?.data ? comparison(theirs(e.index), ourAverage) : null
              return (
                <li key={e.index}>
                  <label className="check">
                    <input type="checkbox" checked={at >= 0} onChange={(ev) => toggle(e.index, ev.target.checked)} />
                    <strong>{e.author}</strong> · {e.audio_types.join(', ') || 'audio not said'}
                    {e.edition && ` · ${e.edition}`}{e.season && ` · season ${e.season}`}{e.episodes && ` · episodes ${e.episodes}`}
                    {!e.same_track && <span className="muted"> (another track)</span>}
                  </label>
                  {e.catalogue_url && <a href={e.catalogue_url} target="_blank" rel="noreferrer">page</a>}
                  {charts[at]?.isError && <span className="problem"> {charts[at]!.error!.message}</span>}
                  <details className="published-filters">
                    <summary>{e.filter_count} filter{e.filter_count === 1 ? '' : 's'}</summary>
                    <FilterTable rows={e.filters ?? []} label={`${e.author}'s filters`} />
                  </details>
                  {summary && (
                    <span className="muted" aria-label={`How ${e.author}'s BEQ compares`}>
                      {summary.peak && ` · boosts up to ${summary.peak.db.toFixed(1)} dB at ${summary.peak.hz.toFixed(1)} Hz`}
                      {summary.rms !== null && ` · ${summary.rms.toFixed(1)} dB RMS from ours over ${COMPARED[0]}-${COMPARED[1]} Hz`}
                    </span>
                  )}
                </li>
              )
            })}
          </ul>
        </>
      )}
      <ChartControls view={view} series={series} onChange={onView} />
      <MagnitudeChart series={series} view={view} highlighted={picked}
                      onZoom={(x, y) => onView({ ...view, x, y })} onReset={() => onView({ ...view, x: [1, 160], y: null })} />
    </div>
  )
}
