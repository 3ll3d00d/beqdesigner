// How an execution ran (design/jobs-and-executions.md §2.3): its titles by outcome (each count opening the titles), the
// designers that designed for it with their counts and times, each stage's times and the slowest titles.
import { Link } from 'react-router'

import type { Schemas } from '../api/client'
import { useExecutionTitles } from '../api/hooks'
import { count, duration } from '../format'

type Statistics = Schemas['ExecutionStatistics']
type Figures = Schemas['StageFigures']

export const OUTCOME_WORDS: Record<string, string> = {
  designed: 'Designed', declined: 'Declined by the designer', extracted: 'Extracted', already_done: 'Already done',
  failed: 'Failed', unavailable: 'Unavailable', skipped: 'Skipped', not_reached: 'Not reached',
}

const STAGE_WORDS: Record<string, string> = {
  extract: 'Extraction', waiting: 'Waiting for the designer', designer: 'At the designer', writing: 'Writing the result',
}

const seconds = (value: number | undefined) => (value === undefined ? '—' : duration(value))

function FiguresCells({ figures }: { figures: Figures | null | undefined }) {
  if (!figures) return <><td>—</td><td>—</td><td>—</td><td>—</td></>
  return <><td className="num">{seconds(figures.mean)}</td><td className="num">{seconds(figures.p50)}</td>
    <td className="num">{seconds(figures.p90)}</td><td className="num">{seconds(figures.max)}</td></>
}

/** The titles by outcome, each count opening the titles it counts (asked for once one is opened). */
function Outcomes({ executionId, statistics, live }: { executionId: string; statistics: Statistics; live: boolean }) {
  const titles = useExecutionTitles(executionId, true)
  const outcomes = Object.entries(statistics.outcomes)
  if (!outcomes.length) return <p className="muted">No titles yet.</p>
  return (
    <ul className="outcomes" aria-label="Outcomes">
      {outcomes.map(([outcome, n]) => {
        const these = (titles.data ?? []).filter((t) => t.outcome === outcome)
        return (
          <li key={outcome}>
            <details>
              <summary>{OUTCOME_WORDS[outcome] ?? outcome}: {n.toLocaleString()}{outcome === 'not_reached' && live ? ' (yet)' : ''}</summary>
              <ul className="titles">
                {these.map((t) => (
                  <li key={t.id}><Link to={`/titles/${encodeURIComponent(t.id)}`}>{t.title || t.id}</Link>
                    {t.reason && <span className="muted"> — {t.reason}</span>}</li>
                ))}
              </ul>
            </details>
          </li>
        )
      })}
    </ul>
  )
}

export function ExecutionStatistics({ executionId, statistics, live }:
                                    { executionId: string; statistics: Statistics; live: boolean }) {
  const stages = Object.entries(statistics.stages)
  return (
    <section aria-label="How it ran">
      <h2>How it ran</h2>
      <p>{statistics.wall_seconds != null ? <>Took {duration(statistics.wall_seconds)}</> : 'Not started'}
        {statistics.per_hour != null && <>, {statistics.per_hour} titles an hour</>}.</p>
      <Outcomes executionId={executionId} statistics={statistics} live={live} />
      {statistics.failures.length > 0 && (
        <table className="list" aria-label="Why titles failed">
          <thead><tr><th>Why titles failed</th><th className="num">Titles</th></tr></thead>
          <tbody>{statistics.failures.map((f) => <tr key={f.reason}><td>{f.reason}</td><td className="num">{f.count}</td></tr>)}</tbody>
        </table>
      )}
      {statistics.skips.length > 0 && (
        <table className="list" aria-label="Why titles were skipped">
          <thead><tr><th>Why titles were skipped</th><th className="num">Titles</th></tr></thead>
          <tbody>{statistics.skips.map((f) => <tr key={f.reason}><td>{f.reason}</td><td className="num">{f.count}</td></tr>)}</tbody>
        </table>
      )}
      {statistics.designers.length > 0 && <>
        {statistics.designers_differ && <p role="alert" className="problem">Its designs came from more than one release or
          design version.</p>}
        <table className="list" aria-label="Designers">
          <thead><tr><th>Designer</th><th>Release</th><th className="num">Version</th><th className="num">Designed</th>
            <th className="num">Declined</th><th className="num">Failed</th><th className="num">Mean</th>
            <th className="num">Median</th><th className="num">p90</th><th className="num">Max</th></tr></thead>
          <tbody>
            {statistics.designers.map((d) => (
              <tr key={`${d.endpoint} ${d.release} ${d.version}`}>
                <td>{d.endpoint}</td><td>{d.release || '—'}</td><td className="num">{d.version || '—'}</td>
                <td className="num">{d.designed}</td><td className="num">{d.declined}</td><td className="num">{d.failed}</td>
                <FiguresCells figures={d.seconds} />
              </tr>
            ))}
          </tbody>
        </table>
      </>}
      {stages.length > 0 && (
        <table className="list" aria-label="Stage times">
          <thead><tr><th>Each title</th><th className="num">Titles</th><th className="num">Mean</th><th className="num">Median</th>
            <th className="num">p90</th><th className="num">Max</th></tr></thead>
          <tbody>
            {stages.map(([stage, figures]) => (
              <tr key={stage}><td>{STAGE_WORDS[stage] ?? stage}</td><td className="num">{figures.count}</td>
                <FiguresCells figures={figures} /></tr>
            ))}
          </tbody>
        </table>
      )}
      {statistics.slowest.length > 0 && (
        <details>
          <summary>Slowest {count(statistics.slowest.length, 'title')}</summary>
          <ul className="titles">
            {statistics.slowest.map((t) => (
              <li key={t.id}><Link to={`/titles/${encodeURIComponent(t.id)}`}>{t.title || t.id}</Link>
                <span className="muted"> — {[t.extract_seconds != null && `extract ${duration(t.extract_seconds)}`,
                  t.waiting_seconds != null && `waiting ${duration(t.waiting_seconds)}`,
                  t.designer_seconds != null && `at the designer ${duration(t.designer_seconds)}`,
                  t.endpoint && `on ${t.endpoint}`].filter(Boolean).join(', ')}</span></li>
            ))}
          </ul>
        </details>
      )}
    </section>
  )
}
