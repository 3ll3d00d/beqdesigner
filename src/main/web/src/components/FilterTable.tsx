// A design's filters as a table -- type, frequency, Q, gain, how many stacked -- read-only: under the highlighted design on
// the review page, and for each published BEQ on its tab, so what a BEQ is can be read as well as seen on the chart.
import { TYPE_NAMES, type FilterRow } from '../chart/biquad'

interface Props {
  rows: FilterRow[]
  label: string
}

const fixed = (value: number, digits: number) => (Number.isInteger(value) ? String(value) : value.toFixed(digits))

export function FilterTable({ rows, label }: Props) {
  if (!rows.length) return <p className="muted">No filters.</p>
  return (
    <table className="list filter-table" aria-label={label}>
      <thead><tr><th>Type</th><th>Frequency (Hz)</th><th>Q</th><th>Gain (dB)</th><th>Stacked</th></tr></thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i}>
            <td>{TYPE_NAMES[r.type]}</td>
            <td>{fixed(r.freq_hz, 1)}</td>
            <td>{fixed(r.q, 3)}</td>
            <td>{r.gain_db > 0 ? '+' : ''}{fixed(r.gain_db, 1)}</td>
            <td>{r.count > 1 ? `×${r.count}` : ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
