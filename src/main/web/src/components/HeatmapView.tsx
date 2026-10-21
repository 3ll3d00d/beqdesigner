// The publication heatmap of a design (design/web-parity.md WP2): exactly the image publish would write -- the filtered
// track on the left, as measured on the right -- rendered by the service once and kept.
import { useHeatmapUrl } from '../api/hooks'

export function HeatmapView({ titleId, candidate, title }: { titleId: string; candidate: number; title: string }) {
  const heatmap = useHeatmapUrl(titleId, candidate, true)
  if (heatmap.error) return <p role="alert" className="problem">{heatmap.error.message}</p>
  if (!heatmap.url) {
    return <p className="muted" role="status">Drawing the heatmap… (the service renders it once, then keeps it)</p>
  }
  return <img className="heatmap" src={heatmap.url} alt={`The publication heatmap of ${title}, design ${candidate + 1}`} />
}
