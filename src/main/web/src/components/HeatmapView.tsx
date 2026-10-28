// The heatmap of a design (design/web-parity.md WP2): the publication heatmap's two panes -- after the design on the
// left, as measured on the right, 1-40 Hz across and time down -- drawn here from the spectrogram's grids, so it has the
// spectrogram's colour scale and the page's background, follows light and dark, and zooms. The scale is the title's, not
// the design's (`scaleTop`): it stays put while designs are compared. A dot is each cell within
// LEVELS_DB of the loudest (as the published image keeps those within 60 dB), quietest first so the loud ones lie on top.
// The image publish writes is still there, as a link.
import { useEffect, useMemo, useRef, useState } from 'react'

import { useHeatmapUrl, useSpectrogram } from '../api/hooks'
import { clock, decodeGrid, gridTimes, levelAt, levelColour, type Grid } from '../chart/grid'
import { pxOf, pyOf, zoomAbout, type Axes } from '../chart/plane'
import type { Range } from '../chart/view'
import { cssColour, useColourScheme } from '../theme'
import { PlaneControls, useZoomHistory, type Plane } from './PlaneControls'
import { SelectionBox } from './SelectionBox'
import { ColourBar, LEVELS_DB, linearTicks, MAX_HZ, scaleTop, useWidth, type DesignFilters } from './Spectrogram'
import { usePlaneGestures } from './usePlaneGestures'

const DEFAULT_X: Range = [1, 40]
const HEIGHT = 520
const MARGIN = { left: 64, top: 22, bottom: 28, gap: 16 }
const PANES = ['After the design', 'As measured']

/** Each cell's index, grouped by its byte (its level), so a draw sets each colour once and paints quietest first. */
export function byLevel(grid: Grid): Uint32Array[] {
  const counts = new Uint32Array(256)
  for (const v of grid.cells) if (v) counts[v]!++   // 0 is below the floor: never drawn
  const groups = Array.from({ length: 256 }, (_, v) => new Uint32Array(counts[v]!))
  const filled = new Uint32Array(256)
  grid.cells.forEach((v, i) => { if (v) groups[v]![filled[v]!++] = i })
  return groups
}

/** The dots of `grid` in `axes`: each cell inside the view and above `low` dB, as an ellipse a cell wide. */
function drawDots(context: CanvasRenderingContext2D, grid: Grid, groups: Uint32Array[], axes: Axes, levels: Range) {
  const cellWidth = (axes.width * grid.df) / (axes.x[1] - axes.x[0])
  const cellHeight = (axes.height * grid.dt) / (axes.t[1] - axes.t[0])
  const rx = Math.max(2.5, cellWidth * 0.45), ry = Math.max(1.5, cellHeight * 0.55)
  const [f0, f1] = axes.x, [t0, t1] = axes.t
  context.save()
  context.beginPath()
  context.rect(axes.left, axes.top, axes.width, axes.height)
  context.clip()
  for (let v = 1; v < 256; v++) {
    const cells = groups[v]!
    if (!cells.length) continue
    const db = grid.floorDb + (v - 1) * grid.stepDb
    if (db < levels[0]) continue
    const [r, g, b] = levelColour(db, levels)
    context.fillStyle = `rgb(${r},${g},${b})`
    context.beginPath()
    for (const i of cells) {
      const hz = grid.f0 + (i % grid.freqs) * grid.df, seconds = grid.t0 + Math.floor(i / grid.freqs) * grid.dt
      if (hz < f0 || hz > f1 || seconds < t0 || seconds > t1) continue
      const cx = pxOf(axes, hz), cy = pyOf(axes, seconds)
      context.moveTo(cx + rx, cy)
      context.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI)
    }
    context.fill()
  }
  context.restore()
}

/** The published image, opened on request: the PNG publish writes for this design. */
function PublishedImage({ titleId, candidate, variant }: { titleId: string; candidate: number; variant: string | null }) {
  const [asked, setAsked] = useState(false)
  const heatmap = useHeatmapUrl(titleId, candidate, asked, variant)
  if (!asked) return <button type="button" className="secondary small" onClick={() => setAsked(true)}>Published image…</button>
  if (heatmap.error) return <span role="alert" className="problem">{heatmap.error.message}</span>
  if (!heatmap.url) return <span className="muted" role="status">Drawing the published image…</span>
  return <a href={heatmap.url} target="_blank" rel="noreferrer">Open the published image</a>
}

export function HeatmapView({ titleId, candidate, variant = null, title, designs = [] }: {
  titleId: string; candidate: number; variant?: string | null; title: string
  /** Every design of the title, so the colour scale is the same whichever is drawn (`scaleTop`). */
  designs?: DesignFilters[]
}) {
  const before = useSpectrogram(titleId, candidate, false, MAX_HZ, true, variant)
  const after = useSpectrogram(titleId, candidate, true, MAX_HZ, true, variant)
  const canvas = useRef<HTMLCanvasElement>(null)
  const holder = useRef<HTMLDivElement>(null)
  const width = useWidth(holder)
  const scheme = useColourScheme()
  const beforeData = before.data, afterData = after.data
  const grids = useMemo(() => {
    const decode = (data: typeof beforeData) => (data && !data.pending ? decodeGrid(data.data) : null)
    const a = decode(afterData), b = decode(beforeData)
    return a && b ? { after: a, before: b, groups: [byLevel(a), byLevel(b)] } : null
  }, [beforeData, afterData])

  const fullT = grids ? gridTimes(grids.before) : [0, 1] as Range
  const fullX: Range = [0, grids ? Math.min(MAX_HZ, grids.before.f0 + grids.before.df * (grids.before.freqs - 1)) : MAX_HZ]
  const [plane, setPlane] = useState<Plane | null>(null)
  const shown: Plane = plane ?? { x: DEFAULT_X, t: fullT }
  const history = useZoomHistory(shown, setPlane)
  const reset = () => {
    history.clear()
    setPlane(null)
  }
  const top = useMemo(() => (grids ? scaleTop(grids.before, designs, grids.after) : 0), [grids, designs])
  const levels: Range = [top - LEVELS_DB, top]
  const paneWidth = Math.floor((width - MARGIN.left - MARGIN.gap) / 2), paneHeight = HEIGHT - MARGIN.top - MARGIN.bottom
  const panes: Axes[] = [0, 1].map((i) => ({ left: MARGIN.left + i * (paneWidth + MARGIN.gap), top: MARGIN.top,
                                              width: paneWidth, height: paneHeight, x: shown.x, t: shown.t, logX: false }))
  const { handlers, selection, hover } = usePlaneGestures({
    canvas, panes,
    onSelect: (x, t) => history.go({ x, t }),
    onWheel: (factor, hz, seconds) => history.go({ x: zoomAbout(shown.x, factor, fullX, false, hz),
                                                   t: zoomAbout(shown.t, factor, fullT, false, seconds) }),
    onReset: reset,
  })

  useEffect(() => {
    const context = canvas.current?.getContext('2d')
    if (!context || !grids) return
    context.clearRect(0, 0, width, HEIGHT)
    const ink = cssColour('muted', '#888'), line = cssColour('line', '#ccc'), fg = cssColour('fg', '#222')
    context.font = '11px system-ui, sans-serif'
    panes.forEach((axes, i) => {
      drawDots(context, i === 0 ? grids.after : grids.before, grids.groups[i]!, axes, levels)
      context.strokeStyle = line
      context.lineWidth = 1
      context.beginPath()
      for (const hz of linearTicks(axes.x, 6)) {
        const px = Math.round(pxOf(axes, hz)) + 0.5
        context.moveTo(px, axes.top)
        context.lineTo(px, axes.top + axes.height)
      }
      context.stroke()
      context.strokeRect(axes.left + 0.5, axes.top + 0.5, axes.width - 1, axes.height - 1)
      context.fillStyle = ink
      for (const hz of linearTicks(axes.x, 6)) context.fillText(String(hz), pxOf(axes, hz) - 6, HEIGHT - 10)
      context.fillStyle = fg
      context.fillText(PANES[i]!, axes.left, 14)
    })
    context.fillStyle = ink
    for (const s of linearTicks(shown.t, 8)) context.fillText(clock(s), 2, pyOf(panes[0]!, s) + 4)
  }, [grids, width, shown.x[0], shown.x[1], shown.t[0], shown.t[1], top, scheme])   // eslint-disable-line react-hooks/exhaustive-deps

  function readout() {
    if (!hover || !grids) return null
    const grid = hover.pane === 0 ? grids.after : grids.before
    const db = levelAt(grid, hover.seconds, hover.hz)
    return `${PANES[hover.pane]}: ${hover.hz.toFixed(1)} Hz · ${clock(hover.seconds)} · ${db === null ? 'below the scale'
      : `${db.toFixed(1)} dB`}`
  }

  const failed = before.error ?? after.error
  const waiting = before.data?.pending || after.data?.pending
  return (
    <div className="spectrogram heatmap" ref={holder}>
      <div className="row">
        <ColourBar low={`${Math.round(levels[0])} dB`} high={`${Math.round(levels[1])} dB`}
                   label={`${Math.round(levels[0])} to ${Math.round(levels[1])} dB`} />
        <PublishedImage titleId={titleId} candidate={candidate} variant={variant} />
      </div>
      {grids && <PlaneControls label="Heatmap view" plane={shown} full={{ x: fullX, t: fullT }} logX={false}
                               canBack={history.canBack} onChange={history.go} onBack={history.back} onReset={reset} />}
      {failed && <p role="alert" className="problem">{failed.message}</p>}
      {!failed && waiting && <p className="muted" role="status">Drawing the heatmap… (the service renders it once, then keeps it)</p>}
      {!failed && grids && (
        <>
          <div className="plane">
            <canvas ref={canvas} width={width} height={HEIGHT} role="img" tabIndex={0}
                    aria-label={`The heatmap of ${title}, design ${candidate + 1}: after the design, and as measured`}
                    {...handlers} />
            {selection && <SelectionBox selection={selection} width={width} height={HEIGHT} />}
          </div>
          <p className="muted readout">{readout() ?? 'Drag a box in either pane to zoom both, scroll to zoom about the pointer, double-click for the whole title.'}</p>
        </>
      )}
    </div>
  )
}
