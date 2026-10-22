// The box being dragged on a spectrogram or heatmap, over the canvas, with the ranges it would zoom to.
import { clock } from '../chart/grid'
import type { Selection } from './usePlaneGestures'

const hz = (v: number) => v.toFixed(v < 10 ? 2 : 1)

export function SelectionBox({ selection, width, height }: { selection: Selection; width: number; height: number }) {
  const percent = (v: number, of: number) => `${(100 * v) / of}%`
  return (
    <div className="selection" aria-hidden="true"
         style={{ left: percent(selection.left, width), top: percent(selection.top, height),
                  width: percent(selection.width, width), height: percent(selection.height, height) }}>
      <span className="selection-label">
        {hz(selection.x[0])}–{hz(selection.x[1])} Hz · {clock(selection.t[0])}–{clock(selection.t[1])}
      </span>
    </div>
  )
}
