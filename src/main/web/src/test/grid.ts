// A spectrogram grid as the service encodes it (pipeline.library.spectrum.SpectrogramGrid.encode), for tests.
export function encodeGrid({ times, freqs, cells, t0 = 0, dt = 1, f0 = 0, df = 10, floorDb = -100, stepDb = 1 }: {
  times: number; freqs: number; cells: number[]; t0?: number; dt?: number; f0?: number; df?: number; floorDb?: number;
  stepDb?: number
}): ArrayBuffer {
  const buffer = new ArrayBuffer(40 + times * freqs)
  const view = new DataView(buffer)
  'BEQS'.split('').forEach((c, i) => view.setUint8(i, c.charCodeAt(0)))
  view.setUint16(4, 1, true)
  view.setUint32(8, times, true)
  view.setUint32(12, freqs, true)
  ;[t0, dt, f0, df, floorDb, stepDb].forEach((v, i) => view.setFloat32(16 + 4 * i, v, true))
  new Uint8Array(buffer, 40).set(cells)
  return buffer
}
