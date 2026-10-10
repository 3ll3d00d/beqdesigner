import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

import { clearToken } from '../api/token'

// jsdom has no layout: the chart observes its size
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver

// nor media queries, which uPlot asks for its pixel ratio when it loads
window.matchMedia ??= ((query: string) => ({
  matches: false, media: query, onchange: null, addEventListener() {}, removeEventListener() {}, addListener() {},
  removeListener() {}, dispatchEvent: () => false,
})) as unknown as typeof window.matchMedia

afterEach(() => {
  cleanup()
  clearToken()   // storage, and the in-memory copy a blocked storage falls back to
  window.sessionStorage.clear()
  window.localStorage.clear()
})
