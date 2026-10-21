import '@testing-library/jest-dom/vitest'
import { cleanup, configure } from '@testing-library/react'
import { afterEach } from 'vitest'

import { clearToken } from '../api/token'

// findBy*/waitFor wait 1 s by default: a parallel run on a loaded machine has missed that (the scan shortcut's redirect,
// once in about ten runs). 5 s lets a slow render through and still fails a real hang well inside a test's own limit.
configure({ asyncUtilTimeout: 5000 })

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
