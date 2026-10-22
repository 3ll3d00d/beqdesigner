// The system's light or dark setting, followed live: the page's colours are CSS variables that switch by themselves
// (styles.css), but a canvas or uPlot chart reads them once when it draws, so it draws again when the setting changes.
import { useSyncExternalStore } from 'react'

export type ColourScheme = 'light' | 'dark'

const QUERY = '(prefers-color-scheme: dark)'

function media(): MediaQueryList | null {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function' ? window.matchMedia(QUERY) : null
}

function subscribe(changed: () => void): () => void {
  const list = media()
  if (!list) return () => undefined
  list.addEventListener('change', changed)
  return () => list.removeEventListener('change', changed)
}

const current = (): ColourScheme => (media()?.matches ? 'dark' : 'light')

/** 'dark' or 'light', as the system says now; a component using it renders again when that changes. */
export function useColourScheme(): ColourScheme {
  return useSyncExternalStore(subscribe, current, () => 'light')
}

/** A colour the stylesheet defines (`--name`), as it is now; `fallback` when it defines none. */
export function cssColour(name: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback
  return getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim() || fallback
}
