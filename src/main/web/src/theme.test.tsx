// The system's light or dark setting, followed live (src/theme.ts).
import { act, render, screen } from '@testing-library/react'

import { useColourScheme } from './theme'

function Shows() {
  return <p>{useColourScheme()}</p>
}

it('follows the system setting as it changes', () => {
  const listeners = new Set<() => void>()
  const list = { matches: false, addEventListener: (_: string, l: () => void) => listeners.add(l),
                 removeEventListener: (_: string, l: () => void) => listeners.delete(l) }
  const original = window.matchMedia
  window.matchMedia = vi.fn(() => list) as unknown as typeof window.matchMedia
  try {
    render(<Shows />)
    expect(screen.getByText('light')).toBeInTheDocument()
    act(() => {
      list.matches = true
      listeners.forEach((l) => l())
    })
    expect(screen.getByText('dark')).toBeInTheDocument()
  } finally {
    window.matchMedia = original
  }
})
