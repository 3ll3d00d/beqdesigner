import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

import { clearToken } from '../api/token'

afterEach(() => {
  cleanup()
  clearToken()   // storage, and the in-memory copy a blocked storage falls back to
  window.sessionStorage.clear()
  window.localStorage.clear()
})
