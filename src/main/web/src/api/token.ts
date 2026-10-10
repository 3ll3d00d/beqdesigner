// The service token, kept in the browser (design/web-app.md §4): for this tab only (sessionStorage), or on this device
// when the person asks to be remembered (localStorage). Storage can be missing or refuse (a private window, blocked site
// data), so every access is guarded and the app still works for the tab's lifetime from memory.

const KEY = 'beqdesigner.service.token'

let inMemory: string | null = null

function storage(kind: 'session' | 'local'): Storage | null {
  try {
    return kind === 'session' ? window.sessionStorage : window.localStorage
  } catch {
    return null
  }
}

function read(kind: 'session' | 'local'): string | null {
  try {
    return storage(kind)?.getItem(KEY) ?? null
  } catch {
    return null
  }
}

export function readToken(): string | null {
  return read('session') ?? read('local') ?? inMemory
}

export function saveToken(token: string, remember: boolean): void {
  clearToken()
  inMemory = token
  try {
    storage(remember ? 'local' : 'session')?.setItem(KEY, token)
  } catch {
    // kept in memory only: signing in again is needed after a reload
  }
}

export function clearToken(): void {
  inMemory = null
  for (const kind of ['session', 'local'] as const) {
    try {
      storage(kind)?.removeItem(KEY)
    } catch {
      // nothing to clear
    }
  }
}
