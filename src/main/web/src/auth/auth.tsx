// Who is signed in: the service token, and the client that carries it. A 401 from any call signs the person out, so the
// app returns to sign-in rather than showing errors on every screen.
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

import { makeClient, problemText, type ServiceClient } from '../api/client'
import { clearToken, readToken, saveToken } from '../api/token'

interface Auth {
  token: string | null
  client: ServiceClient
  /** Checks the token against the service, and keeps it if it works. Resolves to the reason it did not, or ''. */
  signIn: (token: string, remember: boolean) => Promise<string>
  signOut: () => void
}

const AuthContext = createContext<Auth | null>(null)

export function AuthProvider({ children, fetch }: { children: ReactNode; fetch?: typeof globalThis.fetch }) {
  // api/token.ts holds the token (storage, else memory) and every call reads it from there; the state re-renders on a change
  const [token, setToken] = useState<string | null>(() => readToken())

  const signOut = useCallback(() => {
    clearToken()
    setToken(null)
  }, [])

  const client = useMemo(() => makeClient({ token: readToken, onUnauthorized: signOut, fetch }), [signOut, fetch])

  const signIn = useCallback(
    async (candidate: string, remember: boolean) => {
      const trimmed = candidate.trim()
      if (!trimmed) return 'Enter the service token.'
      const trial = makeClient({ token: () => trimmed, onUnauthorized: () => {}, fetch })
      try {
        const { error, response } = await trial.GET('/v1/status')
        if (response.status === 401) return 'The service did not accept that token.'
        if (error !== undefined) return problemText(error, `The service answered ${response.status}.`)
      } catch (error) {
        return `The service could not be reached: ${problemText(error)}`
      }
      saveToken(trimmed, remember)
      setToken(trimmed)
      return ''
    },
    [fetch],
  )

  const value = useMemo(() => ({ token, client, signIn, signOut }), [token, client, signIn, signOut])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): Auth {
  const auth = useContext(AuthContext)
  if (!auth) throw new Error('useAuth needs an AuthProvider')
  return auth
}
