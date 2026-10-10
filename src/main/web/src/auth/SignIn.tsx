import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate, useSearchParams } from 'react-router'

import { useAuth } from './auth'

/** Where to go after signing in: a path inside the app, never another site. */
function safeNext(next: string | null): string {
  return next && next.startsWith('/') && !next.startsWith('//') ? next : '/'
}

export function SignIn() {
  const { token, signIn } = useAuth()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const [value, setValue] = useState('')
  const [remember, setRemember] = useState(false)
  const [problem, setProblem] = useState('')
  const [busy, setBusy] = useState(false)
  const next = safeNext(params.get('next'))

  if (token) return <Navigate to={next} replace />

  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    const reason = await signIn(value, remember)
    setBusy(false)
    if (reason) setProblem(reason)
    else navigate(next, { replace: true })
  }

  return (
    <main className="signin">
      <form onSubmit={submit} aria-labelledby="signin-heading">
        <h1 id="signin-heading">BEQDesigner pipeline</h1>
        <p className="muted">Sign in with the service token (BEQ_SERVICE_TOKEN).</p>
        <label>
          Token
          <input type="password" autoComplete="current-password" autoFocus value={value}
                 onChange={(e) => setValue(e.target.value)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
          Remember on this device
        </label>
        {problem && <p role="alert" className="problem">{problem}</p>}
        <button type="submit" disabled={busy}>{busy ? 'Checking…' : 'Sign in'}</button>
      </form>
    </main>
  )
}
