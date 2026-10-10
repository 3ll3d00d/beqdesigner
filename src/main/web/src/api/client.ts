// A typed client for the pipeline service (docs/schema/service.openapi.json, generated into ./schema.d.ts): every call
// carries the bearer token, and a 401 signs the person out.
import createClient, { type Middleware } from 'openapi-fetch'

import type { components, paths } from './schema'

export type Schemas = components['schemas']
export type Problem = Schemas['Problem']
export type ServiceClient = ReturnType<typeof createClient<paths>>

export interface ClientOptions {
  token: () => string | null
  onUnauthorized: () => void
  baseUrl?: string
  fetch?: typeof globalThis.fetch
}

export function makeClient({ token, onUnauthorized, baseUrl = '', fetch }: ClientOptions): ServiceClient {
  const client = createClient<paths>({ baseUrl: baseUrl || globalThis.location?.origin || '', fetch })
  const auth: Middleware = {
    onRequest({ request }) {
      const value = token()
      if (value) request.headers.set('Authorization', `Bearer ${value}`)
      return request
    },
    onResponse({ response }) {
      if (response.status === 401) onUnauthorized()
      return response
    },
  }
  client.use(auth)
  return client
}

/** What went wrong, in words, from a Problem body (or anything else a failed call gave). */
export function problemText(error: unknown, fallback = 'The service did not answer.'): string {
  if (error && typeof error === 'object') {
    const { title, detail } = error as Partial<Problem>
    if (title || detail) return [title, detail].filter(Boolean).join(': ')
  }
  if (error instanceof Error && error.message) return error.message
  return fallback
}

/** The data of a call, or a thrown Error carrying its Problem's words: for TanStack Query, which wants a throw. */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (result.error !== undefined || result.data === undefined) {
    const error = new Error(problemText(result.error, `${result.response.status} ${result.response.statusText}`))
    Object.assign(error, { status: result.response.status, problem: result.error })
    throw error
  }
  return result.data
}
