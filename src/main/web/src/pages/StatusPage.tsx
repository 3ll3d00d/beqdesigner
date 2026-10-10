// What the service is doing. W3 shows only that the app reaches it; W4 makes this the status screen of design/web-review.md §4.
import { useQuery } from '@tanstack/react-query'

import { unwrap } from '../api/client'
import { useAuth } from '../auth/auth'

export function StatusPage() {
  const { client } = useAuth()
  const status = useQuery({
    queryKey: ['status'],
    queryFn: async () => unwrap(await client.GET('/v1/status')),
    refetchInterval: 10_000,
  })
  return (
    <section>
      <h1>Status</h1>
      {status.isPending && <p className="muted">Asking the service…</p>}
      {status.isError && <p role="alert" className="problem">{status.error.message}</p>}
      {status.data && (
        <dl className="facts">
          <dt>Service</dt>
          <dd>{status.data.version}</dd>
          <dt>Titles</dt>
          <dd>{status.data.index ? status.data.index.titles.toLocaleString() : 'not scanned yet'}</dd>
          <dt>Job</dt>
          <dd>{status.data.current_job ? `${status.data.current_job.kind}, ${status.data.current_job.state}` : 'none running'}</dd>
        </dl>
      )}
    </section>
  )
}
