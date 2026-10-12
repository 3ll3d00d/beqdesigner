import { useNavigate } from 'react-router'

import { useSubmitScan } from '../api/hooks'

/** List all configured sources again and follow the queued scan. */
export function ScanSources({ scanned }: { scanned: boolean }) {
  const scan = useSubmitScan()
  const navigate = useNavigate()
  return (
    <div>
      <button type="button" className="secondary" disabled={scan.isPending}
              onClick={() => scan.mutate({ sources: [], allow_empty: false }, {
                onSuccess: (job) => navigate(`/jobs/${job.id}`),
              })}>
        {scan.isPending ? 'Queuing scan…' : scanned ? 'Rescan sources' : 'Scan sources'}
      </button>
      {scan.error && <p role="alert" className="problem">{scan.error.message}</p>}
    </div>
  )
}
