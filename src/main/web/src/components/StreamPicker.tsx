// Choosing a title's audio stream (design/web-parity.md WP5): the streams the source lists, the one extracted now, and a
// run job that extracts and designs another -- as the desktop's *Choose audio stream* does.
import { useState } from 'react'
import { useNavigate } from 'react-router'

import { useChooseStream, useStreams } from '../api/hooks'

export function StreamPicker({ titleId, blocked }: { titleId: string; blocked: string }) {
  const [open, setOpen] = useState(false)
  const streams = useStreams(titleId, open)
  const choose = useChooseStream(titleId)
  const navigate = useNavigate()
  const [picked, setPicked] = useState<number | null>(null)
  if (!open) {
    return (
      <button type="button" className="secondary" disabled={!!blocked} title={blocked || undefined}
              onClick={() => setOpen(true)}>Audio stream…</button>
    )
  }
  const data = streams.data
  const chosen = picked ?? data?.chosen ?? 0
  return (
    <div className="confirm" role="group" aria-label="Audio stream">
      {streams.isPending && <p className="muted">Reading the streams…</p>}
      {streams.isError && <p role="alert" className="problem">{streams.error.message}</p>}
      {data && (data.streams.length === 0
        ? <p className="muted">The source lists no audio streams for this title: choose one in the BEQDesigner app, which
            reads them from the file.</p>
        : (
          <>
            <label className="inline">Extract
              <select value={chosen} onChange={(e) => setPicked(Number(e.target.value))}>
                {data.streams.map((s) => (
                  <option key={s.index} value={s.index}>
                    {s.text}{s.index === data.chosen ? ' (extracted now)' : ''}{s.preferred ? ' (preferred)' : ''}
                  </option>
                ))}
              </select>
            </label>
            {data.source === 'manual' && <p className="muted">The stream extracted now was chosen by a person.</p>}
            <p>Choosing another forgets the extraction and design; one job extracts and designs that stream, and the new
              designs need review.</p>
          </>
        ))}
      <div className="actions">
        <button type="button" disabled={!data || !data.streams.length || chosen === data.chosen || choose.isPending}
                onClick={() => choose.mutate(chosen, { onSuccess: (job) => navigate(`/jobs/${job.id}`) })}>
          Extract this stream
        </button>
        <button type="button" className="secondary" disabled={choose.isPending} onClick={() => setOpen(false)}>Back</button>
      </div>
      {choose.error && <p role="alert" className="problem">{choose.error.message}</p>}
    </div>
  )
}
