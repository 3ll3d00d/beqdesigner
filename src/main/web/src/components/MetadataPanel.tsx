// A title's metadata and artwork, edited in the browser (design/web-parity.md WP4): the desktop title page's Metadata tab
// over the service's routes. A field is saved when it is left (or Enter, or Save), only the fields touched, against the
// metadata as it was read: if a run or someone else changed it since, the service refuses, the page reads it again and
// what was typed is kept to save again. TMDB Reload fills the form without saving it.
import { useState } from 'react'

import type { Schemas } from '../api/client'
import { useArtworkUrl, useClearArtwork, useEditMetadata, useMetadataFields, useSetArtwork, useStatus, useTmdbLookup }
  from '../api/hooks'

type Review = Schemas['Review']
type Field = Schemas['MetadataField']

/** The field a problem the service reported is about, so it can be marked (as the title page marks it). */
export function problemField(problem: string): string | null {
  const text = problem.toLowerCase()
  if (text.includes('audio type')) return 'audio_types'
  for (const key of ['title', 'year', 'episodes']) if (text.includes(key)) return key
  return null
}

/** The audio types of a comma-separated text, with `choice` added or taken out; values not offered are kept. */
export function toggleChoice(text: string, choice: string, on: boolean): string {
  const values = text.split(',').map((v) => v.trim()).filter(Boolean)
  const kept = values.filter((v) => v !== choice)
  return (on ? [...kept, choice] : kept).join(', ')
}

function readAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result).replace(/^data:[^,]*,/, ''))
    reader.onerror = () => reject(reader.error ?? new Error('could not read the file'))
    reader.readAsDataURL(file)
  })
}

const statusOf = (error: unknown) => (error as { status?: number } | null)?.status

export function MetadataPanel({ review }: { review: Review }) {
  const fields = useMetadataFields()
  const edit = useEditMetadata(review.id)
  const tmdb = useTmdbLookup(review.id)
  const setArt = useSetArtwork(review.id)
  const clearArt = useClearArtwork(review.id)
  const status = useStatus()
  const [typed, setTyped] = useState<Record<string, string>>({})
  const [extras, setExtras] = useState<Record<string, unknown>>({})
  const [said, setSaid] = useState<{ text: string; problem: boolean }>({ text: '', problem: false })
  const [artUrl, setArtUrl] = useState('')
  const [shownFor, setShownFor] = useState(review.id)
  if (shownFor !== review.id) {   // another title: nothing typed for this one
    setShownFor(review.id)
    setTyped({})
    setExtras({})
    setSaid({ text: '', problem: false })
    setArtUrl('')
  }
  const poster = useArtworkUrl(review.id, review.artwork?.present ? review.metadata_digest : null)
  const locked = review.in_flight
  const busy = edit.isPending || setArt.isPending || clearArt.isPending
  const dirty = Object.keys(typed).length + Object.keys(extras).length > 0
  const value = (key: string) => typed[key] ?? review.metadata_texts[key] ?? ''
  const marked = new Set(review.metadata_problems.map(problemField).filter(Boolean))

  function type(key: string, text: string) {
    setTyped((t) => ({ ...t, [key]: text }))
  }

  function save() {
    if (!dirty || locked || edit.isPending) return
    const texts = typed, sent = extras
    edit.mutate({ digest: review.metadata_digest, texts, extras: sent }, {
      onSuccess: () => {
        // keep anything typed while the save was on its way
        setTyped((t) => Object.fromEntries(Object.entries(t).filter(([k, v]) => texts[k] !== v)))
        setExtras((e) => Object.fromEntries(Object.entries(e).filter(([k]) => !(k in sent))))
        setSaid({ text: 'Saved.', problem: false })
      },
      onError: (error) => setSaid({
        text: statusOf(error) === 409 && error.message.startsWith('Changed since it was read')
          ? 'The metadata changed since this page read it (a run, or someone else). It has been read again; what you '
            + 'typed is kept: Save to apply it.'
          : `Not saved: ${error.message}`,
        problem: true,
      }),
    })
  }

  function revert() {
    setTyped({})
    setExtras({})
    setSaid({ text: 'Reverted to what is saved.', problem: false })
  }

  function reload() {
    tmdb.mutate({ tmdb_id: value('the_movie_db'), title: value('title'), year: value('year'), season: value('season') }, {
      onSuccess: (proposal) => {
        if (proposal.found) {
          setTyped((t) => ({ ...t, ...proposal.texts }))
          setExtras((e) => ({ ...e, ...proposal.extras }))
        }
        setSaid({ text: proposal.message, problem: !proposal.found })
      },
      onError: (error) => setSaid({ text: error.message, problem: true }),
    })
  }

  function artDone(text: string) {
    return { onSuccess: () => setSaid({ text, problem: false }),
             onError: (error: Error) => setSaid({ text: error.message, problem: true }) }
  }

  async function upload(file: File | undefined) {
    if (!file) return
    try {
      const data = await readAsBase64(file)
      setArt.mutate({ digest: review.metadata_digest, data_base64: data }, artDone('Artwork set.'))
    } catch (error) {
      setSaid({ text: `Not used: ${(error as Error).message}`, problem: true })
    }
  }

  function input(field: Field) {
    const common = {
      id: `meta-${field.key}`, value: value(field.key), disabled: locked, title: field.tip,
      'aria-invalid': marked.has(field.key) || undefined,
      onChange: (e: React.ChangeEvent<HTMLInputElement>) => type(field.key, e.target.value),
      onBlur: save,
      onKeyDown: (e: React.KeyboardEvent) => { if (e.key === 'Enter') save() },
    }
    if (field.kind === 'choices') {
      const chosen = new Set(value(field.key).split(',').map((v) => v.trim()).filter(Boolean))
      return (
        <div className="choices" role="group" aria-label={field.label}>
          {field.choices.map((choice) => (
            <label key={choice} className="check">
              <input type="checkbox" checked={chosen.has(choice)} disabled={locked}
                     onChange={(e) => type(field.key, toggleChoice(value(field.key), choice, e.target.checked))}
                     onBlur={save} />
              {choice}
            </label>
          ))}
          {[...chosen].filter((v) => !field.choices.includes(v)).map((v) => <span key={v} className="muted">{v}</span>)}
        </div>
      )
    }
    if (field.kind === 'choice') {
      return (
        <>
          <input {...common} list={`meta-${field.key}-choices`} />
          <datalist id={`meta-${field.key}-choices`}>{field.choices.map((c) => <option key={c} value={c} />)}</datalist>
        </>
      )
    }
    return <input {...common} />
  }

  const row = (field: Field) => (
    <div key={field.key} className={`meta-row${marked.has(field.key) ? ' marked' : ''}`}>
      {field.kind === 'choices' ? <span className="meta-label">{field.label}</span>
        : <label className="meta-label" htmlFor={`meta-${field.key}`}>{field.label}</label>}
      {input(field)}
    </div>
  )
  const all = fields.data ?? []
  const genres = Array.isArray(extras.genres) ? extras.genres : (review.metadata as Record<string, unknown>).genres
  return (
    <section className="card metadata-panel" aria-labelledby="metadata-heading">
      <h2 id="metadata-heading">Edit metadata and artwork{dirty && <span className="muted"> (unsaved edits)</span>}</h2>
      {locked && <p className="muted">A run is working on this title: its metadata cannot be edited until it finishes.</p>}
      {review.metadata_problems.length > 0 && (
        <div className="problem" aria-label="What the metadata lacks">
          <p>Not complete, so it cannot be accepted:</p>
          <ul>{review.metadata_problems.map((p) => <li key={p}>{p}</li>)}</ul>
        </div>
      )}
      {fields.isError && <p role="alert" className="problem">{fields.error.message}</p>}
      <div className="meta-form">{all.filter((f) => f.group === 'essentials').map(row)}</div>
      {Array.isArray(genres) && genres.length > 0 && (
        <p className="muted">Genres: {genres.map((g) => (g && typeof g === 'object' && 'name' in g ? String(g.name) : String(g)))
          .join(', ')}</p>
      )}
      <details>
        <summary>More</summary>
        <div className="meta-form">{all.filter((f) => f.group === 'more').map(row)}</div>
      </details>
      <div className="actions">
        <button type="button" onClick={save} disabled={!dirty || locked || busy}>Save</button>
        {/* pressing Revert must not first leave the field, which would save what it is to throw away */}
        <button type="button" className="secondary" onMouseDown={(e) => e.preventDefault()} onClick={revert}
                disabled={!dirty || busy}>Revert</button>
        <button type="button" className="secondary" onClick={reload} disabled={locked || tmdb.isPending || status.data?.tmdb === false}
                title={status.data?.tmdb === false ? 'No TMDB key is set on the service' : 'Fill in what TMDB knows'}>
          {tmdb.isPending ? 'Asking TMDB…' : 'Reload from TMDB'}
        </button>
      </div>
      <h3>Artwork</h3>
      <div className="artwork">
        {poster ? <img src={poster} alt={`Poster for ${review.title}`} />
          : <p className="muted">{review.artwork?.name ? 'The chosen file is missing.' : 'No artwork.'}</p>}
        <div className="artwork-actions">
          <label className="button secondary">Upload…
            <input type="file" accept="image/png,image/jpeg" hidden disabled={locked || busy}
                   onChange={(e) => { void upload(e.target.files?.[0]); e.target.value = '' }} />
          </label>
          <span className="limits">
            <input aria-label="Image address" placeholder="https://… (png or jpeg)" value={artUrl} disabled={locked}
                   onChange={(e) => setArtUrl(e.target.value)} />
            <button type="button" className="secondary" disabled={!artUrl.trim() || locked || busy}
                    onClick={() => setArt.mutate({ digest: review.metadata_digest, url: artUrl.trim() },
                                                 { ...artDone('Artwork downloaded and set.'), onSuccess: () => {
                                                   setArtUrl('')
                                                   setSaid({ text: 'Artwork downloaded and set.', problem: false })
                                                 } })}>Download</button>
          </span>
          <button type="button" className="secondary" disabled={!review.artwork?.name || locked || busy}
                  onClick={() => clearArt.mutate(review.metadata_digest, artDone('Artwork cleared.'))}>Clear</button>
        </div>
      </div>
      {said.text && <p role={said.problem ? 'alert' : 'status'} className={said.problem ? 'problem' : 'muted'}>{said.text}</p>}
    </section>
  )
}
