// Editing a title's metadata and artwork in the browser (design/web-parity.md WP4) against a mocked service: saving what
// was touched when a field is left, a change made meanwhile kept to save again, TMDB filling the form unsaved, the
// artwork, and a title a run has in hand. uPlot is replaced (jsdom has no canvas).
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from '../api/token'
import { problemField, toggleChoice } from '../components/MetadataPanel'
import { chart, index, METADATA_FIELDS, review } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

vi.mock('uplot', () => ({
  default: class {
    over = document.createElement('div')
    cursor = { left: -1 }
    setSize() {}
    destroy() {}
  },
}))

beforeEach(() => saveToken(TOKEN, false))

const json = (body: unknown, code?: number) => HttpResponse.json(body as never, code ? { status: code } : undefined)

function routes(over: Parameters<typeof review>[0] = {}, tmdb = true) {
  const edits: Record<string, unknown>[] = []
  const art: unknown[] = []
  let current = review(over)
  server.use(
    statusRoute(status({ index: index(), tmdb })),
    http.get('/v1/titles/a/review', () => json(current)),
    http.get('/v1/metadata/fields', () => json(METADATA_FIELDS)),
    http.get('/v1/titles/a/chart', () => json(chart(0))),
    http.patch('/v1/titles/a/metadata', async ({ request }) => {
      const body = await request.json() as { digest: string; texts: Record<string, string> }
      edits.push(body)
      if (body.digest !== current.metadata_digest) {
        return json({ title: 'Changed since it was read', status: 409, detail: 'read it again' }, 409)
      }
      current = review({ ...over, metadata_digest: `m${edits.length + 1}`,
                         metadata_texts: { ...current.metadata_texts, ...body.texts } })
      return json(current)
    }),
    http.post('/v1/titles/a/metadata/tmdb', async ({ request }) => {
      const body = await request.json() as Record<string, string>
      return json(body.title === 'Nope'
        ? { found: false, texts: {}, extras: {}, message: 'TMDB has no match for "Nope". Nothing was changed.' }
        : { found: true, texts: { the_movie_db: '348', year: '1979' }, extras: { genres: [{ name: 'Horror' }] },
            message: 'Filled in from TMDB. Nothing is saved until you save the edit.' })
    }),
    http.put('/v1/titles/a/artwork', async ({ request }) => {
      const body = await request.json() as Record<string, string>
      art.push(body)
      if (body.url?.includes('bad')) return json({ title: 'Not used', status: 422, detail: 'Not used: that is not a png or jpeg image.' }, 422)
      current = review({ ...over, metadata_digest: 'art', artwork: { present: true, overridden: true, name: 'a.png' } })
      return json(current)
    }),
    http.get('/v1/titles/a/artwork', () => new HttpResponse(new Blob(['png'], { type: 'image/png' }))),
    http.delete('/v1/titles/a/artwork', ({ request }) => {
      art.push({ cleared: new URL(request.url).searchParams.get('digest') })
      current = review(over)
      return json(current)
    }),
  )
  return { edits, art, change: (to: Parameters<typeof review>[0]) => { current = review({ ...over, ...to }) } }
}

async function open() {
  renderApp('/titles/a')
  const panel = await screen.findByRole('region', { name: /Edit metadata and artwork/ })
  await within(panel).findByLabelText('Title')   // the fields come from their own request
  return panel
}

describe('the words', () => {
  it('marks the field a problem is about', () => {
    expect(problemField('audio types are missing')).toBe('audio_types')
    expect(problemField('No year')).toBe('year')
    expect(problemField('something else')).toBeNull()
  })

  it('toggles an audio type and keeps values that are not offered', () => {
    expect(toggleChoice('Atmos, Odd', 'DD+', true)).toBe('Atmos, Odd, DD+')
    expect(toggleChoice('Atmos, Odd', 'Atmos', false)).toBe('Odd')
  })
})

describe('the metadata panel', () => {
  it('saves only what was touched, against what was read, when the field is left', async () => {
    const { edits } = routes()
    const panel = await open()
    const note = within(panel).getByLabelText('Note')
    await userEvent.type(note, 'Kept')
    expect(within(panel).getByRole('heading', { level: 2 })).toHaveTextContent('(unsaved edits)')
    await userEvent.tab()
    await waitFor(() => expect(edits).toEqual([{ digest: 'm1', texts: { note: 'Kept' }, extras: {} }]))
    expect(await within(panel).findByText('Saved.')).toBeInTheDocument()
    expect(within(panel).getByLabelText('Note')).toHaveValue('Kept')
    expect(within(panel).getByRole('heading', { level: 2 })).not.toHaveTextContent('unsaved')
  })

  it('sends audio types chosen by checkbox as text', async () => {
    const { edits } = routes()
    const panel = await open()
    await userEvent.click(within(panel).getByRole('checkbox', { name: 'Atmos' }))
    await userEvent.click(within(panel).getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(edits.at(-1)).toMatchObject({ texts: { audio_types: 'DTS-HD MA 5.1, Atmos' } }))
  })

  it('keeps what was typed when the metadata changed meanwhile, and saves it against what is there now', async () => {
    const { edits, change } = routes()
    const panel = await open()
    change({ metadata_digest: 'm-run' })   // a run wrote the entry after the page read it
    await userEvent.type(within(panel).getByLabelText('Note'), 'Mine{Enter}')
    expect(await within(panel).findByRole('alert')).toHaveTextContent('changed since this page read it')
    expect(within(panel).getByLabelText('Note')).toHaveValue('Mine')
    await userEvent.click(within(panel).getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(edits.at(-1)).toEqual({ digest: 'm-run', texts: { note: 'Mine' }, extras: {} }))
  })

  it('marks what is missing, and Revert throws away what was typed', async () => {
    routes({ metadata_problems: ['No year'], metadata_texts: { title: 'Alien', year: '' } })
    const panel = await open()
    expect(within(panel).getByLabelText('What the metadata lacks')).toHaveTextContent('No year')
    expect(within(panel).getByLabelText('Year')).toHaveAttribute('aria-invalid', 'true')
    await userEvent.type(within(panel).getByLabelText('Title'), ' 2')
    await userEvent.click(within(panel).getByRole('button', { name: 'Revert' }))
    expect(within(panel).getByLabelText('Title')).toHaveValue('Alien')
  })

  it('fills the form from TMDB without saving it', async () => {
    const { edits } = routes()
    const panel = await open()
    await userEvent.click(within(panel).getByRole('button', { name: 'Reload from TMDB' }))
    await waitFor(() => expect(within(panel).getByLabelText('TMDB id')).toHaveValue('348'))
    expect(within(panel).getByText('Genres: Horror')).toBeInTheDocument()
    expect(edits).toEqual([])
    await userEvent.click(within(panel).getByRole('button', { name: 'Save' }))
    await waitFor(() => expect(edits.at(-1)).toEqual({ digest: 'm1', texts: { the_movie_db: '348', year: '1979' },
                                                       extras: { genres: [{ name: 'Horror' }] } }))
  })

  it('says when TMDB has no match', async () => {
    routes({ metadata_texts: { title: 'Nope' } })
    const panel = await open()
    await userEvent.click(within(panel).getByRole('button', { name: 'Reload from TMDB' }))
    expect(await within(panel).findByRole('alert')).toHaveTextContent('no match for "Nope"')
  })

  it('offers no reload when the service has no TMDB key', async () => {
    routes({}, false)
    const panel = await open()
    await waitFor(() => expect(within(panel).getByRole('button', { name: 'Reload from TMDB' })).toBeDisabled())
  })

  it('sets artwork from a URL or a file, shows it, says why one is refused, and clears it', async () => {
    URL.createObjectURL = vi.fn(() => 'blob:poster')
    URL.revokeObjectURL = vi.fn()
    const { art } = routes()
    const panel = await open()
    await userEvent.type(within(panel).getByLabelText('Image address'), 'https://example.org/bad.html')
    await userEvent.click(within(panel).getByRole('button', { name: 'Download' }))
    expect(await within(panel).findByRole('alert')).toHaveTextContent('not a png or jpeg')
    const file = new File(['png'], 'p.png', { type: 'image/png' })
    fireEvent.change(panel.querySelector('input[type="file"]')!, { target: { files: [file] } })
    await waitFor(() => expect(art.at(-1)).toEqual({ digest: 'm1', data_base64: btoa('png') }))
    expect(await within(panel).findByRole('img', { name: 'Poster for Alien' })).toHaveAttribute('src', 'blob:poster')
    await userEvent.click(within(panel).getByRole('button', { name: 'Clear' }))
    await waitFor(() => expect(art.at(-1)).toEqual({ cleared: 'art' }))
    expect(await within(panel).findByText('No artwork.')).toBeInTheDocument()
  })

  it('edits nothing while a run has the title in hand', async () => {
    routes({ in_flight: true })
    const panel = await open()
    expect(within(panel).getByText(/A run is working on this title/)).toBeInTheDocument()
    expect(within(panel).getByLabelText('Note')).toBeDisabled()
    expect(within(panel).getByRole('button', { name: 'Reload from TMDB' })).toBeDisabled()
  })
})
