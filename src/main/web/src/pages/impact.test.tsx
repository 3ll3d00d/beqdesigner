// Which out-of-date designs to redesign (contract 1.6, design/redesign-scope.md step 5), against a mocked service: the
// check, its answer by tier with the reasons, and redesigning or keeping the titles chosen.
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import type { Schemas } from '../api/client'
import { saveToken } from '../api/token'
import { index, title } from '../test/fixtures'
import { renderApp } from '../test/render'
import { server, status, statusRoute, TOKEN } from '../test/server'

beforeEach(() => saveToken(TOKEN, false))

const json = (body: unknown, code?: number) => HttpResponse.json(body as never, code ? { status: code } : undefined)

const reason = (detail: string) => ({ change: 'fit-choice', feature: 'fit.choice', detail })

function check(overrides: Partial<Schemas['ImpactCheck']> = {}): Schemas['ImpactCheck'] {
  return {
    state: 'succeeded', started_at: '2026-10-25T00:00:00Z', finished_at: '2026-10-25T00:01:00Z', done: 4, total: 4,
    error: '', designer_url: 'http://d/design', design_version: 7, design_settings: 's0',
    changes: [{ design_version: 7, id: 'fit-choice', summary: 'the fitter chooses among near-equal fits',
                moved: 'swapped designed/declined or moved > 1 dB', basis: [{ class: 'declined_after_fit', measured: 34, moved: 18 }] }],
    titles: [
      { id: 'a', title: 'Alien', year: '1979', tier: 'will_move', score: 1, from_design_version: 6, reasons: [reason('a threshold moved past it')] },
      { id: 'h', title: 'Heat', year: '1995', tier: 'unknown', score: null, from_design_version: null, unknown_reason: 'no_record',
        reasons: [{ change: null, feature: null, detail: 'the designer named no run record for this design' }] },
      { id: 'b', title: 'Blade', year: '1998', tier: 'likely', score: 0.7, from_design_version: 6, reasons: [reason('near-equal fits')] },
      { id: 'd', title: 'Dune', year: '2021', tier: 'unaffected', score: 0, from_design_version: 6, reasons: [] },
    ],
    ...overrides,
  }
}

function designer(version: number): Schemas['DesignerStatus'] {
  return { name: 'beqforge', reachable: true, detail: 'http://d/design', design_version: version } as Schemas['DesignerStatus']
}

function routes(answer: () => Schemas['ImpactCheck'], sent: ReturnType<typeof sentTo>, version = 7) {
  server.use(statusRoute(status({ index: index(), designer: designer(version) })),
             http.get('/v1/titles', () => json({ total: 4, offset: 0, limit: 100,
                                                 titles: ['a', 'h', 'b', 'd'].map((id) => title(id)) })),
             http.get('/v1/impact', () => json(answer())),
             http.post('/v1/impact', async ({ request }) => {
               sent.impact.push(await request.json())
               return json(check({ state: 'running', done: 0, titles: [] }), 202)
             }),
             http.post('/v1/revise', async ({ request }) => {
               sent.revise.push(await request.json())
               return json({ revised: [], failed: [], job: null })
             }),
             http.post('/v1/reviewed', async ({ request }) => {
               const body = await request.json() as { ids: string[] }
               sent.reviewed.push(body)
               return json({ design_version: 7, design_settings: 's0', titles: body.ids.map((id) => (
                 id === 'b' ? { id, reviewed: false, reason: 'a run is working on this title now' } : { id, reviewed: true, reason: '' })) })
             }))
}

const sentTo = () => ({ impact: [] as unknown[], revise: [] as unknown[], reviewed: [] as unknown[] })

describe('Which out-of-date designs to redesign', () => {
  it('asks the designer, and shows its answer by tier, most likely to move first, with the reasons', async () => {
    const sent = sentTo()
    let answer = check({ state: 'idle', titles: [], changes: [], design_version: null })
    routes(() => answer, sent)
    renderApp('/titles?out_of_date=true')

    await userEvent.click(await screen.findByRole('button', { name: 'Check what a redesign would move' }))
    expect(sent.impact).toEqual([{ ids: [] }])
    answer = check()

    const panel = await screen.findByRole('region', { name: 'What a redesign would move' })
    await within(panel).findByRole('heading', { name: 'Will move (1)' })
    expect(within(panel).getAllByRole('heading').map((h) => h.textContent)).toEqual(
      ['Will move (1)', 'Unknown: read as will move (1)', 'Likely to move (1)', 'Unaffected (1)'])
    expect(panel).toHaveTextContent('fit.choice: a threshold moved past it')
    expect(panel).toHaveTextContent('the designer named no run record for this design')
    expect(panel).toHaveTextContent('v7 the fitter chooses among near-equal fits; moved means swapped designed/declined '
                                    + 'or moved > 1 dB; measured: declined after fit 18 of 34')
    expect(panel).toHaveTextContent('The designer expects about 3 of 4 titles to move')   // 1 + 1 (unknown) + 0.7 + 0
    expect(within(panel).getByRole('link', { name: 'Alien' })).toHaveAttribute('href', '/titles/a?out_of_date=true')
  })

  it('redesigns the titles chosen in one run, and keeps the rest, saying which could not be kept', async () => {
    const sent = sentTo()
    routes(() => check(), sent)
    renderApp('/titles?out_of_date=true')
    const panel = await screen.findByRole('region', { name: 'What a redesign would move' })

    await userEvent.click(await within(panel).findByRole('button', { name: 'Select those likely to move' }))
    expect(within(panel).getByText('3 titles selected, about 3 expected to move')).toBeInTheDocument()
    await userEvent.click(within(panel).getByRole('checkbox', { name: 'Choose Blade' }))   // a person decides otherwise
    expect(within(panel).getByText('2 titles selected, about 2 expected to move')).toBeInTheDocument()
    await userEvent.click(within(panel).getByRole('button', { name: 'Redesign selected' }))
    await waitFor(() => expect(sent.revise).toHaveLength(1))
    expect(sent.revise[0]).toMatchObject({ ids: ['a', 'h'], to: 'design', run: true })
    expect(await within(panel).findByText('Sent 2 titles back to the designer in one run.')).toBeInTheDocument()
    expect(within(panel).getByRole('checkbox', { name: 'Choose Alien' })).toBeDisabled()

    await userEvent.click(within(panel).getByRole('button', { name: 'Select the rest' }))
    await userEvent.click(within(panel).getByRole('checkbox', { name: 'Choose Blade' }))   // not in "the rest": added
    await userEvent.click(within(panel).getByRole('button', { name: 'Keep selected' }))
    await waitFor(() => expect(sent.reviewed).toEqual([{ ids: ['d', 'b'] }]))
    expect(await within(panel).findByText(/Kept 1 design at design version 7\. Not kept: b \(a run is working/))
      .toBeInTheDocument()
  })

  it('says a check made at another design version is stale', async () => {
    routes(() => check(), sentTo(), 8)
    renderApp('/titles?out_of_date=true')
    expect(await screen.findByText(/This answer is for design version 7; the designer now designs with 8/)).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Will move (1)' })).not.toBeInTheDocument()
  })

  it('says why a check failed', async () => {
    routes(() => check({ state: 'failed', error: 'the impact answer is not in the order asked', titles: [] }), sentTo())
    renderApp('/titles?out_of_date=true')
    expect(await screen.findByText('The check failed: the impact answer is not in the order asked')).toBeInTheDocument()
  })

  it('follows a check while it runs', async () => {
    const sent = sentTo()
    routes(() => check({ state: 'running', done: 500, total: 1287, titles: [] }), sent)
    renderApp('/titles?out_of_date=true')
    expect(await screen.findByText('Asked about 500 of 1,287 titles…')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Checking…' })).toBeDisabled()
  })
})
