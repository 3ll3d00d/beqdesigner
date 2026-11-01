// Which out-of-date designs to redesign (contract 1.6, design/redesign-scope.md step 5), against a mocked service: the
// check, its answer by tier with the reasons, and redesigning or keeping the titles chosen.
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import type { Schemas } from '../api/client'
import { saveToken } from '../api/token'
import { index, runJob, title } from '../test/fixtures'
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
             http.post('/v1/jobs', async ({ request }) => {
               sent.jobs.push(await request.json())
               return json(runJob({ id: 'j1' }), 201)
             }),
             http.post('/v1/reviewed', async ({ request }) => {
               const body = await request.json() as { ids: string[] }
               sent.reviewed.push(body)
               return json({ design_version: 7, design_settings: 's0', titles: body.ids.map((id) => (
                 id === 'b' ? { id, reviewed: false, reason: 'a run is working on this title now' } : { id, reviewed: true, reason: '' })) })
             }))
}

const sentTo = () => ({ impact: [] as unknown[], jobs: [] as unknown[], reviewed: [] as unknown[] })

describe('Which out-of-date designs to redesign', () => {
  it('asks the designer, and shows its answer by tier, most likely to move first, with the reasons', async () => {
    const sent = sentTo()
    let answer = check({ state: 'idle', titles: [], changes: [], design_version: null })
    routes(() => answer, sent)
    renderApp('/titles?out_of_date=true')

    await userEvent.click(await screen.findByRole('button', { name: 'Assess Impact' }))
    expect(sent.impact).toEqual([{ ids: [] }])
    answer = check()

    const panel = await screen.findByRole('region', { name: 'What a redesign would move' })
    const table = await within(panel).findByRole('table', { name: 'Titles a redesign could move' })
    const rows = within(table).getAllByRole('row').slice(2)   // under the labels and the filters
    expect(rows.map((r) => [...r.querySelectorAll('td')].slice(1, 5).map((td) => td.textContent))).toEqual([
      ['Alien', '1979', 'Will move', '1.00'], ['Heat', '1995', 'Unknown: read as will move', '—'],
      ['Blade', '1998', 'Likely to move', '0.70'], ['Dune', '2021', 'Unaffected', '0.00']])
    expect(panel).toHaveTextContent('fit.choice: a threshold moved past it')
    expect(panel).toHaveTextContent('the designer named no run record for this design')
    expect(within(panel).getAllByRole('listitem').find((li) => li.textContent?.startsWith('v7'))).toHaveTextContent(
      'v7 the fitter chooses among near-equal fits' + 'Measured: declined after fit 18 of 34 (53%)')
    expect(panel).toHaveTextContent('Moved means swapped designed/declined or moved > 1 dB')
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
    await waitFor(() => expect(sent.jobs).toHaveLength(1))
    expect(sent.jobs[0]).toMatchObject({ kind: 'run', name: 'Redesign 2 titles by impact', request: {
      filter: { ids: ['a', 'h'] }, through: 'design', redesign: true, revise_to: 'design', scan_first: false } })
    expect(await within(panel).findByText('Started a job to redesign 2 titles in one run.')).toBeInTheDocument()
    expect(within(panel).getByRole('link', { name: 'Follow the job' })).toHaveAttribute('href', '/jobs/j1')
    expect(within(panel).getByRole('checkbox', { name: 'Choose Alien' })).toBeDisabled()

    await userEvent.click(within(panel).getByRole('button', { name: 'Select the rest' }))
    await userEvent.click(within(panel).getByRole('checkbox', { name: 'Choose Blade' }))   // not in "the rest": added
    await userEvent.click(within(panel).getByRole('button', { name: 'Keep selected' }))
    await waitFor(() => expect(sent.reviewed).toEqual([{ ids: ['d', 'b'] }]))
    expect(await within(panel).findByText(/Kept 1 design at design version 7\. Not kept: b \(a run is working/))
      .toBeInTheDocument()
  })

  it('lists the changes newest first, one per line, saying what "moved" means once when they share it', async () => {
    const moved = 'swapped designed/declined or moved > 1 dB'
    routes(() => check({ design_version: 9, changes: [
      { design_version: 7, id: 'fit-choice', summary: 'fit choice', moved, basis: [{ class: 'no_fit', measured: 8, moved: 0 }] },
      { design_version: 9, id: 'ceiling', summary: 'a 60 Hz ceiling', moved, basis: [{ class: 'designed_all_passed', measured: 65, moved: 54 }] },
      { design_version: 8, id: 'linear', summary: 'linear scoring', moved, basis: [] },
    ] }), sentTo(), 9)
    renderApp('/titles?out_of_date=true')
    const changes = (await screen.findByText('What changed: 3 changes (v7–v9)')).closest('details')!
    const items = within(changes).getAllByRole('listitem')
    expect(items.map((li) => li.querySelector('strong')?.textContent)).toEqual(['v9', 'v8', 'v7'])
    expect(items[0]).toHaveTextContent('a 60 Hz ceiling' + 'Measured: designed all passed 54 of 65 (83%)')
    expect(items[1]).not.toHaveTextContent('Measured')
    expect(within(changes).getAllByText(`Moved means ${moved}`)).toHaveLength(1)
  })

  it('filters each column, selects all shown, and adds or drops titles by hand on top', async () => {
    const sent = sentTo()
    routes(() => check(), sent)
    renderApp('/titles?out_of_date=true')
    const panel = await screen.findByRole('region', { name: 'What a redesign would move' })
    const table = await within(panel).findByRole('table', { name: 'Titles a redesign could move' })
    const names = () => within(table).queryAllByRole('link').map((a) => a.textContent)

    await userEvent.type(within(table).getByRole('spinbutton', { name: 'Minimum probability' }), '0.5')
    expect(names()).toEqual(['Alien', 'Heat', 'Blade'])   // an unknown is read as will move, so it stays
    await userEvent.click(within(panel).getByRole('button', { name: 'Select all 3 shown' }))
    expect(within(panel).getByText('3 titles selected, about 3 expected to move')).toBeInTheDocument()

    await userEvent.click(within(panel).getByRole('button', { name: 'Clear filters' }))
    await userEvent.click(within(table).getByRole('checkbox', { name: 'Choose Dune' }))      // added by hand
    await userEvent.click(within(table).getByRole('checkbox', { name: 'Choose Heat' }))      // dropped by hand
    await userEvent.selectOptions(within(table).getByRole('combobox', { name: 'Filter tier' }), 'likely')
    expect(names()).toEqual(['Blade'])
    await userEvent.click(within(panel).getByRole('button', { name: 'Deselect shown' }))
    await userEvent.selectOptions(within(table).getByRole('combobox', { name: 'Filter tier' }), '')
    await userEvent.type(within(table).getByRole('textbox', { name: 'Filter reasons' }), 'THRESHOLD')
    expect(names()).toEqual(['Alien'])
    await userEvent.clear(within(table).getByRole('textbox', { name: 'Filter reasons' }))
    await userEvent.type(within(table).getByRole('textbox', { name: 'Filter title' }), 'zzz')
    expect(within(table).getByText('No title matches these filters.')).toBeInTheDocument()

    await userEvent.click(within(panel).getByRole('button', { name: 'Redesign selected' }))
    await waitFor(() => expect(sent.jobs).toHaveLength(1))
    expect(sent.jobs[0]).toMatchObject({ request: { filter: { ids: ['a', 'd'] }, redesign: true } })
  })

  const many = Array.from({ length: 520 }, (_, i) => ({ id: `t${i}`, title: `T${i}`, year: '2000', tier: 'likely' as const,
                                                         score: 0.6, from_design_version: 8, reasons: [] }))

  it('keeps more titles than one request takes, in batches the service takes', async () => {
    const sent = sentTo()
    routes(() => check({ titles: many }), sent)
    renderApp('/titles?out_of_date=true')
    const panel = await screen.findByRole('region', { name: 'What a redesign would move' })
    await userEvent.click(await within(panel).findByRole('button', { name: 'Select all 520 shown' }))
    await userEvent.click(within(panel).getByRole('button', { name: 'Keep selected' }))
    await waitFor(() => expect(sent.reviewed.map((b) => (b as { ids: string[] }).ids.length)).toEqual([500, 20]))
    expect(await within(panel).findByText(/Kept 520 designs/)).toBeInTheDocument()
  })

  it('redesigns more titles than a revise takes, as one job', async () => {
    const sent = sentTo()
    routes(() => check({ titles: many }), sent)
    renderApp('/titles?out_of_date=true')
    const panel = await screen.findByRole('region', { name: 'What a redesign would move' })
    await userEvent.click(await within(panel).findByRole('button', { name: 'Select all 520 shown' }))
    expect(within(panel).queryByText(/at most/)).not.toBeInTheDocument()
    await userEvent.click(within(panel).getByRole('button', { name: 'Redesign selected' }))
    await waitFor(() => expect(sent.jobs).toHaveLength(1))
    expect((sent.jobs[0] as { request: { filter: { ids: string[] } } }).request.filter.ids).toHaveLength(520)
  })

  it('says a check made at another design version is stale', async () => {
    routes(() => check(), sentTo(), 8)
    renderApp('/titles?out_of_date=true')
    expect(await screen.findByText(/This answer is for design version 7; the designer now designs with 8/)).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: 'Titles a redesign could move' })).not.toBeInTheDocument()
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
    expect(screen.getByRole('button', { name: 'Assessing…' })).toBeDisabled()
  })
})
