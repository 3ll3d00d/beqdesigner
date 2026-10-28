// The service's data as TanStack Query hooks: one query key per resource, so a change (a job submitted, a decision) can
// refresh exactly what it affects.
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

import { useAuth } from '../auth/auth'
import { FINISHED_STATES } from '../format'
import { unwrap, type Schemas } from './client'
import { followExecution, type JobEvent } from './events'
import { cleanFilter, type Sort, type TitleFilter } from './filter'
import { readToken } from './token'

export type Job = Schemas['ScanJob'] | Schemas['RunJob'] | Schemas['AcceptJob']
export type JobState = Schemas['JobState']
export type Execution = Schemas['ScanExecution'] | Schemas['RunExecution'] | Schemas['AcceptExecution']
export type ExecutionState = Schemas['ExecutionState']
type JobCreate = Schemas['ScanJobCreate'] | Schemas['RunJobCreate'] | Schemas['AcceptJobCreate']
/** How a job is made beside what it does: its name, `every` N minutes (null: once) and its window (null: the default). */
export type Timing = { name?: string; every?: number | null; window?: string[] | null }

export const keys = {
  status: ['status'] as const,
  jobs: (state?: JobState | null) => ['jobs', state ?? 'all'] as const,
  job: (id: string) => ['job', id] as const,
  executions: (job?: string | null, state?: ExecutionState | null) => ['executions', job ?? 'all', state ?? 'all'] as const,
  execution: (id: string) => ['execution', id] as const,
  log: (id: string) => ['log', id] as const,
  titles: (query: string) => ['titles', query] as const,
  review: (id: string) => ['review', id] as const,
  chart: (id: string, query: Record<string, unknown>) => ['chart', id, query] as const,
  impact: ['impact'] as const,
  window: ['job-window'] as const,
}

export function useStatus(refetchInterval = 10_000) {
  const { client } = useAuth()
  return useQuery({ queryKey: keys.status, queryFn: async () => unwrap(await client.GET('/v1/status')), refetchInterval })
}

/** The jobs (what is done, and when), newest first. */
export function useJobs(state?: JobState | null) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.jobs(state),
    queryFn: async () => unwrap(await client.GET('/v1/jobs', { params: { query: { state: state ?? undefined } } }))
      .jobs as Job[],
    refetchInterval: 10_000,
  })
}

export function useJob(id: string) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.job(id),
    queryFn: async () => unwrap(await client.GET('/v1/jobs/{job_id}', { params: { path: { job_id: id } } })) as Job,
    refetchInterval: (query) => (query.state.data && ['done', 'removed'].includes(query.state.data.state) ? false : 5_000),
  })
}

/** The log: the running execution, each job's next one, and every one kept, newest first. */
export function useExecutions(job?: string | null, state?: ExecutionState | null) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.executions(job, state),
    queryFn: async () => unwrap(await client.GET('/v1/executions', {
      params: { query: { job: job ?? undefined, state: state ?? undefined, limit: 100 } } })),
    refetchInterval: 10_000,
  })
}

export function useExecution(id: string) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.execution(id),
    queryFn: async () => unwrap(await client.GET('/v1/executions/{execution_id}',
                                                 { params: { path: { execution_id: id } } })) as Execution,
    refetchInterval: (query) => (query.state.data && FINISHED_STATES.has(query.state.data.state) ? false : 5_000),
  })
}

/** An execution's titles, each with its outcome and stage times. */
export function useExecutionTitles(id: string, enabled: boolean) {
  const { client } = useAuth()
  return useQuery({
    queryKey: ['execution-titles', id],
    queryFn: async () => unwrap(await client.GET('/v1/executions/{execution_id}/titles',
                                                 { params: { path: { execution_id: id } } })).titles,
    enabled,
  })
}

export function useExecutionLog(id: string, enabled: boolean) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.log(id),
    queryFn: async () => unwrap(await client.GET('/v1/executions/{execution_id}/log',
                                                 { params: { path: { execution_id: id } } })),
    enabled,
  })
}

/**
 * A running execution's events as they happen (the stream), from the start of its buffer. When it ends, its queries and
 * the status are refreshed. `null` id follows nothing.
 */
export function useExecutionEvents(id: string | null, limit = 500) {
  const queries = useQueryClient()
  // kept with the execution they are of, so a change of one shows none of the last one's without resetting state
  const [followed, setFollowed] = useState<{ id: string | null; events: JobEvent[]; problem: string }>(
    { id: null, events: [], problem: '' })
  useEffect(() => {
    if (!id) return
    const controller = new AbortController()
    const mine = (seen: typeof followed) => (seen.id === id ? seen : { id, events: [], problem: '' })
    followExecution(id, {
      token: readToken,
      signal: controller.signal,
      onEvent: (event) => setFollowed((seen) => ({ ...mine(seen), events: [...mine(seen).events, event].slice(-limit) })),
    }).then((ended) => {
      if (!ended) return
      void queries.invalidateQueries({ queryKey: keys.execution(id) })
      void queries.invalidateQueries({ queryKey: ['executions'] })
      void queries.invalidateQueries({ queryKey: ['jobs'] })
      void queries.invalidateQueries({ queryKey: ['job'] })
      void queries.invalidateQueries({ queryKey: keys.status })
      void queries.invalidateQueries({ queryKey: ['titles'] })
    }, (error: unknown) => {
      if (!controller.signal.aborted) {
        setFollowed((seen) => ({ ...mine(seen), problem: error instanceof Error ? error.message : String(error) }))
      }
    })
    return () => controller.abort()
  }, [id, limit, queries])
  return followed.id === id && id ? { events: followed.events, problem: followed.problem } : { events: [], problem: '' }
}

/** After anything that makes, changes or ends work: the lists that show it. */
export function useRefreshWork() {
  const queries = useQueryClient()
  return () => {
    void queries.invalidateQueries({ queryKey: ['jobs'] })
    void queries.invalidateQueries({ queryKey: ['job'] })
    void queries.invalidateQueries({ queryKey: ['executions'] })
    void queries.invalidateQueries({ queryKey: keys.status })
  }
}

export function useCancelExecution() {
  const { client } = useAuth()
  const queries = useQueryClient()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async (id: string) =>
      unwrap(await client.POST('/v1/executions/{execution_id}/cancel', { params: { path: { execution_id: id } } })) as Execution,
    onSuccess: (execution) => {
      queries.setQueryData(keys.execution(execution.id), execution)
      refresh()
    },
  })
}

export function useDefaultWindow() {
  const { client } = useAuth()
  return useQuery({ queryKey: keys.window, queryFn: async () => unwrap(await client.GET('/v1/job-window')),
                    refetchInterval: 60_000 })
}

export function useSaveDefaultWindow() {
  const { client } = useAuth()
  const queries = useQueryClient()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async (body: Schemas['DefaultWindowUpdate']) => unwrap(await client.PUT('/v1/job-window', { body })),
    onSuccess: (saved) => {
      queries.setQueryData(keys.window, saved)
      refresh()   // a job given no window follows it
    },
  })
}

/** What a job body says beside its request: only what was given, so a window left out is the default. */
function timing({ name, every, window }: Timing) {
  return { ...(name ? { name } : {}), ...(every ? { every } : {}), ...(window !== undefined && window !== null ? { window } : {}) }
}

export function useCreateJob() {
  const { client } = useAuth()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async (body: JobCreate) => unwrap(await client.POST('/v1/jobs', { body })) as Job,
    onSuccess: refresh,
  })
}

/** A scan job: once, unless `every` says how often. */
export function useSubmitScan() {
  const create = useCreateJob()
  return {
    ...create,
    mutate: ({ name, every, window, ...request }: Schemas['ScanJobRequest'] & Timing,
             options?: Parameters<typeof create.mutate>[1]) =>
      create.mutate({ kind: 'scan', request, ...timing({ name, every, window }) } as JobCreate, options),
  }
}

/** A run job: once, unless `every` says how often (only a run through extract or design recurs). */
export function useSubmitRun() {
  const create = useCreateJob()
  return {
    ...create,
    mutate: ({ name, every, window, ...request }: Schemas['RunJobRequest'] & Timing,
             options?: Parameters<typeof create.mutate>[1]) =>
      create.mutate({ kind: 'run', request, ...timing({ name, every, window }) } as JobCreate, options),
  }
}

export function useEditJob() {
  const { client } = useAuth()
  const queries = useQueryClient()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async ({ id, ...body }: Schemas['JobUpdate'] & { id: string }) =>
      unwrap(await client.PATCH('/v1/jobs/{job_id}', { params: { path: { job_id: id } }, body })) as Job,
    onSuccess: (job) => {
      queries.setQueryData(keys.job(job.id), job)
      refresh()
    },
  })
}

/** Pause, resume, run now (a recurring job) or remove a job. */
export function useJobAction() {
  const { client } = useAuth()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async ({ id, action }: { id: string; action: 'pause' | 'resume' | 'run' | 'remove' }) => {
      const path = { params: { path: { job_id: id } } }
      switch (action) {
        case 'pause': return unwrap(await client.POST('/v1/jobs/{job_id}/pause', path))
        case 'resume': return unwrap(await client.POST('/v1/jobs/{job_id}/resume', path))
        case 'run': return unwrap(await client.POST('/v1/jobs/{job_id}/run', path))
        case 'remove': return unwrap(await client.DELETE('/v1/jobs/{job_id}', path))
      }
    },
    onSuccess: refresh,
  })
}

export function usePlan() {
  const { client } = useAuth()
  return useMutation({
    mutationFn: async (body: Schemas['RunJobRequest']) => unwrap(await client.POST('/v1/plan', { body })),
  })
}

// --- titles and review (W5) -----------------------------------------------------------------------------------------------

export const PAGE_SIZE = 100

export function useTitles(filter: TitleFilter, includeDone: boolean, offset: number, order: Sort = { sort: null, descending: false }) {
  const { client } = useAuth()
  const query = { ...cleanFilter(filter), include_done: includeDone, limit: PAGE_SIZE, offset,
                  ...(order.sort ? { sort: order.sort, descending: order.descending } : {}) }
  return useQuery({
    queryKey: keys.titles(JSON.stringify(query)),
    queryFn: async () => unwrap(await client.GET('/v1/titles', { params: { query } })),
    placeholderData: (previous) => previous,
  })
}

export function useReview(id: string) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.review(id),
    queryFn: async () => unwrap(await client.GET('/v1/titles/{title_id}/review', { params: { path: { title_id: id } } })),
    retry: false,
  })
}

/** What the chart is asked for: one design (or none), or every design; the smoothing; each design's own response. */
export interface ChartAsk {
  candidate: number | null
  all?: boolean
  /** Several designs by index (more than one: sent as `candidates`, else as `candidate`). */
  candidates?: number[]
  smoothing?: string
  responses?: boolean
  /** An earlier design run (Review.variants) whose designs these are; absent or null: the current run's. */
  variant?: string | null
}

export function chartQuery(ask: ChartAsk) {
  return {
    ...(ask.variant ? { variant: ask.variant } : {}),
    ...(ask.all ? { candidates: 'all' } : ask.candidates && ask.candidates.length > 1 ? { candidates: ask.candidates.join(',') }
      : ask.candidate === null ? {} : { candidate: ask.candidate }),
    ...(ask.smoothing && ask.smoothing !== 'none' ? { smoothing: ask.smoothing } : {}),
    ...(ask.responses ? { responses: true } : {}),
  }
}

export function useChart(id: string, ask: ChartAsk, enabled: boolean) {
  const { client } = useAuth()
  const query = chartQuery(ask)
  return useQuery({
    queryKey: keys.chart(id, query),
    queryFn: async () => unwrap(await client.GET('/v1/titles/{title_id}/chart', {
      params: { path: { title_id: id }, query } })),
    enabled,
    placeholderData: (previous) => previous,
    staleTime: 60_000,
  })
}

/** The next title waiting for a decision after `after` in the filtered list, or null if there is none. */
export function useNextTitle() {
  const { client } = useAuth()
  return async (filter: TitleFilter, after?: string): Promise<Schemas['NextTitle'] | null> => {
    const result = await client.GET('/v1/review/next', { params: { query: { ...cleanFilter(filter), after } } })
    if (result.response.status === 204) return null
    return unwrap(result)
  }
}

export function useDecide(id: string) {
  const { client } = useAuth()
  const queries = useQueryClient()
  return useMutation({
    mutationFn: async (body: Schemas['Decision']) =>
      unwrap(await client.POST('/v1/titles/{title_id}/decision', { params: { path: { title_id: id } }, body })),
    onSuccess: (review) => queries.setQueryData(keys.review(id), review),
    onError: () => void queries.invalidateQueries({ queryKey: keys.review(id) }),   // show what is there now
    onSettled: () => {
      void queries.invalidateQueries({ queryKey: ['titles'] })
      void queries.invalidateQueries({ queryKey: keys.status })
    },
  })
}

// --- editing a title (design/web-parity.md WP4) ------------------------------------------------------------------------

/** The metadata fields a title page edits, in order: they never change while the service runs. */
export function useMetadataFields() {
  const { client } = useAuth()
  return useQuery({
    queryKey: ['metadata-fields'],
    queryFn: async () => unwrap(await client.GET('/v1/metadata/fields')),
    staleTime: Infinity,
  })
}

/** A write that answers with the title's Review: it replaces the one shown, and the list learns of it. */
function useReviewWrite<T>(id: string, write: (body: T) => Promise<Schemas['Review']>) {
  const queries = useQueryClient()
  return useMutation({
    mutationFn: write,
    onSuccess: (review) => queries.setQueryData(keys.review(id), review),
    onError: () => void queries.invalidateQueries({ queryKey: keys.review(id) }),   // show what is there now
    onSettled: () => void queries.invalidateQueries({ queryKey: ['titles'] }),
  })
}

export function useEditMetadata(id: string) {
  const { client } = useAuth()
  return useReviewWrite(id, async (body: Schemas['MetadataEdit']) =>
    unwrap(await client.PATCH('/v1/titles/{title_id}/metadata', { params: { path: { title_id: id } }, body })))
}

export function useTmdbLookup(id: string) {
  const { client } = useAuth()
  return useMutation({
    mutationFn: async (body: Schemas['TmdbLookup']) =>
      unwrap(await client.POST('/v1/titles/{title_id}/metadata/tmdb', { params: { path: { title_id: id } }, body })),
  })
}

export function useSetArtwork(id: string) {
  const { client } = useAuth()
  return useReviewWrite(id, async (body: Schemas['ArtworkChoice']) =>
    unwrap(await client.PUT('/v1/titles/{title_id}/artwork', { params: { path: { title_id: id } }, body })))
}

export function useClearArtwork(id: string) {
  const { client } = useAuth()
  return useReviewWrite(id, async (digest: string) =>
    unwrap(await client.DELETE('/v1/titles/{title_id}/artwork', { params: { path: { title_id: id }, query: { digest } } })))
}

/**
 * The title's poster as an object URL (an <img> cannot send the token), fetched again when `version` changes; null while
 * there is none. The URL is revoked when it is replaced or the page goes.
 */
export function useArtworkUrl(id: string, version: string | null) {
  const { client } = useAuth()
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (version === null) return
    let made: string | null = null
    let live = true
    void client.GET('/v1/titles/{title_id}/artwork', { params: { path: { title_id: id } }, parseAs: 'blob' })
      .then((result) => {
        if (!live || !result.response.ok || !(result.data instanceof Blob)) return
        made = URL.createObjectURL(result.data)
        setUrl(made)
      })
      .catch(() => {})
    return () => {
      live = false
      if (made) URL.revokeObjectURL(made)
      setUrl(null)
    }
  }, [client, id, version])
  return version === null ? null : url
}

// --- revise and the audio stream (design/web-parity.md WP5) -------------------------------------------------------------

/** Send titles back; with `run`, the answer carries the job doing it. The list, the status and each review learn of it. */
export function useRevise() {
  const { client } = useAuth()
  const queries = useQueryClient()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async (body: Schemas['ReviseRequest']) => unwrap(await client.POST('/v1/revise', { body })),
    onSuccess: (_outcome, body) => {
      refresh()
      void queries.invalidateQueries({ queryKey: ['titles'] })
      for (const id of body.ids) void queries.invalidateQueries({ queryKey: keys.review(id) })
    },
  })
}

// --- redesign scope (contract 1.6, design/redesign-scope.md) ------------------------------------------------------------

/** The last check of what a redesign now would move, followed every two seconds while it runs. */
export function useImpact() {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.impact,
    queryFn: async () => unwrap(await client.GET('/v1/impact')),
    refetchInterval: (query) => (query.state.data?.state === 'running' ? 2_000 : false),
  })
}

/** Ask the designer about the out-of-date designs (or `ids`); the check runs in the background. */
export function useStartImpact() {
  const { client } = useAuth()
  const queries = useQueryClient()
  return useMutation({
    mutationFn: async (ids: string[] = []) => unwrap(await client.POST('/v1/impact', { body: { ids } })),
    onSuccess: (check) => queries.setQueryData(keys.impact, check),
  })
}

/** Keep these titles' designs at the identity a redesign now would use: they stop showing as out of date. */
export function useKeepReviewed() {
  const { client } = useAuth()
  const queries = useQueryClient()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async (ids: string[]) => unwrap(await client.POST('/v1/reviewed', { body: { ids } })),
    onSuccess: () => {
      refresh()
      void queries.invalidateQueries({ queryKey: ['titles'] })
    },
  })
}

export function useStreams(id: string, enabled: boolean) {
  const { client } = useAuth()
  return useQuery({
    queryKey: ['streams', id],
    queryFn: async () => unwrap(await client.GET('/v1/titles/{title_id}/streams', { params: { path: { title_id: id } } })),
    enabled,
    retry: false,
  })
}

export function useChooseStream(id: string) {
  const { client } = useAuth()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async (audioStream: number) => unwrap(await client.PUT('/v1/titles/{title_id}/stream', {
      params: { path: { title_id: id } }, body: { audio_stream: audioStream } })) as Job,   // a one-off job
    onSuccess: refresh,
  })
}

// --- what a title's audio looks like (design/web-parity.md WP2) ----------------------------------------------------------

/** A drawing the service renders off the request: `pending` while it answers 202, then the bytes. */
type Rendered<T> = { pending: true } | { pending: false; data: T }

function renderedQuery<T>(key: unknown[], fetchIt: () => Promise<{ response: Response; data?: unknown; error?: unknown }>,
                          enabled: boolean) {
  return {
    queryKey: key,
    queryFn: async (): Promise<Rendered<T>> => {
      const result = await fetchIt()
      if (result.response.status === 202) return { pending: true }
      return { pending: false, data: unwrap(result) as T }
    },
    enabled,
    retry: false,
    staleTime: 5 * 60_000,
    refetchInterval: (query: { state: { data?: Rendered<T> } }) => (query.state.data?.pending ? 2000 : false),
  }
}

/** A title's spectrogram grid, as measured or after a design (`decodeGrid()` reads it). */
export function useSpectrogram(id: string, candidate: number, filtered: boolean, maxHz: number, enabled: boolean,
                               variant: string | null = null) {
  const { client } = useAuth()
  return useQuery(renderedQuery<ArrayBuffer>(['spectrogram', id, variant, candidate, filtered, maxHz], () =>
    client.GET('/v1/titles/{title_id}/spectrogram', {
      params: { path: { title_id: id }, query: { candidate, filtered, max_hz: maxHz, ...(variant ? { variant } : {}) } },
      parseAs: 'arrayBuffer' }), enabled))
}

/** The publication heatmap of a design, as an object URL (revoked when replaced or the page goes). */
export function useHeatmapUrl(id: string, candidate: number, enabled: boolean, variant: string | null = null) {
  const { client } = useAuth()
  const query = useQuery(renderedQuery<Blob>(['heatmap', id, variant, candidate], () =>
    client.GET('/v1/titles/{title_id}/heatmap.png', {
      params: { path: { title_id: id }, query: { candidate, ...(variant ? { variant } : {}) } }, parseAs: 'blob' }), enabled))
  const blob = query.data && !query.data.pending ? query.data.data : null
  const url = useMemo(() => (blob ? URL.createObjectURL(blob) : null), [blob])
  useEffect(() => () => { if (url) URL.revokeObjectURL(url) }, [url])
  return { ...query, url, pending: !!query.data?.pending }
}

// --- published BEQs (design/web-parity.md WP3) -----------------------------------------------------------------------------

export function useCatalogue(id: string, enabled: boolean) {
  const { client } = useAuth()
  return useQuery({
    queryKey: ['catalogue', id],
    queryFn: async () => unwrap(await client.GET('/v1/titles/{title_id}/catalogue', { params: { path: { title_id: id } } })),
    enabled,
    retry: false,
    staleTime: 10 * 60_000,
  })
}

/** Each chosen published BEQ's response and the title's curves under it, in the order chosen. */
export function usePublishedCharts(id: string, indexes: number[], smoothing: string) {
  const { client } = useAuth()
  return useQueries({
    queries: indexes.map((index) => ({
      queryKey: ['catalogue-chart', id, index, smoothing],
      queryFn: async () => unwrap(await client.GET('/v1/titles/{title_id}/catalogue/{index}/chart', {
        params: { path: { title_id: id, index }, query: smoothing === 'none' ? {} : { smoothing } } })),
      staleTime: 10 * 60_000,
    })),
  })
}

// --- design variants (design/web-parity.md WP7) ------------------------------------------------------------------------------

/** Designs of the title's other design runs (`all`, or by index), after it: for drawing them beside the highlighted one. */
export function useVariantCharts(id: string, runs: { variant: string; candidates: number[] | 'all' }[], smoothing: string,
                                 responses = false) {
  const { client } = useAuth()
  return useQueries({
    queries: runs.map(({ variant, candidates }) => {
      const designs = candidates === 'all' ? 'all' : candidates.join(',')
      return {
        queryKey: ['variant-chart', id, variant, designs, smoothing, responses],
        queryFn: async () => unwrap(await client.GET('/v1/titles/{title_id}/chart', {
          params: { path: { title_id: id },
                    query: { variant, candidates: designs, ...(smoothing === 'none' ? {} : { smoothing }),
                             ...(responses ? { responses: true } : {}) } } })),
        staleTime: 60_000,
      }
    }),
  })
}

export function useUseVariant(id: string) {
  const { client } = useAuth()
  const queries = useQueryClient()
  return useMutation({
    mutationFn: async ({ variant, digest }: { variant: string; digest: string }) =>
      unwrap(await client.POST('/v1/titles/{title_id}/variants/{variant_id}/use', {
        params: { path: { title_id: id, variant_id: variant } }, body: { digest } })),
    onSuccess: (review) => queries.setQueryData(keys.review(id), review),
    onError: () => void queries.invalidateQueries({ queryKey: keys.review(id) }),
    onSettled: () => {
      void queries.invalidateQueries({ queryKey: ['titles'] })
      void queries.invalidateQueries({ queryKey: ['chart', id] })
    },
  })
}

export function useDiscardVariant(id: string) {
  const { client } = useAuth()
  const queries = useQueryClient()
  return useMutation({
    mutationFn: async ({ variant, digest }: { variant: string; digest: string }) =>
      unwrap(await client.DELETE('/v1/titles/{title_id}/variants/{variant_id}', {
        params: { path: { title_id: id, variant_id: variant }, query: { digest } } })),
    onSuccess: (review) => queries.setQueryData(keys.review(id), review),
    onError: () => void queries.invalidateQueries({ queryKey: keys.review(id) }),
  })
}

/** Save a person's edit of a design: a new design run (design/web-parity.md WP8). */
export function useSaveEdited(id: string) {
  const { client } = useAuth()
  const queries = useQueryClient()
  return useMutation({
    mutationFn: async (body: Schemas['EditedDesign']) =>
      unwrap(await client.PUT('/v1/titles/{title_id}/edited', { params: { path: { title_id: id } }, body })),
    onSuccess: (review) => queries.setQueryData(keys.review(id), review),
    onError: () => void queries.invalidateQueries({ queryKey: keys.review(id) }),
    onSettled: () => {
      void queries.invalidateQueries({ queryKey: ['titles'] })
      void queries.invalidateQueries({ queryKey: ['chart', id] })
    },
  })
}

/** How many titles one revise may send back at once (the service's MAX_REDESIGN_IDS). */
export const MAX_REVISED = 500

/** Every title id the filter matches, page by page, up to MAX_REVISED: for selecting a whole list, not one page. */
export function useMatchingIds() {
  const { client } = useAuth()
  return async (filter: TitleFilter, includeDone: boolean): Promise<{ ids: string[]; total: number }> => {
    const ids: string[] = []
    let total = 0
    for (let offset = 0; offset < MAX_REVISED; offset += PAGE_SIZE) {
      const page = unwrap(await client.GET('/v1/titles', {
        params: { query: { ...cleanFilter(filter), include_done: includeDone, limit: PAGE_SIZE, offset } } }))
      total = page.total
      ids.push(...page.titles.map((t) => t.id))
      if (offset + PAGE_SIZE >= page.total) break
    }
    return { ids: ids.slice(0, MAX_REVISED), total }
  }
}

/** A title's .beq projects and whether a person edited them since the design (the desktop title page's badge). */
export function useProjects(id: string, version: string | undefined) {
  const { client } = useAuth()
  return useQuery({
    queryKey: ['projects', id, version],
    queryFn: async () => unwrap(await client.GET('/v1/titles/{title_id}/projects', { params: { path: { title_id: id } } })),
    retry: false,
  })
}
