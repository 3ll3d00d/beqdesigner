// The service's data as TanStack Query hooks: one query key per resource, so a change (a job submitted, a decision) can
// refresh exactly what it affects.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { useAuth } from '../auth/auth'
import { FINISHED_STATES } from '../format'
import { unwrap, type Schemas } from './client'
import { followJob, type JobEvent } from './events'
import { cleanFilter, type TitleFilter } from './filter'
import { readToken } from './token'

export type Job = Schemas['ScanJob'] | Schemas['RunJob'] | Schemas['AcceptJob']
export type JobState = Schemas['JobState']

export const keys = {
  status: ['status'] as const,
  jobs: (state?: JobState | null) => ['jobs', state ?? 'all'] as const,
  job: (id: string) => ['job', id] as const,
  log: (id: string) => ['log', id] as const,
  schedule: ['schedule'] as const,
  titles: (query: string) => ['titles', query] as const,
  review: (id: string) => ['review', id] as const,
  chart: (id: string, candidate: number | null) => ['chart', id, candidate] as const,
}

export function useStatus(refetchInterval = 10_000) {
  const { client } = useAuth()
  return useQuery({ queryKey: keys.status, queryFn: async () => unwrap(await client.GET('/v1/status')), refetchInterval })
}

export function useJobs(state?: JobState | null) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.jobs(state),
    queryFn: async () => unwrap(await client.GET('/v1/jobs', { params: { query: { state: state ?? undefined, limit: 100 } } })),
    refetchInterval: 10_000,
  })
}

export function useJob(id: string) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.job(id),
    queryFn: async () => unwrap(await client.GET('/v1/jobs/{job_id}', { params: { path: { job_id: id } } })) as Job,
    refetchInterval: (query) => (query.state.data && FINISHED_STATES.has(query.state.data.state) ? false : 5_000),
  })
}

export function useJobLog(id: string, enabled: boolean) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.log(id),
    queryFn: async () => unwrap(await client.GET('/v1/jobs/{job_id}/log', { params: { path: { job_id: id } } })),
    enabled,
  })
}

/**
 * A running job's events as they happen (the stream), from the start of its buffer. When the job ends, its queries and the
 * status are refreshed. `null` id follows nothing.
 */
export function useJobEvents(id: string | null, limit = 500) {
  const queries = useQueryClient()
  // kept with the job they are of, so a change of job shows none of the last one's without resetting state in the effect
  const [followed, setFollowed] = useState<{ id: string | null; events: JobEvent[]; problem: string }>(
    { id: null, events: [], problem: '' })
  useEffect(() => {
    if (!id) return
    const controller = new AbortController()
    const mine = (seen: typeof followed) => (seen.id === id ? seen : { id, events: [], problem: '' })
    followJob(id, {
      token: readToken,
      signal: controller.signal,
      onEvent: (event) => setFollowed((seen) => ({ ...mine(seen), events: [...mine(seen).events, event].slice(-limit) })),
    }).then((ended) => {
      if (!ended) return
      void queries.invalidateQueries({ queryKey: keys.job(id) })
      void queries.invalidateQueries({ queryKey: ['jobs'] })
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

export function useSchedule() {
  const { client } = useAuth()
  return useQuery({ queryKey: keys.schedule, queryFn: async () => unwrap(await client.GET('/v1/schedule')) })
}

/** After anything that queues or ends work: the lists that show it. */
export function useRefreshWork() {
  const queries = useQueryClient()
  return () => {
    void queries.invalidateQueries({ queryKey: ['jobs'] })
    void queries.invalidateQueries({ queryKey: keys.status })
    void queries.invalidateQueries({ queryKey: keys.schedule })
  }
}

export function useCancelJob() {
  const { client } = useAuth()
  const queries = useQueryClient()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async (id: string) =>
      unwrap(await client.POST('/v1/jobs/{job_id}/cancel', { params: { path: { job_id: id } } })) as Job,
    onSuccess: (job) => {
      queries.setQueryData(keys.job(job.id), job)
      refresh()
    },
  })
}

export function useSaveSchedule() {
  const { client } = useAuth()
  const queries = useQueryClient()
  return useMutation({
    mutationFn: async (body: Schemas['ScheduleUpdate']) => unwrap(await client.PUT('/v1/schedule', { body })),
    onSuccess: (schedule) => {
      queries.setQueryData(keys.schedule, schedule)
      void queries.invalidateQueries({ queryKey: keys.status })
    },
  })
}

export function useTriggerSchedule() {
  const { client } = useAuth()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async () => unwrap(await client.POST('/v1/schedule/trigger')) as Job,
    onSuccess: refresh,
  })
}

export function useSubmitScan() {
  const { client } = useAuth()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async (body: Schemas['ScanJobRequest']) => unwrap(await client.POST('/v1/jobs/scan', { body })) as Job,
    onSuccess: refresh,
  })
}

export function useSubmitRun() {
  const { client } = useAuth()
  const refresh = useRefreshWork()
  return useMutation({
    mutationFn: async (body: Schemas['RunJobRequest']) => unwrap(await client.POST('/v1/jobs/run', { body })) as Job,
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

export function useTitles(filter: TitleFilter, includeDone: boolean, offset: number) {
  const { client } = useAuth()
  const query = { ...cleanFilter(filter), include_done: includeDone, limit: PAGE_SIZE, offset }
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

export function useChart(id: string, candidate: number | null, enabled: boolean) {
  const { client } = useAuth()
  return useQuery({
    queryKey: keys.chart(id, candidate),
    queryFn: async () => unwrap(await client.GET('/v1/titles/{title_id}/chart', {
      params: { path: { title_id: id }, query: candidate === null ? {} : { candidate } } })),
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
