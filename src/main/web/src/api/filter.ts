// A TitleFilter (design/pipeline-service.md §6.2) and its form in the address bar, so a filtered list can be linked to.
// The titles table's sort rides along beside it (`sortFromQuery`): it orders the list but selects nothing.
import type { Schemas } from './client'

/** What a form holds: every field optional. `cleanFilter` makes the request's shape. */
export type TitleFilter = Partial<Schemas['TitleFilter']>
export type Needs = Schemas['Needs']
export type Kind = Schemas['Kind']
export type Flag = Schemas['Flag']
export type TitleSort = Schemas['TitleSort']

export const FLAGS: readonly Flag[] = ['Ignored', 'Shadowed', 'Gone', 'Possible duplicate', 'Already in catalogue']

const NEEDS: readonly Needs[] = ['attention', 'review', 'extract', 'design', 'publish', 'commit', 'done']

export function filterFromQuery(params: URLSearchParams): TitleFilter {
  const filter: TitleFilter = {}
  const needs = params.getAll('needs').filter((n): n is Needs => (NEEDS as readonly string[]).includes(n))
  if (needs.length) filter.needs = needs
  const kind = params.get('kind')
  if (kind === 'movie' || kind === 'tv') filter.kind = kind
  for (const name of ['source', 'match', 'year', 'detail'] as const) {
    const value = params.get(name)?.trim()
    if (value) filter[name] = value
  }
  const flags = params.getAll('flags').filter((f): f is Flag => (FLAGS as readonly string[]).includes(f))
  if (flags.length) filter.flags = flags
  if (params.get('new_since_scan') === 'true') filter.new_since_scan = true
  const declined = params.get('declined')
  if (declined === 'true' || declined === 'false') filter.declined = declined === 'true'
  const ids = params.getAll('ids').filter(Boolean)
  if (ids.length) filter.ids = ids
  return filter
}

export function filterToQuery(filter: TitleFilter): URLSearchParams {
  const params = new URLSearchParams()
  for (const need of filter.needs ?? []) params.append('needs', need)
  if (filter.kind) params.set('kind', filter.kind)
  for (const name of ['source', 'match', 'year', 'detail'] as const) {
    const value = filter[name]
    if (value) params.set(name, value)
  }
  for (const flag of filter.flags ?? []) params.append('flags', flag)
  if (filter.new_since_scan) params.set('new_since_scan', 'true')
  if (typeof filter.declined === 'boolean') params.set('declined', String(filter.declined))
  for (const id of filter.ids ?? []) params.append('ids', id)
  return params
}

/** The filter with its empty fields dropped, in a request's shape: what is sent, so no field set is every title. */
export function cleanFilter(filter: TitleFilter): Schemas['TitleFilter'] {
  return { ...filterFromQuery(filterToQuery(filter)), new_since_scan: !!filter.new_since_scan }
}

export function describeFilter(filter: TitleFilter): string {
  const parts: string[] = []
  if (filter.needs?.length) parts.push(`needing ${filter.needs.join(' or ')}`)
  if (filter.kind) parts.push(filter.kind === 'movie' ? 'movies' : 'TV')
  if (filter.year) parts.push(`year ${filter.year}`)
  if (filter.source) parts.push(`from ${filter.source}`)
  if (filter.match) parts.push(`matching “${filter.match}”`)
  if (filter.detail) parts.push(`detail matching “${filter.detail}”`)
  if (filter.flags?.length) parts.push(`flagged ${filter.flags.join(' and ')}`)
  if (filter.new_since_scan) parts.push('new since the last scan')
  if (filter.declined === true) parts.push('declined by the designer')
  if (filter.declined === false) parts.push('not declined by the designer')
  if (filter.ids?.length) parts.push(`${filter.ids.length} chosen`)
  return parts.length ? parts.join(', ') : 'every title'
}

const SORTS: readonly TitleSort[] = ['title', 'year', 'kind', 'source', 'needs', 'detail', 'confidence', 'candidate_count', 'flags']

/** The column the titles table is ordered by, and which way; none is work-list order. */
export interface Sort { sort: TitleSort | null; descending: boolean }

export function sortFromQuery(params: URLSearchParams): Sort {
  const sort = params.get('sort')
  return (SORTS as readonly string[]).includes(sort ?? '')
    ? { sort: sort as TitleSort, descending: params.get('descending') === 'true' }
    : { sort: null, descending: false }
}

export function sortToQuery({ sort, descending }: Sort, params: URLSearchParams): URLSearchParams {
  if (sort) params.set('sort', sort)
  if (sort && descending) params.set('descending', 'true')
  return params
}
