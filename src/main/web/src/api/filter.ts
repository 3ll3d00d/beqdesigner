// A TitleFilter (design/pipeline-service.md §6.2) and its form in the address bar, so a filtered list can be linked to.
import type { Schemas } from './client'

/** What a form holds: every field optional. `cleanFilter` makes the request's shape. */
export type TitleFilter = Partial<Schemas['TitleFilter']>
export type Needs = Schemas['Needs']
export type Kind = Schemas['Kind']

const NEEDS: readonly Needs[] = ['attention', 'review', 'extract', 'design', 'publish', 'commit', 'done']

export function filterFromQuery(params: URLSearchParams): TitleFilter {
  const filter: TitleFilter = {}
  const needs = params.getAll('needs').filter((n): n is Needs => (NEEDS as readonly string[]).includes(n))
  if (needs.length) filter.needs = needs
  const kind = params.get('kind')
  if (kind === 'movie' || kind === 'tv') filter.kind = kind
  for (const name of ['source', 'match', 'year'] as const) {
    const value = params.get(name)?.trim()
    if (value) filter[name] = value
  }
  if (params.get('new_since_scan') === 'true') filter.new_since_scan = true
  const ids = params.getAll('ids').filter(Boolean)
  if (ids.length) filter.ids = ids
  return filter
}

export function filterToQuery(filter: TitleFilter): URLSearchParams {
  const params = new URLSearchParams()
  for (const need of filter.needs ?? []) params.append('needs', need)
  if (filter.kind) params.set('kind', filter.kind)
  for (const name of ['source', 'match', 'year'] as const) {
    const value = filter[name]
    if (value) params.set(name, value)
  }
  if (filter.new_since_scan) params.set('new_since_scan', 'true')
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
  if (filter.new_since_scan) parts.push('new since the last scan')
  if (filter.ids?.length) parts.push(`${filter.ids.length} chosen`)
  return parts.length ? parts.join(', ') : 'every title'
}
