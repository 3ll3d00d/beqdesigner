// The Titles list's address-bar filter (useListParams): changes made before the page draws again must all hold. A typed
// filter's pause ending as a list is picked used to lose the typed one, each change starting from the filter last drawn.
import { act, render } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'

import { useListParams } from './TitlesPage'

function setUp(path = '/titles?match=star') {
  const seen: { update?: ReturnType<typeof useListParams>['update'] } = {}
  function Probe() {
    seen.update = useListParams().update
    return null
  }
  const router = createMemoryRouter([{ path: '/titles', element: <Probe /> }], { initialEntries: [path] })
  render(<RouterProvider router={router} />)
  return { router, seen }
}

it('keeps every change made before the page draws again', () => {
  const { router, seen } = setUp()
  act(() => {
    seen.update!((f) => ({ ...f, detail: 'Rolloff' }))
    seen.update!((f) => ({ ...f, flags: ['Gone'] }))
  })
  expect(router.state.location.search).toBe('?match=star&detail=Rolloff&flags=Gone')
})

it('takes a whole filter as it is', () => {
  const { router, seen } = setUp()
  act(() => seen.update!({ kind: 'tv' }))
  expect(router.state.location.search).toBe('?kind=tv')
})

it('starts from the address bar again after navigating', async () => {
  const { router, seen } = setUp()
  await act(() => router.navigate('/titles?kind=movie'))
  act(() => seen.update!((f) => ({ ...f, year: '2020' })))
  expect(router.state.location.search).toBe('?kind=movie&year=2020')
})
