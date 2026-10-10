import { render } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router'

import { Providers, routes } from '../App'

/** The whole app at `path`, as the router and providers have it; returns the router so a test can see where it went. */
export function renderApp(path = '/') {
  const router = createMemoryRouter(routes, { initialEntries: [path] })
  render(<Providers><RouterProvider router={router} /></Providers>)
  return router
}
