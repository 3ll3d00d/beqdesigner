// The app's routes (design/web-review.md §4). Everything but sign-in needs the token; the service serves the app at /ui.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { createBrowserRouter, Navigate, NavLink, Outlet, RouterProvider, useLocation, type RouteObject } from 'react-router'

import { AuthProvider, useAuth } from './auth/auth'
import { SignIn } from './auth/SignIn'
import { JobPage } from './pages/JobPage'
import { JobsPage } from './pages/JobsPage'
import { NewJobPage } from './pages/NewJobPage'
import { NotFound } from './pages/NotFound'
import { ReviewPage } from './pages/ReviewPage'
import { StatusPage } from './pages/StatusPage'
import { TitlesPage } from './pages/TitlesPage'

function Shell() {
  const { token, signOut } = useAuth()
  const location = useLocation()
  if (!token) {
    const next = encodeURIComponent(location.pathname + location.search)
    return <Navigate to={`/signin?next=${next}`} replace />
  }
  return (
    <div className="shell">
      <header>
        <span className="brand">BEQDesigner pipeline</span>
        <nav aria-label="Main">
          <NavLink to="/" end>Status</NavLink>
          <NavLink to="/jobs">Jobs</NavLink>
          <NavLink to="/titles">Titles</NavLink>
        </nav>
        <button type="button" className="link" onClick={signOut}>Sign out</button>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  )
}

export const routes: RouteObject[] = [
  { path: '/signin', element: <SignIn /> },
  {
    element: <Shell />,
    children: [
      { index: true, element: <StatusPage /> },
      { path: 'jobs', element: <JobsPage /> },
      { path: 'jobs/new', element: <NewJobPage /> },
      { path: 'jobs/:jobId', element: <JobPage /> },
      { path: 'titles', element: <TitlesPage /> },
      { path: 'titles/:titleId', element: <ReviewPage /> },
      { path: '*', element: <NotFound /> },
    ],
  },
]

export function Providers({ children, fetch }: { children: React.ReactNode; fetch?: typeof globalThis.fetch }) {
  const [queries] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: true } } }))
  return (
    <QueryClientProvider client={queries}>
      <AuthProvider fetch={fetch}>{children}</AuthProvider>
    </QueryClientProvider>
  )
}

export function App() {
  const [router] = useState(() => createBrowserRouter(routes, { basename: import.meta.env.BASE_URL.replace(/\/$/, '') }))
  return (
    <Providers>
      <RouterProvider router={router} />
    </Providers>
  )
}
