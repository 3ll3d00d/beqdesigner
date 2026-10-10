import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw/http'

import { saveToken } from './api/token'
import { schedule } from './test/fixtures'
import { server, status, statusRoute, TOKEN } from './test/server'
import { renderApp } from './test/render'

describe('signing in', () => {
  it('sends a person who is not signed in to sign-in, then back to where they were going', async () => {
    server.use(statusRoute())
    const router = renderApp('/titles?needs=review')

    await screen.findByRole('heading', { name: 'BEQDesigner pipeline' })
    expect(router.state.location.pathname).toBe('/signin')
    await userEvent.type(screen.getByLabelText('Token'), TOKEN)
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))

    await waitFor(() => expect(router.state.location.pathname).toBe('/titles'))
    expect(router.state.location.search).toBe('?needs=review')
    expect(window.sessionStorage.getItem('beqdesigner.service.token')).toBe(TOKEN)
    expect(window.localStorage.length).toBe(0)
  })

  it('says so when the token is wrong, and keeps nothing', async () => {
    server.use(statusRoute())
    renderApp('/signin')

    await userEvent.type(await screen.findByLabelText('Token'), 'wrong')
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('The service did not accept that token.')
    expect(window.sessionStorage.length + window.localStorage.length).toBe(0)
  })

  it('remembers the token on the device when asked', async () => {
    server.use(statusRoute())
    renderApp('/signin')
    await userEvent.type(await screen.findByLabelText('Token'), TOKEN)
    await userEvent.click(screen.getByLabelText('Remember on this device'))
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))
    await screen.findByRole('navigation', { name: 'Main' })
    expect(window.localStorage.getItem('beqdesigner.service.token')).toBe(TOKEN)
  })

  it('says when the service cannot be reached', async () => {
    server.use(http.get('/v1/status', () => HttpResponse.error()))
    renderApp('/signin')
    await userEvent.type(await screen.findByLabelText('Token'), TOKEN)
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('The service could not be reached')
  })

  it('never sends a person to another site after signing in', async () => {
    server.use(statusRoute())
    const router = renderApp('/signin?next=//evil.example/')
    await userEvent.type(await screen.findByLabelText('Token'), TOKEN)
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/'))
  })
})

describe('signed in', () => {
  it('shows the status of the service it reached', async () => {
    saveToken(TOKEN, false)
    server.use(statusRoute(status({ version: '2.3.0' })), http.get('/v1/schedule', () => HttpResponse.json(schedule())))
    renderApp('/')
    expect(await screen.findByText('2.3.0')).toBeInTheDocument()
    expect(screen.getByText(/Not scanned yet/)).toBeInTheDocument()
  })

  it('goes back to sign-in when the service stops accepting the token', async () => {
    saveToken('revoked', false)
    server.use(statusRoute())
    const router = renderApp('/')
    await waitFor(() => expect(router.state.location.pathname).toBe('/signin'))
    expect(window.sessionStorage.length).toBe(0)
  })

  it('signs out', async () => {
    saveToken(TOKEN, true)
    server.use(statusRoute())
    const router = renderApp('/')
    await userEvent.click(await screen.findByRole('button', { name: 'Sign out' }))
    await waitFor(() => expect(router.state.location.pathname).toBe('/signin'))
    expect(window.localStorage.length).toBe(0)
  })
})
