import { clearToken, readToken, saveToken } from './token'

describe('the token', () => {
  it('is kept for the tab unless the person asks to be remembered on the device', () => {
    saveToken('tab', false)
    expect(window.sessionStorage.getItem('beqdesigner.service.token')).toBe('tab')
    expect(window.localStorage.length).toBe(0)
    saveToken('device', true)
    expect(window.localStorage.getItem('beqdesigner.service.token')).toBe('device')
    expect(window.sessionStorage.length).toBe(0)
    expect(readToken()).toBe('device')
  })

  it('is cleared from both places', () => {
    saveToken('a', true)
    clearToken()
    expect(readToken()).toBeNull()
  })

  it('is kept in memory when storage refuses it', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError')
    })
    saveToken('only-here', false)
    expect(readToken()).toBe('only-here')
  })
})
