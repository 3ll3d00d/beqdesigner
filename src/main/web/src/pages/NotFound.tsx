import { Link } from 'react-router'

export function NotFound() {
  return (
    <section>
      <h1>Not found</h1>
      <p className="muted">There is no such page. <Link to="/">Status</Link></p>
    </section>
  )
}
