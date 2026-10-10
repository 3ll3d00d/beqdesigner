/** A screen still to come (design/web-review.md §6: W4 builds Jobs, W5 Titles and review). */
export function NotBuilt({ what }: { what: string }) {
  return (
    <section>
      <h1>{what}</h1>
      <p className="muted">Not built yet. Use the BEQDesigner app, or the service's API at <a href="/docs">/docs</a>.</p>
    </section>
  )
}
