// A title's .beq projects in a line (design/web-parity.md): the desktop title page's badge -- as designed, modified since
// design (a person's edit: what gets published), not written yet, or unreadable -- with what an edit means as its tip.
import type { Schemas } from '../api/client'

export function ProjectsLine({ projects }: { projects: Schemas['TitleProjects'] | undefined }) {
  if (!projects || !projects.projects.length) return null
  return (
    <p className={projects.warn ? 'warn' : 'muted'} aria-label="Projects" title={projects.note}>
      Projects: {projects.summary}
      {projects.projects.some((p) => p.edited) && ' — what is saved in it is what gets published'}
    </p>
  )
}
