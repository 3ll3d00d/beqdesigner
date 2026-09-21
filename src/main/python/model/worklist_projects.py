'''
A title's `.beq` projects, as the title page shows them -- design/library-sync/workflow-rework/design.md §12.10, chunk 27c.

Designing a title writes a **mono** project (`<work_dir>/<id>/<id>.mono.beq`) and, where the extraction was multichannel, a
**multichannel** one (`<id>.multichannel.beq`, the designed filter linked across every channel): the same files
*File > Save Project* writes, so they open in the main window like any other project. That is where a person tunes a
filter by ear or by eye before deciding, and **what they save there is what gets published**: `publish` reads the
project's filter, not the designer's candidate (design.md §3.3.1).

The pipeline tells a person's edit from its own by a hash it stamps on the project when it writes it
(`pipeline_filter_hash`), which the main window's own save never writes. So a project is *modified since design* when its
filter no longer hashes to the stamp -- **including a project that was opened and saved again without changing anything**,
since the main window's save carries no stamp. What that means for the title, which the badge's tooltip says:

* an accepted or published title is *Publish: out of date* (the catalogue copy differs from the project's filter);
* a redesign keeps the edit (the project is not overwritten while it is modified);
* bulk accept leaves the title out (the top pick is no longer what would be published);
* if the mono and the multichannel project were both edited and now disagree, the title *needs attention*.

This module is the no-widget half: where the projects are and what state each is in (read with `read_project_filter()`, as
the pipeline does). It never raises for a file it cannot read: it says so.
'''
import os
from dataclasses import dataclass
from typing import List, Tuple

from pipeline.publish.project import read_project_filter
from pipeline.review import project_paths

MONO, MULTICHANNEL = 'mono', 'multichannel'

# badge levels
LEVEL_NEUTRAL, LEVEL_OK, LEVEL_WARN = 'neutral', 'ok', 'warn'

EDITED_TOOLTIP = ('A project is "modified since design" when its filter no longer matches what the pipeline wrote.\n'
                  'What you save in the project is what gets published, not the designer\'s candidate; a redesign keeps '
                  'your edit; bulk accept leaves this title out; and if the mono and multichannel projects were both '
                  'edited and now disagree, the title needs attention.\n'
                  'A project opened and saved again without a change also counts as modified: the main window does not '
                  'write the pipeline\'s stamp.')


@dataclass(frozen=True)
class ProjectState:
    '''
    :param kind: `mono` or `multichannel`.
    :param path: where the project is (or would be).
    :param exists: whether the file is there (it is written when the title is designed).
    :param edited: the file is there, can be read, and its filter is no longer what the pipeline wrote.
    :param error: why an existing file could not be read; empty otherwise.
    '''
    kind: str
    path: str
    exists: bool
    edited: bool = False
    error: str = ''

    @property
    def readable(self) -> bool:
        return self.exists and not self.error


def read_state(kind: str, path: str) -> ProjectState:
    ''' One project's state; a file that cannot be read is an `error`, never an exception. '''
    if not os.path.isfile(path):
        return ProjectState(kind, path, False)
    try:
        _, pure = read_project_filter(path)
    except Exception as error:   # a truncated gzip, JSON that is not a project, a missing key, a permission
        return ProjectState(kind, path, True, error=f'{type(error).__name__}: {error}')
    return ProjectState(kind, path, True, edited=not pure)


def project_states(work_dir: str, title_id: str) -> List[ProjectState]:
    '''
    The mono project always, and the multichannel one when the title has a multichannel extraction (whether or not its
    project has been written yet) or its project is there anyway. Empty without a work directory.
    '''
    if not work_dir or not title_id:
        return []
    project_dir, mono, multichannel, _ = project_paths(work_dir, title_id)
    states = [read_state(MONO, mono)]
    on_disk = os.path.join(project_dir, f'{title_id}.multichannel.beq')
    if multichannel or os.path.isfile(on_disk):
        states.append(read_state(MULTICHANNEL, multichannel or on_disk))
    return states


def badge(states: List[ProjectState]) -> Tuple[str, str]:
    '''
    :return: (text, level) for the badge beside the Open buttons.
    '''
    if not states:
        return '', LEVEL_NEUTRAL
    problems = [s for s in states if s.error]
    if problems:
        first = problems[0]
        return f'The {first.kind} project could not be read: {first.error}', LEVEL_WARN
    existing = [s for s in states if s.exists]
    if not existing:
        return 'No project yet: it is written when the title is designed.', LEVEL_NEUTRAL
    edited = [s.kind for s in existing if s.edited]
    if edited:
        return f'Modified since design: {" and ".join(edited)} project', LEVEL_WARN
    return 'Projects as designed', LEVEL_OK
