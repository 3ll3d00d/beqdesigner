'''
What the Review folder window (`model.worklist_review`) works out about a queue directory, with no widgets: the index row
the title page reads of an entry, where each published entry stands in git, and which accepted entries can be published from
their projects. Split out of the window so each is tested on its own, and so the window stays a window.
'''
import os
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

from pipeline.library.index import TitleRow
from pipeline.publish.catalogue import catalogue_paths
from pipeline.publish.git import RepoTarget, committed_paths, posix_path, repo_state
from pipeline.review import QueueEntry, project_paths

_NEEDS = {'pending': 'review', 'accepted': 'publish', 'published': 'done', 'skipped': 'done', 'rejected': 'done'}


def entry_row(entry: QueueEntry, commit_state: str = 'none') -> TitleRow:
    '''
    The index row the title page reads, made up from the entry alone: a title waiting for a decision `needs` review, an accepted
    one publish, and every other is done. (The page reads `needs`, `review_state`, `commit_state`, `kind`, the title and the
    year; the rest is what an index row holds and a queue directory does not know.)
    :param commit_state: where a *published* entry's files are in git (`commit_states()`); `none` for any other status.
    '''
    title = str(entry.meta.get('title') or entry.id)
    review = 'accepted' if entry.status == 'published' else entry.status
    return TitleRow(
        id=entry.id, unit='item', source='', item_id=entry.id, members=(), path='', display_name=title, title=title,
        year=str(entry.meta.get('year') or ''), kind='tv' if entry.meta.get('season') else 'movie',
        season=str(entry.meta.get('season') or ''), episodes=(), external_ids={}, fingerprint='', first_seen_generation=0,
        last_seen=0.0, also_in=(), shadowed_by='', ignored='', gone=False, duplicates=(), in_catalogue=False,
        extract_state='current', design_state='protected' if entry.status in ('accepted', 'published') else 'current',
        review_state=review, publish_state='written' if entry.status == 'published' else
        'not_written' if entry.status == 'accepted' else 'none',
        commit_state=commit_state if entry.status == 'published' else 'none',
        needs=_NEEDS.get(entry.status, 'done'), tier='human' if entry.status == 'pending' else 'done', detail='',
        state_since=0.0, confidence=entry.candidates[0].confidence if entry.candidates else None,
        candidate_count=len(entry.candidates), failure='')


def commit_states(ids: Iterable[str], xml_repo: str = '', xml_dir: str = '', image_dir: str = '') -> Dict[str, str]:
    '''
    Where each published entry's XML stands in the XML repository, in the words the index uses for `commit_state` -- and, for
    the two that matter to a revise, in the terms `pipeline.library.revise` decides by: `committed` is a file HEAD holds and the
    working tree has not touched (reopening it **keeps** the files and starts a revision); `uncommitted` is one written and
    not committed, or committed and since rewritten (reopening it **takes the files out of the working tree**, deleting them or
    putting them back as last committed). One `git status` and one `git ls-tree` for the whole repository, however many titles.
    Never raises: a repository that cannot be asked is `unknown` for every title (the revise then refuses it in its own words),
    and no repository at all is `none`, as in the index.
    '''
    ids = list(ids)
    if not ids:
        return {}
    if not xml_repo:
        return {i: 'none' for i in ids}
    target = RepoTarget(xml_repo)
    paths = {i: posix_path(catalogue_paths(i, xml_dir, image_dir)[0]) for i in ids}
    try:
        state = repo_state(target)
        if state.uncommitted is None:
            return {i: 'unknown' for i in ids}
        held = committed_paths(target, list(paths.values()))
    except (OSError, subprocess.SubprocessError):
        return {i: 'unknown' for i in ids}
    return {i: 'committed' if path in held and path not in state.uncommitted else 'uncommitted'
            for i, path in paths.items()}


def _project_files(work_dir: str, entry_id: str) -> List[str]:
    ''' The `.beq` projects of a title that are there (the mono one, and the multichannel one if it was written). '''
    directory = os.path.join(work_dir, entry_id)
    return [path for path in (os.path.join(directory, f'{entry_id}.mono.beq'),
                              os.path.join(directory, f'{entry_id}.multichannel.beq')) if os.path.isfile(path)]


def split_for_publish(work_dir: Optional[str], ids: Iterable[str]) -> Tuple[List[str], List[str], List[dict]]:
    '''
    How each accepted entry can be published from a folder, given the library's work directory:

    * **with projects** -- `<work_dir>/<id>/mono.wav` is there: publish reads the title's `.beq` projects (writing them first if
      they are missing), so a person's edit to one is what is published;
    * **without** -- nothing of the title is under the work directory (an entry Batch Extract & Design designed): published
      from its candidates, as there is no project to read;
    * **refused** -- a project file is there and its audio is not. Publish would fail reading the audio if it were asked to use
      the project (as a run does, loudly), and publishing from the candidate would leave out whatever the person changed in the
      project without a word: neither is done. The result is the one `publish_library` gives for an entry it will not publish.

    :return: (ids to publish with the work directory, ids to publish without it, one refusal per entry left out)
    '''
    with_projects: List[str] = []
    without: List[str] = []
    refused: List[dict] = []
    for entry_id in ids:
        if work_dir and os.path.isfile(os.path.join(work_dir, entry_id, 'mono.wav')):
            with_projects.append(entry_id)
            continue
        found = _project_files(work_dir, entry_id) if work_dir else []
        if not found:
            without.append(entry_id)
            continue
        wav = os.path.join(project_paths(work_dir, entry_id)[0], 'mono.wav')
        refused.append({
            'id': entry_id, 'error': 'publish_failed',
            'message': f'{os.path.basename(found[0])} is there but the audio it was made from ({wav}) is not, so it cannot be '
                       f'published from the project, and publishing the designer\'s candidate instead would leave out any '
                       f'change made in the project. Extract it again from the Library Work List, or delete the project '
                       f'file to publish the candidate.'})
    return with_projects, without, refused
