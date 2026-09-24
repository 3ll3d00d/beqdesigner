'''
What the work list's settings drawer (`model.worklist_settings`, chunk 26c) decides, with no widget in it: editing a
`Profile`'s parts, checking a folder or a repository, evaluating ignore rules against the index's rows for the live
"would ignore N titles" preview, and what makes the index out of date.

The profile file is durable state and the index a disposable cache: nothing here writes the index, and every edit is a new
`Profile` built with `dataclasses.replace`, whose `to_config()` keeps whatever the drawer does not manage (`designers:`,
the rest of `run:` and `sync:`, unknown sections).
'''
import os
import re
import subprocess
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pipeline.library.ignore import IgnoreRule, evaluate
from pipeline.library.index import TitleRow
from pipeline.library.profile import Profile
from pipeline.library.source import LibraryItem
from pipeline.publish.git import RepoTarget, is_repo, parse_github_remote

LEVEL_OK = 'ok'
LEVEL_INFO = 'info'       # fine, and something to know ("will be created")
LEVEL_ERROR = 'error'     # cannot be used: the edit is refused
LEVEL_EMPTY = 'empty'     # not set


# --- editing what a profile does not have a field for -------------------------------------------------------------------

def config_value(profile: Profile, section: str, key: str, default: Any = '') -> Any:
    ''' What the profile's file says `section.key` is. '''
    value = (profile.config.get(section) or {}).get(key)
    return default if value is None else value


def with_config(profile: Profile, section: str, key: str, value: Any) -> Profile:
    '''
    The profile with `section.key` set in the file's own config (`run.designer`, `run.tv_mode`, `sync.image_owner`...); None
    or '' removes it, and a section left empty goes with it. Everything else in the config is untouched.
    '''
    config = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in profile.config.items()}
    block = dict(config.get(section) or {})
    if value is None or value == '':
        block.pop(key, None)
    else:
        block[key] = value
    if block:
        config[section] = block
    else:
        config.pop(section, None)
    return replace(profile, config=config)


def move_item(items: Sequence, source: int, target: int) -> list:
    ''' `items` with the one at `source` moved to index `target` (in the result). '''
    moved = list(items)
    moved.insert(target, moved.pop(source))
    return moved


def unique_name(base: str, taken: Iterable[str]) -> str:
    ''' `base`, or `base-2`, `base-3`... whichever is not taken. '''
    taken = set(taken)
    if base not in taken:
        return base
    number = 2
    while f'{base}-{number}' in taken:
        number += 1
    return f'{base}-{number}'


# --- folders and repositories ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PathCheck:
    level: str
    message: str

    @property
    def usable(self) -> bool:
        return self.level != LEVEL_ERROR


def check_directory(path: str) -> PathCheck:
    '''
    Whether a folder can be the work or queue directory: it exists and is writable, or it does not yet and the nearest
    folder that does can be written to (the drawer creates it).
    '''
    path = path.strip()
    if not path:
        return PathCheck(LEVEL_EMPTY, 'Not set')
    if os.path.exists(path):
        if not os.path.isdir(path):
            return PathCheck(LEVEL_ERROR, 'This is a file, not a folder')
        if not os.access(path, os.W_OK | os.X_OK):
            return PathCheck(LEVEL_ERROR, 'This folder is not writable')
        return PathCheck(LEVEL_OK, 'Folder exists and is writable')
    parent = os.path.dirname(os.path.abspath(path))
    while parent and not os.path.exists(parent):
        higher = os.path.dirname(parent)
        if higher == parent:
            break
        parent = higher
    if os.path.isdir(parent) and os.access(parent, os.W_OK | os.X_OK):
        return PathCheck(LEVEL_INFO, 'Does not exist yet: it will be created')
    return PathCheck(LEVEL_ERROR, f'Does not exist and cannot be created (no write access under {parent})')


def check_repository(path: str) -> PathCheck:
    ''' Whether a folder is a git working tree (a clone to publish into). '''
    path = path.strip()
    if not path:
        return PathCheck(LEVEL_EMPTY, 'Not set')
    if not os.path.isdir(path):
        return PathCheck(LEVEL_ERROR, 'This folder does not exist')
    if not is_repo(RepoTarget(path)):
        return PathCheck(LEVEL_ERROR, 'This folder is not a git repository (clone the repository, then choose it)')
    return PathCheck(LEVEL_OK, 'Git repository')


def repository_location(path: str) -> Tuple[Optional[str], Optional[str], PathCheck]:
    '''Resolve a chosen publish location to ``(git root, relative folder, check)``.

    A person chooses where records/images belong, which may be anywhere below
    a clone.  The profile still stores root and relative path separately for
    the publisher, but never makes the person identify that implementation
    detail.
    '''
    path = path.strip()
    if not path:
        return '', '', PathCheck(LEVEL_EMPTY, 'Not set')
    if not os.path.isdir(path):
        return None, None, PathCheck(LEVEL_ERROR, 'This folder does not exist')
    selected = os.path.abspath(path)
    current = selected
    while True:
        dot_git = os.path.join(current, '.git')
        if (os.path.isdir(dot_git) or os.path.isfile(dot_git)) and is_repo(RepoTarget(current)):
            relative = os.path.relpath(selected, current).replace(os.sep, '/')
            return current, '' if relative == '.' else relative, PathCheck(LEVEL_OK, 'Inside git repository')
        parent = os.path.dirname(current)
        if parent == current:
            return None, None, PathCheck(LEVEL_ERROR, 'This folder must be inside a Git repository; initialize it here or choose one')
        current = parent


def check_relative_dir(text: str) -> PathCheck:
    ''' A folder inside a repository, written relative to its root (`beq/xml`); empty means the root itself. '''
    text = text.strip()
    if not text:
        return PathCheck(LEVEL_EMPTY, "The repository's top folder")
    normal = text.replace('\\', '/')
    if os.path.isabs(text) or normal.startswith('/') or re.match(r'^[A-Za-z]:', normal):
        return PathCheck(LEVEL_ERROR, 'Give a folder inside the repository, relative to its top (not a full path)')
    if '..' in normal.split('/'):
        return PathCheck(LEVEL_ERROR, "The folder must stay inside the repository (no '..')")
    return PathCheck(LEVEL_OK, 'Folder inside the repository')


def remote_owner_and_name(repo: str) -> Optional[Tuple[str, str]]:
    ''' (owner, repository) if the repository's remote is a plain github.com URL, else None (also if it has no remote). '''
    try:
        return parse_github_remote(RepoTarget(repo))
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


# --- what changes discovery -----------------------------------------------------------------------------------------------

def discovery_changed(before, after) -> bool:
    '''
    Whether the index no longer describes what a scan of `after` would say: the sources, the ignore rules, the ignored
    titles or the scan settings (the directories, the designer, the TV mode...) differ. `before`/`after` are
    `model.worklist_profile.WorkListSetup`s.
    '''
    if before.profile is None or after.profile is None:
        return before.profile is not after.profile
    return (before.profile.sources != after.profile.sources or before.profile.ignore != after.profile.ignore
            or before.profile.ignored_titles != after.profile.ignored_titles or before.settings != after.settings)


# --- ignore rules ---------------------------------------------------------------------------------------------------------

_ID_PAIR = re.compile(r'^\s*([A-Za-z0-9_.-]+)\s*[=:]\s*(\S.*?)\s*$')


def parse_external_ids(text: str) -> Dict[str, str]:
    '''
    `imdb=tt0113277, tmdb=603` (commas or new lines) as a mapping.
    :raises ValueError: for a part that is not `name=value`.
    '''
    ids: Dict[str, str] = {}
    for part in re.split(r'[,\n;]', text):
        if not part.strip():
            continue
        match = _ID_PAIR.match(part)
        if not match:
            raise ValueError(f'external ids are written name=value, e.g. imdb=tt0113277; got {part.strip()!r}')
        ids[match.group(1)] = match.group(2)
    return ids


def format_external_ids(ids: Optional[Mapping[str, str]]) -> str:
    return ', '.join(f'{k}={v}' for k, v in (ids or {}).items())


def folder_of(path: str) -> str:
    ''' The folder a title's file (or a disc folder's parent) is in, keeping the path's own separators. '''
    cut = max(path.rfind('/'), path.rfind('\\'))
    return path[:cut] if cut > 0 else path


def prefill_from_row(row: TitleRow) -> Dict[str, str]:
    ''' The values "Ignore titles like this..." offers for a work-list row: its folder, kind, year, title and source. '''
    return {'path': folder_of(row.path) if row.path else '', 'kind': row.kind or '', 'year': row.year or '',
            'title': re.escape(row.title or row.display_name or ''), 'source': row.source or ''}


def row_as_item(row: TitleRow) -> LibraryItem:
    ''' What an ignore rule is evaluated against, from what the index recorded of the title. '''
    return LibraryItem(id=row.id, source_path=row.path, display_name=row.display_name, title=row.title or None,
                       year=row.year or None, kind=row.kind, external_ids=dict(row.external_ids))


@dataclass(frozen=True)
class IgnorePreview:
    '''
    How many titles the rules (and the per-title ignores) would ignore, judged from the index's rows as the last scan left
    them -- no source is listed. `per_rule[i]` is how many titles rule `i` alone matches.
    '''
    total: int                         # titles that are candidates: not shadowed by another source, not gone
    by_rules: int
    by_id: int                         # ignored by id and not by a rule
    per_rule: Tuple[int, ...] = ()
    notes: Tuple[str, ...] = ()        # what the preview cannot judge from the index's rows
    scanned: bool = True               # False: there are no rows, so nothing can be said

    @property
    def matched(self) -> int:
        return self.by_rules + self.by_id

    def text(self) -> str:
        if not self.scanned:
            return 'No titles are known yet: Rescan, then this shows how many titles the rules would ignore.'
        text = f'Would ignore {self.matched:,} of {self.total:,} titles'
        if self.matched and self.by_id:
            text += f' ({self.by_rules:,} by rule, {self.by_id:,} by id)'
        return text + ' (from the last scan)'


def preview_rules(rows: Sequence[TitleRow], rules: Sequence[IgnoreRule], ignored_titles: Mapping[str, str],
                  source_names: Iterable[str] = ()) -> IgnorePreview:
    '''
    Evaluates `rules` against the rows exactly as discovery does against the listed items (`ignore.evaluate` over a
    `LibraryItem` made from the row and its source). Shadowed and gone titles are not counted: they are not titles.

    Some of what a rule reads is not in a row, and the preview says so instead of guessing: a title with no path, year,
    title or external id cannot be matched by a rule on that field, and a season's row is one title whose path and title are
    the season's own (discovery checks each episode).
    '''
    candidates = [row for row in rows if not row.shadowed_by and not row.gone]
    if not rows:
        return IgnorePreview(0, 0, 0, tuple(0 for _ in rules), (), scanned=False)
    per_rule = [0] * len(rules)
    by_rules = by_id = 0
    for row in candidates:
        item, source = row_as_item(row), row.source
        matched_any = False
        for i, rule in enumerate(rules):
            if rule.matches(item, source):
                per_rule[i] += 1
                matched_any = True
        if matched_any:
            by_rules += 1
        elif row.id in ignored_titles:
            by_id += 1
    return IgnorePreview(len(candidates), by_rules, by_id, tuple(per_rule),
                         tuple(_notes(candidates, rules, set(source_names))))


def _notes(candidates: Sequence[TitleRow], rules: Sequence[IgnoreRule], source_names: set) -> List[str]:
    missing = {'path': sum(1 for r in candidates if not r.path), 'title': sum(1 for r in candidates if not r.title),
               'year': sum(1 for r in candidates if not re.fullmatch(r'[0-9]{4}', (r.year or '').strip())),
               'external_ids': sum(1 for r in candidates if not r.external_ids)}
    seasons = sum(1 for r in candidates if r.unit == 'season')
    notes = []
    for number, rule in enumerate(rules, 1):
        label = f'Rule {number}'
        if rule.source is not None and source_names and rule.source not in source_names:
            notes.append(f'{label} names the source "{rule.source}", which the profile does not have, so it matches nothing.')
        for name, count in missing.items():
            if getattr(rule, name) is not None and count:
                what = {'external_ids': 'external ids', 'year': 'a four-digit year'}.get(name, name)
                notes.append(f'{label} uses {name.replace("_", " ")}: {count:,} '
                             f'{"title has" if count == 1 else "titles have"} no {what} in the index, so it cannot '
                             f'match {"it" if count == 1 else "them"}.')
        if seasons and any(getattr(rule, name) is not None for name in ('path', 'title', 'external_ids')):
            notes.append(f'{label}: {seasons:,} season row{"" if seasons == 1 else "s"} '
                         f'{"is" if seasons == 1 else "are"} checked by the season\'s own path, title and ids, not episode '
                         f'by episode as a scan does.')
    return notes


def rule_matches(rule: IgnoreRule, rows: Sequence[TitleRow]) -> int:
    ''' How many titles this one rule matches (the rule editor's live count). '''
    return preview_rules(rows, [rule], {}).per_rule[0]


def first_match(rules: Sequence[IgnoreRule], row: TitleRow) -> Optional[IgnoreRule]:
    return evaluate(rules, row_as_item(row), row.source)
