r'''
One catalogue from several sources -- design/library-sync/workflow-rework/design.md §12.4.

Given each source's items, in the profile's priority order, decide which are *titles*:

- **Hard clash -- the same media file.** Same path after path mapping, case-folding, `\` versus `/`, and with a disc
  rip's clips folded into the disc folder (`BDMV`, `VIDEO_TS`). It needs no disk read. One item owns the title; the
  other is *shadowed* and the owner records "also in <source>".
- **Soft clash -- the same title in a different file.** The same TMDB id, else IMDb id, else title and year (always with
  the kind and, for TV, the season and episodes). It may be a real second entry (an edition, another audio track), so it
  is **flagged, never dropped**: both remain titles and each names the others.
- **Ignored.** A title matching an ignore rule (or ignored by id) stays in the list, labelled with why; it is a state
  derived from the profile, so deleting the rule brings it back.

**Ownership is sticky.** A title's id is its work directory, its queue entry and its catalogue file name, so reordering
the sources must not change it -- that would orphan an expensive extraction and publish a second XML for one film. When
two items are the same file, the one whose id already has a work directory or queue entry (a *claim*) owns it whatever the
priority; only unclaimed clashes go to the higher-priority source. Claims are read back from those directories every
time, so they are never a second copy of the truth. If the owner's item disappears the other one takes over, **under its
own id**: nothing records which path a dangling claim belonged to, so its old outputs are left behind, not reused.
(The discovery index, which keeps a path per title, is the place to improve that.)

TV seasons have a second id shape (`some-show-s01-60ad24`, from the series title and season), so the same guarantee is
given to them separately: `Claims.season_id()` returns the id a season already has, matched by TMDB id and season, or
by title and season, so correcting a series' title in the library does not re-key its season.
'''
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from pipeline.library.ignore import IgnoreRule, evaluate, explain
from pipeline.library.profile import Profile, build_source
from pipeline.library.season import is_season_id
from pipeline.library.source import LibraryItem, LibrarySource
from pipeline.review import read_entry

_DISC_FOLDERS = ('bdmv', 'video_ts')


# --- claims ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Claims:
    '''
    What the outputs already say a title's id is.
    :param ids: every id with a queue entry or a work directory.
    :param seasons: (series key, season) -> the season id an existing queue entry has; the series key is
        `tmdb:<id>` or `title:<casefolded title>`.
    '''
    ids: FrozenSet[str] = frozenset()
    seasons: Mapping[Tuple[str, str], str] = field(default_factory=dict, compare=True, hash=False)

    def season_id(self, item: LibraryItem) -> Optional[str]:
        ''' The id already in use for `item`'s series and season, if any (see season.plan_units). '''
        if not item.season or not str(item.season).isdigit():
            return None
        season = str(int(item.season))
        for key in _series_keys(item.external_ids.get('tmdb'), item.title):
            found = self.seasons.get((key, season))
            if found:
                return found
        return None


def _series_keys(tmdb: Optional[str], title: Optional[str]) -> List[str]:
    keys = []
    if tmdb:
        keys.append(f'tmdb:{tmdb}')
    if title and title.strip():
        keys.append(f'title:{title.casefold().strip()}')
    return keys


def reconstruct_claims(work_dir: Optional[str], queue_dir: Optional[str]) -> Claims:
    '''
    Reads the claims off disk: every queue entry (`<id>.json`) and every work directory. A missing directory is
    simply no claims. Only a *season-shaped* queue entry claims a season (an episode's entry also has a season in its
    metadata, but its id is the episode's, not the season's).
    '''
    ids = set()
    seasons: Dict[Tuple[str, str], str] = {}
    if work_dir and os.path.isdir(work_dir):
        ids.update(name for name in os.listdir(work_dir)
                   if not name.startswith('.') and os.path.isdir(os.path.join(work_dir, name)))
    if queue_dir and os.path.isdir(queue_dir):
        for name in sorted(os.listdir(queue_dir)):
            if not name.endswith('.json'):
                continue
            entry_id = name[:-len('.json')]  # an entry's file is named by its id, so most need not be opened at all
            ids.add(entry_id)
            if not is_season_id(entry_id):
                continue
            try:
                entry = read_entry(queue_dir, entry_id)
            except (OSError, ValueError, TypeError):
                continue
            season = str(entry.meta.get('season') or '')
            if season.isdigit():
                for key in _series_keys(entry.meta.get('the_movie_db'), entry.meta.get('title')):
                    seasons.setdefault((key, str(int(season))), entry.id)
    return Claims(frozenset(ids), seasons)


# --- clashes -----------------------------------------------------------------------------------------------------

def clash_key(path: str) -> str:
    r'''
    What "the same media file" means: the path lower-cased, with one separator style and no trailing or doubled
    separators, and cut back to the disc folder if it runs into a `BDMV` or `VIDEO_TS` folder.
    '''
    parts = re.sub(r'/+', '/', path.replace('\\', '/')).rstrip('/').casefold().split('/')
    for i, part in enumerate(parts):
        if i and part in _DISC_FOLDERS:
            parts = parts[:i]
            break
    return '/'.join(parts)


def _soft_key(item: LibraryItem) -> Optional[Tuple]:
    ''' What makes two different files "the same title", or None if the item says too little to tell. '''
    scope = (item.kind, item.season or '', tuple(item.episodes))  # two episodes of a series are not duplicates
    for identifier in ('tmdb', 'imdb'):
        if item.external_ids.get(identifier):
            return (identifier, str(item.external_ids[identifier])) + scope
    title = (item.title or '').casefold().strip()
    return ('title', title, item.year or '') + scope if title else None


# --- the union ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class UnionTitle:
    item: LibraryItem                     # the owning source's item; item.id is the title's id
    source: str                           # the owning source's name
    also_in: Tuple[str, ...] = ()         # sources whose item for the same file was shadowed by this one
    ignored: str = ''                     # why it is ignored, or '' -- see ignore.explain()
    duplicates: Tuple[str, ...] = ()      # ids of titles that may be the same title in another file

    @property
    def id(self) -> str:
        return self.item.id


@dataclass(frozen=True)
class Shadowed:
    item: LibraryItem
    source: str
    owner: str                            # the id of the title that owns this file


@dataclass(frozen=True)
class UnionResult:
    titles: List[UnionTitle] = field(default_factory=list)      # source priority, then listing order
    shadowed: List[Shadowed] = field(default_factory=list)
    duplicates: List[Tuple[str, ...]] = field(default_factory=list)  # groups of title ids that may be one title

    def active(self) -> List[UnionTitle]:
        return [t for t in self.titles if not t.ignored]


def union_of(listings: Sequence[Tuple[str, Sequence[LibraryItem]]], *, ignore: Iterable[IgnoreRule] = (),
             ignored_titles: Optional[Mapping[str, str]] = None, claims: Claims = Claims()) -> UnionResult:
    '''
    :param listings: (source name, its items) in priority order, highest first.
    :param claims: see reconstruct_claims(); decides who owns a file two sources both have.
    '''
    rules = list(ignore)
    ignored_titles = ignored_titles or {}
    order = [(source, item) for source, items in listings for item in items]
    by_file: Dict[str, List[Tuple[str, LibraryItem]]] = {}
    for source, item in order:
        by_file.setdefault(clash_key(item.source_path), []).append((source, item))

    owners: Dict[str, Tuple[str, LibraryItem]] = {}
    shadowed: List[Shadowed] = []
    also_in: Dict[str, List[str]] = {}
    for group in by_file.values():
        owner = next((entry for entry in group if entry[1].id in claims.ids), group[0])
        owners[owner[1].id] = owner
        for source, item in group:
            if item is not owner[1]:
                shadowed.append(Shadowed(item, source, owner[1].id))
                also_in.setdefault(owner[1].id, []).append(source)

    kept = [(source, item) for source, item in order if owners.get(item.id) == (source, item)]
    titles: List[UnionTitle] = []
    for source, item in kept:
        ignored = ''
        if item.id in ignored_titles:
            ignored = 'ignored by you' + (f': {ignored_titles[item.id]}' if ignored_titles[item.id] else '')
        elif (rule := evaluate(rules, item, source)) is not None:
            ignored = explain(rule)
        titles.append(UnionTitle(item, source, tuple(also_in.get(item.id, ())), ignored))

    groups: Dict[Tuple, List[str]] = {}
    for title in titles:
        key = None if title.ignored else _soft_key(title.item)
        if key is not None:
            groups.setdefault(key, []).append(title.id)
    duplicate_groups = [tuple(ids) for ids in groups.values() if len(ids) > 1]
    flagged = {title_id: tuple(other for other in group if other != title_id)
               for group in duplicate_groups for title_id in group}
    titles = [UnionTitle(t.item, t.source, t.also_in, t.ignored, flagged.get(t.id, ())) for t in titles]
    return UnionResult(titles, shadowed, duplicate_groups)


def union_items(profile: Profile, sources: Optional[Mapping[str, LibrarySource]] = None,
                claims: Optional[Claims] = None) -> UnionResult:
    '''
    Lists every source in the profile and merges them.
    :param sources: already-built sources by name (tests, or a caller that has them); any the profile names that are
        not here are built from their settings.
    :param claims: default: read from the profile's work and queue directories.
    :raises: whatever a source raises while listing -- a source that cannot be read must not look like a source that
        is empty, so nothing is merged from a partial listing.
    '''
    built = dict(sources or {})
    listings = []
    for spec in profile.sources:
        source = built.get(spec.name) or build_source(spec.kind, spec.settings)
        listings.append((spec.name, list(source.list_items())))
    return union_of(listings, ignore=profile.ignore, ignored_titles=profile.ignored_titles,
                    claims=claims if claims is not None else reconstruct_claims(profile.work_dir, profile.queue_dir))


class UnionLibrarySource:
    '''
    A LibrarySource over a whole profile, so run_library() works on several sources as it does on one: the titles
    that are not ignored, in priority order.
    '''

    def __init__(self, profile: Profile, sources: Optional[Mapping[str, LibrarySource]] = None):
        self.profile = profile
        self.__sources = sources

    def list_items(self, **query) -> Iterable[LibraryItem]:
        if query:
            raise TypeError('a profile is configured by its sources; list_items accepts no query')
        return [title.item for title in union_items(self.profile, self.__sources).active()]
