r'''
Ignore rules -- design/library-sync/workflow-rework/design.md §12.4.

A rule says "this kind of title is not for this catalogue" -- a Kids folder, all TV, anything before 1960 -- so it need
not be skipped by hand, title by title. It is evaluated at discovery against a `LibraryItem`; a title it matches is
*Ignored*, a state derived from the rule and never stored on the title, so deleting the rule brings the title back.

A rule lists the fields it constrains, and matches when **all** of them match:

    source        the name of the source the item came from
    path          a folder prefix (`/films/Kids`) or a glob (`/films/Kids/**`, `/films/*/extras`), case-insensitive
                  and indifferent to `\` versus `/`; a prefix matches the folder and everything under it
    title         a regular expression, searched (not anchored), case-insensitive, against the item's title
    year          `1960` (equal), `<1960`, `<=1960`, `>1999`, `>=1999`, or `1990-1999` (inclusive); an item with no
                  numeric year never matches
    kind          `movie` or `tv`
    external_ids  a mapping, e.g. `{imdb: tt0113277}`; every listed id must equal the item's
    reason        optional free text, shown wherever the rule is named

Prefer curating in the library itself (a JRiver browse node) where the source has it; rules mainly serve filesystem
sources and rules that span sources.
'''
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from pipeline.library.source import LibraryItem

_FIELDS = ('source', 'path', 'title', 'year', 'kind', 'external_ids')
_KINDS = ('movie', 'tv')
_YEAR = re.compile(r'^\s*(?:(?P<op><=|>=|<|>|==|=)?\s*(?P<a>\d{4})|(?P<lo>\d{4})\s*-\s*(?P<hi>\d{4}))\s*$')
_GLOB_CHARS = '*?['


@dataclass(frozen=True)
class IgnoreRule:
    source: Optional[str] = None
    path: Optional[str] = None
    title: Optional[str] = None
    year: Optional[str] = None
    kind: Optional[str] = None
    external_ids: Optional[Tuple[Tuple[str, str], ...]] = None  # sorted pairs, so the rule is hashable
    reason: Optional[str] = None
    _title_re: Any = field(default=None, compare=False, repr=False)
    _path_re: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if not any(getattr(self, name) for name in _FIELDS):
            raise ValueError('an ignore rule must constrain at least one of: ' + ', '.join(_FIELDS))
        if self.kind is not None and self.kind not in _KINDS:
            raise ValueError(f"ignore rule kind must be one of {_KINDS}, got {self.kind!r}")
        if self.year is not None and not _YEAR.match(str(self.year)):
            raise ValueError(f"ignore rule year {self.year!r} is not a year, a comparison (<1960, >=1999) or a range "
                             f"(1990-1999)")
        if self.title is not None:
            try:
                object.__setattr__(self, '_title_re', re.compile(self.title, re.IGNORECASE))
            except re.error as error:
                raise ValueError(f"ignore rule title {self.title!r} is not a valid regular expression: {error}")
        if self.path is not None:
            object.__setattr__(self, '_path_re', _path_pattern(self.path))

    def matches(self, item: LibraryItem, source: str = '') -> bool:
        if self.source is not None and self.source != source:
            return False
        if self.kind is not None and self.kind != item.kind:
            return False
        if self._path_re is not None and not self._path_re.match(_normal(item.source_path)):
            return False
        if self._title_re is not None and not self._title_re.search(item.title or item.display_name or ''):
            return False
        if self.year is not None and not _year_matches(str(self.year), item.year):
            return False
        if self.external_ids is not None and any(str(item.external_ids.get(k, '')) != v
                                                 for k, v in self.external_ids):
            return False
        return True

    def to_config(self) -> dict:
        ''' The rule as a profile's `ignore:` entry -- the inverse of rule_from_config(). '''
        entry = {name: getattr(self, name) for name in ('source', 'path', 'title', 'year', 'kind', 'reason')
                 if getattr(self, name) is not None}
        if self.external_ids:
            entry['external_ids'] = dict(self.external_ids)
        return entry

    def describe(self) -> str:
        ''' What the rule says, for wherever a title is labelled with the rule that ignored it. '''
        parts = [f'{name} {getattr(self, name)}' for name in ('source', 'path', 'title', 'year', 'kind')
                 if getattr(self, name) is not None]
        if self.external_ids:
            parts.append('ids ' + ', '.join(f'{k}={v}' for k, v in self.external_ids))
        text = ' and '.join(parts)
        return f'{text} ({self.reason})' if self.reason else text


def _normal(path: str) -> str:
    return re.sub(r'/+', '/', path.replace('\\', '/')).rstrip('/').casefold()


def _path_pattern(pattern: str):
    normal = _normal(pattern)
    if not any(c in normal for c in _GLOB_CHARS):
        return re.compile(re.escape(normal) + r'(?:/.*)?$')  # the folder itself, or anything under it
    out, i = [], 0
    while i < len(normal):
        if normal.startswith('**/', i):
            out.append('(?:.*/)?')
            i += 3
        elif normal.startswith('**', i):
            out.append('.*')
            i += 2
        elif normal[i] == '*':
            out.append('[^/]*')
            i += 1
        elif normal[i] == '?':
            out.append('[^/]')
            i += 1
        else:
            out.append(re.escape(normal[i]))
            i += 1
    return re.compile(''.join(out) + '$')


def _year_matches(expression: str, year: Optional[str]) -> bool:
    if not year or not str(year).strip().isdigit():
        return False
    value = int(str(year).strip())
    m = _YEAR.match(expression)
    if m.group('lo'):
        return int(m.group('lo')) <= value <= int(m.group('hi'))
    limit, op = int(m.group('a')), m.group('op') or '='
    return {'<': value < limit, '<=': value <= limit, '>': value > limit, '>=': value >= limit,
            '=': value == limit, '==': value == limit}[op]


def rule_from_config(entry: Mapping[str, Any]) -> IgnoreRule:
    '''
    :param entry: one item of a profile's `ignore:` list.
    :raises ValueError: for an unknown key or an invalid value -- a typo must not silently ignore nothing.
    '''
    if not isinstance(entry, Mapping):
        raise ValueError(f'an ignore rule must be a mapping, got {entry!r}')
    unknown = set(entry) - set(_FIELDS) - {'reason'}
    if unknown:
        raise ValueError(f"unknown ignore rule key(s) {sorted(unknown)}; a rule takes {', '.join(_FIELDS)}, reason")
    ids = entry.get('external_ids')
    if ids is not None and not isinstance(ids, Mapping):
        raise ValueError(f'ignore rule external_ids must be a mapping, got {ids!r}')
    return IgnoreRule(
        source=entry.get('source'), path=entry.get('path'), title=entry.get('title'),
        year=None if entry.get('year') is None else str(entry['year']), kind=entry.get('kind'),
        external_ids=tuple(sorted((str(k), str(v)) for k, v in ids.items())) if ids else None,
        reason=entry.get('reason'))


def rules_from_config(entries: Optional[Iterable[Mapping[str, Any]]]) -> List[IgnoreRule]:
    return [rule_from_config(entry) for entry in entries or []]


def evaluate(rules: Iterable[IgnoreRule], item: LibraryItem, source: str = '') -> Optional[IgnoreRule]:
    ''' :return: the first rule (in list order) that matches, or None. '''
    return next((rule for rule in rules if rule.matches(item, source)), None)


def explain(rule: Optional[IgnoreRule]) -> str:
    ''' :return: the label for a title a rule ignored, e.g. `path /films/Kids/** (not for the catalogue)`. '''
    return f'ignored by rule: {rule.describe()}' if rule is not None else ''
