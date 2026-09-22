'''JRiver MCWS browse-node implementation of :class:`LibrarySource`.

The selected JRiver browse node is the catalogue boundary.  This module is
deliberately Qt-free: it creates a short-lived async ``hamcws`` client, then
returns ordinary ``LibraryItem`` values to the synchronous pipeline.
'''
import asyncio
import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from hamcws import MediaServer, get_mcws_connection

from model.dvd import pseudo_file_root as dvd_pseudo_file_root
from pipeline.library.pathmap import PathMapping, translate_path, unmapped_path_problem
from pipeline.library.source import LibraryItem

logger = logging.getLogger('library_jriver')


@dataclass(frozen=True)
class BrowseNode:
    id: int
    name: str


@dataclass(frozen=True)
class LibraryField:
    name: str  # what a request and a Browse/Files reply key it by
    display_name: str
    data_type: str  # e.g. 'String', 'Integer', 'List'


# Which JRiver fields hold each external identifier, by kind of title. A library keeps these wherever its owner's
# metadata plugin put them, so this is only a default (checked against a live library: `IMDb ID` on 1239 of 1283
# films, `TheMovieDB Series ID` on 896 of 1657 shows). A TV episode's own `IMDb ID` is *not* its series' id, hence
# separate series fields. Several names for one identifier are tried in order; the first with a value wins.
DEFAULT_EXTERNAL_ID_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    'movie': {'imdb': ('IMDb ID',), 'tmdb': ('TheMovieDB Movie ID', 'TMDb ID')},
    'tv': {'imdb': ('IMDb Series ID',), 'tmdb': ('TheMovieDB Series ID',)},
}
IDENTIFIERS = ('imdb', 'tmdb')


def normalise_external_id_fields(config: Optional[Mapping[str, Any]]) -> dict[str, dict[str, tuple[str, ...]]]:
    '''
    :param config: None for the defaults; `{'movie': {'imdb': [...], 'tmdb': [...]}, 'tv': {...}}` to override per
        kind; or a flat `{'imdb': [...], 'tmdb': [...]}`, which applies to both kinds. Anything not given keeps its
        default, and an empty list switches an identifier off. A field may be one name or a list of them.
    :raises ValueError: for an unknown kind or identifier.
    '''
    result = {kind: dict(fields) for kind, fields in DEFAULT_EXTERNAL_ID_FIELDS.items()}
    config = dict(config or {})
    nested = {kind: config.pop(kind) for kind in list(config) if kind in DEFAULT_EXTERNAL_ID_FIELDS}
    if config:  # what is left is the flat form
        nested = {kind: {**config, **nested.get(kind, {})} for kind in DEFAULT_EXTERNAL_ID_FIELDS}
    for kind, overrides in nested.items():
        for identifier, names in dict(overrides).items():
            if identifier not in IDENTIFIERS:
                raise ValueError(f'unknown external id {identifier!r}; expected one of {", ".join(IDENTIFIERS)}')
            result[kind][identifier] = _names(names)
    return result


def _names(names) -> tuple[str, ...]:
    if isinstance(names, str):
        names = [names]
    return tuple(name.strip() for name in names if name and name.strip())


async def _library_fields(host, port, username, password, ssl, timeout) -> list[LibraryField]:
    connection = get_mcws_connection(host, port, username=username, password=password, ssl=ssl, timeout=timeout)
    server = MediaServer(connection)
    try:
        fields = await server.get_library_fields()
    finally:
        await server.close()
    return [LibraryField(f.name, f.display_name, f.data_type) for f in fields]


def list_library_fields(host: str, port: int, *, username: Optional[str] = None, password: Optional[str] = None,
                        ssl: bool = False, timeout: int = 10) -> list[LibraryField]:
    '''
    The fields the server's library defines (Library/Fields), so a person can say which one holds an external id.
    :raises RuntimeError: when called from a running event loop, as list_items() does.
    '''
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_library_fields(host, port, username, password, ssl, timeout))
    raise RuntimeError('list_library_fields() cannot run inside an active event loop')


def list_browse_children(host: str, port: int, node_id: int = -1, *, username: Optional[str] = None,
                         password: Optional[str] = None, ssl: bool = False, timeout: int = 5) -> list[BrowseNode]:
    '''
    The nodes directly below a browse node (-1 is the root), for choosing the node a JRiverLibrarySource reads.

    hamcws parses the Browse/Children response as {Item name: Item text}; each entry is taken to be
    (display name, node id) and an entry whose text isn't an integer is skipped. That reading is unverified
    against a real server (design/library-sync-pipeline-plan.md chunk 3), so callers must keep a way to enter an
    id by hand.
    :raises RuntimeError: when called from a running event loop, as JRiverLibrarySource.list_items() does.
    '''
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_browse_children(host, port, node_id, username, password, ssl, timeout))
    raise RuntimeError('list_browse_children() cannot run inside an active event loop')


async def _browse_children(host, port, node_id, username, password, ssl, timeout) -> list[BrowseNode]:
    connection = get_mcws_connection(host, port, username=username, password=password, ssl=ssl, timeout=timeout)
    server = MediaServer(connection)
    try:
        response = await server.browse_children(node_id)
    finally:
        await server.close()
    return _map_children(response)


def _map_children(response: Mapping[str, Any]) -> list[BrowseNode]:
    nodes = []
    for name, value in response.items():
        try:
            nodes.append(BrowseNode(int(str(value).strip()), name))
        except ValueError:
            logger.warning('Ignoring Browse/Children entry %r: %r is not a node id', name, value)
    return nodes


class JRiverLibrarySource:
    '''Maps the files below one configured MCWS browse node into library items.'''

    # Always requested. JRiver's reply keys some fields by their internal name rather than the one asked for --
    # 'Year' comes back as 'Date (year)' -- so _map_row() reads both. The external id fields are added to these
    # from external_id_fields (see requested_fields).
    FIELDS = (
        'Filename', 'Year', 'Date Modified', 'File Size', 'Image File',
    )
    DEFAULT_EXTERNAL_ID_FIELDS = DEFAULT_EXTERNAL_ID_FIELDS

    def __init__(self, host: str, port: int, browse_node_id: int, *, username: Optional[str] = None,
                 password: Optional[str] = None, ssl: bool = False, timeout: int = 5,
                 external_id_fields: Optional[Mapping[str, Any]] = None,
                 path_mappings: Sequence[PathMapping] = ()):
        '''
        :param external_id_fields: which fields hold the IMDb / TMDb ids, see normalise_external_id_fields().
        :param path_mappings: server folder -> local folder rules. JRiver reports paths as its own host sees them
            (Windows form), which this machine usually can't open; every path is translated through these, and one
            no rule contains is passed through unchanged (design/library-sync-pipeline-plan.md §11.6).
        '''
        self.path_mappings = tuple(path_mappings)
        self.host = host
        self.port = port
        self.browse_node_id = browse_node_id
        self.username = username
        self.password = password
        self.ssl = ssl
        self.timeout = timeout
        self.external_id_fields = normalise_external_id_fields(external_id_fields)
        identity = json.dumps({'host': host.lower(), 'port': port, 'ssl': ssl}, sort_keys=True)
        self._server_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]

    @property
    def requested_fields(self) -> list[str]:
        ''' The fixed fields plus every configured id field, without repeats. '''
        fields = list(self.FIELDS)
        for by_identifier in self.external_id_fields.values():
            for names in by_identifier.values():
                fields.extend(name for name in names if name not in fields)
        return fields

    def list_items(self, **query: Any) -> Iterable[LibraryItem]:
        '''Return all valid files under the configured browse node.

        Browse filtering belongs in JRiver, so source-specific ``query``
        arguments are intentionally not accepted here.
        '''
        if query:
            raise TypeError('JRiverLibrarySource is configured by browse_node_id; list_items accepts no query')
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._list_items())
        raise RuntimeError('JRiverLibrarySource.list_items() cannot run inside an active event loop')

    async def _list_items(self) -> list[LibraryItem]:
        connection = get_mcws_connection(self.host, self.port, username=self.username, password=self.password,
                                         ssl=self.ssl, timeout=self.timeout)
        server = MediaServer(connection)
        try:
            rows = await server.browse_files(self.browse_node_id, self.requested_fields)
        finally:
            await server.close()
        return self._map_rows(rows)

    def _map_rows(self, rows: Iterable[Mapping[str, Any]]) -> list[LibraryItem]:
        items: list[LibraryItem] = []
        seen_paths: dict[str, str] = {}
        for row in rows:
            item = self._map_row(row)
            prior_path = seen_paths.get(item.id)
            if prior_path is None:
                seen_paths[item.id] = item.source_path
                items.append(item)
            elif prior_path != item.source_path:
                raise ValueError(f'JRiver Key {row["Key"]!r} has conflicting filenames: '
                                 f'{prior_path!r} and {item.source_path!r}')
        return items

    def _map_row(self, row: Mapping[str, Any]) -> LibraryItem:
        key = _value(row, 'Key')
        filename = _value(row, 'Filename')
        if not key:
            raise ValueError('JRiver Browse/Files row has no Key')
        if not filename:
            raise ValueError(f'JRiver Browse/Files row {key!r} has no Filename')

        name = _value(row, 'Name')
        kind = _kind(row)
        # a TV item is titled by its series, not by the episode's name ('Chapter 3', 'The Sofa')
        series = _value(row, 'Series') if kind == 'tv' else ''
        title = series or name or None
        season = _value(row, 'Season') if kind == 'tv' and _value(row, 'Season').isdigit() else None
        episode = _value(row, 'Episode')
        episodes = (int(episode),) if kind == 'tv' and season and episode.isdigit() else ()
        modified = _value(row, 'Date Modified')
        size = _value(row, 'File Size')
        fingerprint = json.dumps({'date_modified': modified, 'file_size': size}, sort_keys=True) \
            if modified or size else ''

        disc_path = _disc_root(filename)
        source_path = translate_path(disc_path, self.path_mappings)
        return LibraryItem(
            id=f'jriver-{self._server_id}-{key}',
            source_path=source_path,
            display_name=_display_name(name, series, season, episodes) or _base_name(filename),
            title=title,
            year=_value(row, 'Year') or _value(row, 'Date (year)') or None,
            kind=kind,
            external_ids=self._external_ids(row, kind),
            art_candidates=self._art_candidates(_value(row, 'Image File'), source_path),
            fingerprint=fingerprint,
            season=season,
            episodes=episodes,
            source_path_problem=unmapped_path_problem(disc_path, self.path_mappings),
        )

    def _art_candidates(self, value: str, media_path: str) -> tuple[str, ...]:
        '''
        Where the poster might be. `Image File` is INTERNAL (JRiver-managed, not a file), an absolute path on the
        server, or -- as seen on a live server for most titles -- a bare file name that lives beside the media file.
        Nothing here touches the disk: listing a whole library must cost one request and no `stat` per title
        (design.md §12.5), so which candidate exists is decided at design time, by artwork.resolve_art().
        '''
        if not value or value.upper() == 'INTERNAL':
            return ()
        translated = translate_path(value, self.path_mappings)
        candidates = [translated, os.path.join(os.path.dirname(media_path), value)]
        return tuple(dict.fromkeys(c for c in candidates if os.path.isabs(c)))

    def _external_ids(self, row: Mapping[str, Any], kind: str) -> dict[str, str]:
        ids = {}
        for identifier, names in self.external_id_fields[kind].items():
            value = next((v for v in (_id_value(row, name) for name in names) if v), None)
            if value:
                ids[identifier] = value
        return ids


def _kind(row: Mapping[str, Any]) -> str:
    '''
    'tv' or 'movie'. JRiver's Media Sub Type is authoritative when it is set ('Movie', 'TV Show', ...).
    A film franchise carries a Series value (and occasionally a stray Season/Episode) without being episodic,
    so those fields only decide when there is no sub type, and then only together.
    '''
    sub_type = _value(row, 'Media Sub Type').lower()
    if sub_type:
        return 'tv' if 'tv' in sub_type else 'movie'
    return 'tv' if _value(row, 'Series') and _value(row, 'Season') and _value(row, 'Episode') else 'movie'


def _display_name(name: str, series: str, season: Optional[str], episodes: tuple) -> str:
    ''' 'Series S01E03 name' for an episode, else just the item's own name. '''
    if series and season and episodes:
        return f"{series} S{int(season):02d}E{episodes[0]:02d}" + (f" {name}" if name else '')
    return name


def _id_value(row: Mapping[str, Any], field: str) -> str:
    ''' An identifier, where JRiver's numeric fields report an unset one as 0. '''
    value = _value(row, field)
    return '' if value in ('', '0') else value


def _value(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    return str(value).strip() if value is not None else ''


_BLURAY_PSEUDO_FILE = re.compile(r'^(?P<root>.+?)[\\/]BDMV[\\/]index\.bluray(?:3d)?;\d+$', re.IGNORECASE)


def _disc_root(filename: str) -> str:
    '''
    JRiver names a disc rip by a pseudo-file rather than a file: `<disc>\\BDMV\\index.bluray;1` (or `index.bluray3d`)
    for a Blu-ray, `<disc>\\VIDEO_TS\\VIDEO_TS.dvd;1` for a DVD. The disc folder is what the pipeline can open (it
    picks the main title itself), so report that. A `BDMV\\PLAYLIST\\index.bluray;N` entry, which names a playlist
    on the disc, is left as reported: which title `N` selects is unknown.
    '''
    match = _BLURAY_PSEUDO_FILE.match(filename)
    if match:
        root = match.group('root')
        # a disc at a drive's top level: `W:` alone means "the current folder on W:", so keep the separator
        return root + filename[len(root)] if root.endswith(':') else root
    return dvd_pseudo_file_root(filename) or filename


def _base_name(path: str) -> str:
    ''' The last component of a path in either separator style (the server's paths are often Windows ones). '''
    return path.replace('\\', '/').rstrip('/').rpartition('/')[2]
