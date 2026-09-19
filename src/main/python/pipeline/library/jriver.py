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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from hamcws import MediaServer, get_mcws_connection

from pipeline.library.source import LibraryItem

logger = logging.getLogger('library_jriver')


@dataclass(frozen=True)
class BrowseNode:
    id: int
    name: str


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

    FIELDS = (
        'Filename', 'Year', 'Date Modified', 'File Size', 'Image File', 'IMDB', 'TheMovieDB',
    )
    DEFAULT_EXTERNAL_ID_FIELDS = {
        'imdb': ('IMDB', 'IMDb'),
        'tmdb': ('TheMovieDB', 'TMDB', 'TMDb'),
    }

    def __init__(self, host: str, port: int, browse_node_id: int, *, username: Optional[str] = None,
                 password: Optional[str] = None, ssl: bool = False, timeout: int = 5,
                 external_id_fields: Optional[Mapping[str, Sequence[str]]] = None):
        self.host = host
        self.port = port
        self.browse_node_id = browse_node_id
        self.username = username
        self.password = password
        self.ssl = ssl
        self.timeout = timeout
        self.external_id_fields = dict(external_id_fields or self.DEFAULT_EXTERNAL_ID_FIELDS)
        identity = json.dumps({'host': host.lower(), 'port': port, 'ssl': ssl}, sort_keys=True)
        self._server_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]

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
            rows = await server.browse_files(self.browse_node_id, list(self.FIELDS))
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
        title = name or None
        series = _value(row, 'Series')
        season = _value(row, 'Season')
        episode = _value(row, 'Episode')
        kind = 'tv' if series or season or episode else 'movie'
        meta = {'season': season} if season else {}
        modified = _value(row, 'Date Modified')
        size = _value(row, 'File Size')
        fingerprint = json.dumps({'date_modified': modified, 'file_size': size}, sort_keys=True) \
            if modified or size else ''

        return LibraryItem(
            id=f'jriver-{self._server_id}-{key}',
            source_path=filename,
            display_name=name or os.path.basename(filename),
            title=title,
            year=_value(row, 'Year') or None,
            kind=kind,
            external_ids=self._external_ids(row),
            art_path=_local_art_path(_value(row, 'Image File')),
            meta=meta,
            fingerprint=fingerprint,
        )

    def _external_ids(self, row: Mapping[str, Any]) -> dict[str, str]:
        ids = {}
        for identifier, aliases in self.external_id_fields.items():
            value = next((_value(row, field) for field in aliases if _value(row, field)), None)
            if value:
                ids[identifier] = value
        return ids


def _value(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    return str(value).strip() if value is not None else ''


def _local_art_path(value: str) -> Optional[str]:
    return value if value and value.upper() != 'INTERNAL' and os.path.isfile(value) else None
