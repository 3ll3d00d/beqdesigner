'''
Reading the local XML catalogue repo for the TMDB ids it already holds -- design/library-sync/workflow-rework/design.md
§12.5 ("Repo awareness"), so a title someone else already published can be labelled *Already in catalogue*.

The id is `<beq_metadata><beq_theMovieDB>`. The XML carries no movie/tv marker, and TMDB numbers films and series
separately (a film and a series can share an id), so the match is on the id **together with** whether `<beq_season>`
has anything in it (§12.14). Matching is never on the file name: ours are entry ids, other people's are not.

A large catalogue has thousands of files, so a scan parses only those whose mtime or size changed since the last
one (the index keeps what was read, keyed by relative path).
'''
import os
import xml.etree.ElementTree as ET
from typing import Dict, Mapping, NamedTuple, Optional, Set, Tuple


class XmlRecord(NamedTuple):
    mtime_ns: int
    size: int
    tmdb: str       # '' if the XML has no id (it can never match)
    is_tv: bool


def parse_xml(path: str) -> Optional[Tuple[str, bool]]:
    '''
    :return: (TMDB id, is tv) of one catalogue XML, or None if it is not readable XML.
    '''
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    tmdb = next((e.text or '' for e in root.iter('beq_theMovieDB')), '').strip()
    season = next(root.iter('beq_season'), None)
    is_tv = season is not None and bool((season.text or '').strip() or len(season) or season.attrib)
    return tmdb, is_tv


def scan_xml_repo(repo_path: str, known: Optional[Mapping[str, XmlRecord]] = None) -> Dict[str, XmlRecord]:
    '''
    :param known: what the last scan read, by path relative to the repo; a file whose mtime and size are unchanged
        is not opened again.
    :return: every readable `*.xml` in the repo (outside `.git`), by relative path. Empty if the repo is missing.
    '''
    known = known or {}
    found: Dict[str, XmlRecord] = {}
    if not repo_path or not os.path.isdir(repo_path):
        return found
    for folder, subfolders, names in os.walk(repo_path):
        subfolders[:] = [name for name in subfolders if name != '.git']
        for name in names:
            if not name.lower().endswith('.xml'):
                continue
            path = os.path.join(folder, name)
            relative = os.path.relpath(path, repo_path).replace(os.sep, '/')
            try:
                stat = os.stat(path)
            except OSError:
                continue
            before = known.get(relative)
            if before is not None and before.mtime_ns == stat.st_mtime_ns and before.size == stat.st_size:
                found[relative] = before
                continue
            parsed = parse_xml(path)
            if parsed is not None:
                found[relative] = XmlRecord(stat.st_mtime_ns, stat.st_size, *parsed)
    return found


def tmdb_index(records: Mapping[str, XmlRecord]) -> Dict[Tuple[str, bool], Set[str]]:
    ''' :return: (TMDB id, is tv) -> the file stems (name without `.xml`) that carry it. '''
    index: Dict[Tuple[str, bool], Set[str]] = {}
    for relative, record in records.items():
        if record.tmdb:
            stem = relative.rsplit('/', 1)[-1][:-len('.xml')]
            index.setdefault((record.tmdb, record.is_tv), set()).add(stem)
    return index
