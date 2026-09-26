'''
pipeline/publish/catalogue.py: where a title lives in the two catalogue repos, and the digest that says whether what
was written there is still what would be written now -- design/archive/library-sync/workflow-rework/design.md §12.6/§12.7.

Qt-free, and running no git: naming and hashing only, shared by publishing (writes the files) and committing (needs to
know which files those were).
'''
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Optional, Tuple, Union

from pipeline.metadata import BeqMetadata
from pipeline.publish.git import join_posix
from pipeline.publish.report import ReportSpec


DEFAULT_MOVIES_DIR = 'movies'
DEFAULT_TV_DIR = 'tv'


@dataclass(frozen=True)
class CategoryFolders:
    '''
    Where movies and TV go beneath a repository location. Passed wherever a `category_folders` flag was, in its place:
    it is truthy (folders on) and names the two folders. `True` still means the default names.
    '''
    movies: str = DEFAULT_MOVIES_DIR
    tv: str = DEFAULT_TV_DIR


class FolderName(str):
    '''A category that is already a folder name (configured), not the legacy 'film'/'TV' marker.'''


def category_folders_from_values(values: Mapping) -> Union[bool, CategoryFolders]:
    '''
    The setting from a profile's flat options (`sync:` merged with `run:`): **on unless `category_folders` is false**, with
    `movies_dir` / `tv_dir` naming the folders. False when off.
    '''
    if not values.get('category_folders', True):
        return False
    return CategoryFolders(_folder_name(values.get('movies_dir'), DEFAULT_MOVIES_DIR),
                           _folder_name(values.get('tv_dir'), DEFAULT_TV_DIR))


def _folder_name(value, default: str) -> str:
    name = str(value or '').strip().replace('\\', '/').strip('/')
    if not name:
        return default
    if any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError(f'{name!r} is not a folder within the repository')
    return name


def category_folder(season) -> str:
    '''The default repository subfolder for a published title with (or without) season metadata.'''
    return DEFAULT_TV_DIR if season else DEFAULT_MOVIES_DIR


def category_for_season(season, enabled: Union[bool, CategoryFolders, None]) -> Optional[str]:
    '''The path category, or the flat layout when the setting is off.'''
    if not enabled:
        return None
    if isinstance(enabled, CategoryFolders):
        return FolderName(enabled.tv if season else enabled.movies)
    return 'TV' if season else 'film'


def category_for_metadata(meta: Mapping, defaults: Optional[Mapping], enabled: Union[bool, CategoryFolders, None]
                          ) -> Optional[str]:
    '''Use the same season value as publication metadata, including profile defaults.'''
    return category_for_season(meta.get('season', (defaults or {}).get('season')), enabled)


def _folder(category: Optional[str]) -> str:
    if category is None:
        return ''
    if isinstance(category, FolderName):
        return str(category)
    if category not in ('film', 'TV'):
        raise ValueError(f'unknown catalogue content type {category!r}')
    return category_folder(category == 'TV')


_AUDIO_SEPARATOR = ' + '
_MAX_STEM = 150


def catalogue_stem(meta: Mapping, fallback: str = '') -> str:
    '''
    The readable file name (no extension) a title is published under, in the order beqcatalogue's own files were named:
    `Title (Year) (Edition) S01 Audio`, e.g. `1917 (2019) (Amazon) DD+` or `The Expanse (2015) S01 DD+`. The master volume is
    deliberately not in it (a revision changes the gain, and must rewrite the same path).
    :param fallback: the name if there is no title (the entry id).
    '''
    from pipeline.library.workdir import _safe_name
    title = str(meta.get('title') or '').strip()
    if not title:
        return fallback
    parts = [title]
    if meta.get('year'):
        parts.append(f"({meta['year']})")
    if meta.get('edition'):
        parts.append(f"({meta['edition']})")
    season = meta.get('season')
    if season:
        season = str(season)
        parts.append(f'S{int(season):02d}' if season.isdigit() else season)
    audio = meta.get('audio_types') or []
    if isinstance(audio, str):
        audio = [audio]
    if audio:
        parts.append(_AUDIO_SEPARATOR.join(str(a) for a in audio))
    return _safe_name(' '.join(parts))[:_MAX_STEM].rstrip(' .') or fallback


def unique_stem(stem: str, taken: Callable[[str], bool]) -> str:
    '''`stem`, or `stem (2)`, `stem (3)` ... for the first that `taken()` does not say is in use.'''
    candidate, number = stem, 1
    while taken(candidate):
        number += 1
        candidate = f'{stem} ({number})'
    return candidate


def catalogue_paths(entry_id: str, xml_dir: str = '', image_dir: str = '', *,
                    category: Optional[str] = None, stem: Optional[str] = None) -> Tuple[str, str]:
    '''
    :return: (filter_relative_path, image_relative_path) of an entry within its repos: `<xml_dir>/<stem>.json` and
        `<image_dir>/<stem>.png`. The stem is what publishing recorded on the entry (`QueueEntry.published_stem`, see
        catalogue_stem()) and is the stable entry id for a title published before names were readable; it is stable across
        reorderings and re-runs, so a revision rewrites the same path. With a category, movies and TV go under their
        folders (`movies/` and `tv/` unless configured, see CategoryFolders) beneath the configured prefix.
        BEQCatalogue reads individual JSON records recursively.

        Always `/`-separated, on Windows too, because that is how git spells a path (`git status` says `filters/one.json`);
        a path with the platform's separator would never match what git reports. The file system is reached by
        splitting the path on `/` (see pipeline.publish.git.write_files()). Backslashes in a directory are read as
        separators.
    '''
    folder = _folder(category)
    name = stem or entry_id
    return join_posix(xml_dir, folder, f"{name}.json"), join_posix(image_dir, folder, f"{name}.png")


def heatmap_path(image_relative_path: str) -> str:
    '''Where a title's heatmap image goes: beside its report image, `<stem> heatmap.png`.'''
    return image_relative_path[:-len('.png')] + ' heatmap.png' if image_relative_path.endswith('.png') \
        else image_relative_path + ' heatmap.png'


def aggregate_path(xml_dir: str = '', *, category: Optional[str] = None) -> str:
    '''The filter repo's derived aggregate, beside its individual records.'''
    folder = _folder(category)
    return join_posix(xml_dir, folder, 'database.json')


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_digest(filter_json: dict, meta: BeqMetadata, art_path: Optional[str], has_image: bool,
                   mv_offset: float, report_spec: Optional[ReportSpec] = None, image_owner: Optional[str] = None,
                   image_repo_name: Optional[str] = None) -> str:
    '''
    Hash of everything a publish is built from. Two publishes with the same digest write the same catalogue entry,
    so a title whose current digest differs from the one recorded when it was published is *out of date* -- which
    is how a metadata typo, a re-picked poster or an edited project reaches the catalogue without a second review.

    :param filter_json: the published filter (`CompleteFilter.to_json()`) -- the project's if a human edited it,
        else the chosen candidate's.
    :param meta: the metadata as it will be published, *before* the image URLs are filled in (those derive from the
        repo, not from the title).
    :param art_path: the poster file; its content is hashed, so replacing the file changes the digest and merely
        touching it does not.
    :param has_image: whether a report image is published at all.
    :param mv_offset: the master-volume offset drawn on the report image.
    :param report_spec: the report image's layout; a changed style redraws every image, so it changes the digest.
        Only counted when an image is published and it is not the default, so a digest recorded before this existed
        (which had no report spec) is still the digest of a default-styled publish.

    :param image_owner/image_repo_name: the GitHub owner and repository the report image's URL is built from, **as given
        to publish** (None when publish works them out from the images repo's remote). They are in the XML (the URL),
        so a change is a change of content; only counted when an image is published and one is given, so a digest
        recorded before they were counted -- or with neither set -- is unchanged. The discovery index is given them by
        `ScanSettings` (`image_owner`, `image_repo_name`, from `sync:`), which must match what publish is given.

    Deliberately **not** in it: the designer that produced the filter (the filter itself is), and `xml_dir` and
    `image_dir`. A different directory is a different *location*, not a different content; a title published to a new
    `xml_dir` is written there by the next publish, and the file at the old location is left behind for a person to
    remove.
    '''
    payload = {
        'filter': filter_json,
        'meta': asdict(meta),
        'art': _file_sha256(art_path),
        'image': has_image,
        'mv_offset': mv_offset,
    }
    if has_image and report_spec is not None and report_spec != ReportSpec():
        payload['report_spec'] = asdict(report_spec)
    if has_image and (image_owner or image_repo_name):
        payload['image_owner'] = image_owner or ''
        payload['image_repo_name'] = image_repo_name or ''
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
                          ).hexdigest()
