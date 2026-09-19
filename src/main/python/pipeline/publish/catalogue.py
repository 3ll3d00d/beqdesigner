'''
pipeline/publish/catalogue.py: where a title lives in the two catalogue repos, and the digest that says whether what
was written there is still what would be written now -- design/library-sync/workflow-rework/design.md §12.6/§12.7.

Qt-free, and with no git: naming and hashing only, shared by publishing (writes the files) and committing (needs to
know which files those were).
'''
import hashlib
import json
import os
from dataclasses import asdict
from typing import Optional, Tuple

from pipeline.metadata import BeqMetadata


def catalogue_paths(entry_id: str, xml_dir: str = '', image_dir: str = '') -> Tuple[str, str]:
    '''
    :return: (xml_relative_path, image_relative_path) of an entry within its repos: `<xml_dir>/<entry_id>.xml` and
        `<image_dir>/<entry_id>.png`. The entry id is stable across reorderings and re-runs, so a revision rewrites
        the same path. beqcatalogue globs **/*.xml, so the naming is ours to choose.
    '''
    return os.path.join(xml_dir, f"{entry_id}.xml"), os.path.join(image_dir, f"{entry_id}.png")


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_digest(filter_json: dict, meta: BeqMetadata, art_path: Optional[str], has_image: bool,
                   mv_offset: float) -> str:
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
    '''
    payload = {
        'filter': filter_json,
        'meta': asdict(meta),
        'art': _file_sha256(art_path),
        'image': has_image,
        'mv_offset': mv_offset,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
                          ).hexdigest()
