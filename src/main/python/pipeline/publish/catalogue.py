'''
pipeline/publish/catalogue.py: where a title lives in the two catalogue repos, and the digest that says whether what
was written there is still what would be written now -- design/library-sync/workflow-rework/design.md §12.6/§12.7.

Qt-free, and running no git: naming and hashing only, shared by publishing (writes the files) and committing (needs to
know which files those were).
'''
import hashlib
import json
import os
from dataclasses import asdict
from typing import Optional, Tuple

from pipeline.metadata import BeqMetadata
from pipeline.publish.git import join_posix
from pipeline.publish.report import ReportSpec


def catalogue_paths(entry_id: str, xml_dir: str = '', image_dir: str = '') -> Tuple[str, str]:
    '''
    :return: (xml_relative_path, image_relative_path) of an entry within its repos: `<xml_dir>/<entry_id>.xml` and
        `<image_dir>/<entry_id>.png`. The entry id is stable across reorderings and re-runs, so a revision rewrites
        the same path. beqcatalogue globs **/*.xml, so the naming is ours to choose.

        Always `/`-separated, on Windows too, because that is how git spells a path (`git status` says `xml/one.xml`);
        a path with the platform's separator would never match what git reports. The file system is reached by
        splitting the path on `/` (see pipeline.publish.git.write_files()). Backslashes in a directory are read as
        separators.
    '''
    return join_posix(xml_dir, f"{entry_id}.xml"), join_posix(image_dir, f"{entry_id}.png")


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_digest(filter_json: dict, meta: BeqMetadata, art_path: Optional[str], has_image: bool,
                   mv_offset: float, report_spec: Optional[ReportSpec] = None) -> str:
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

    Deliberately **not** in it: the image's GitHub owner/repo (`image_owner`/`image_repo_name`) -- the discovery index
    computes this digest too, from settings that do not carry them, so counting them would make every title look out
    of date there; a repo that moves needs a republish by hand -- and `xml_dir` and `image_dir`. A different directory is a different *location*, not a
    different content; a title published to a new `xml_dir` is written there by the next publish, and the file at the
    old location is left behind for a person to remove.
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
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
                          ).hexdigest()
