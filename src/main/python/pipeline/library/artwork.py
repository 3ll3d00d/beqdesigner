'''
Automatic report-poster resolution for a library run -- design/archive/library-sync-pipeline-plan.md §3.1.3.

Tiers 2 and 3 of the plan's resolution order (tier 1, a reviewer's explicit choice, is stored as
QueueEntry.art_overridden and is never replaced by anything here):

  2. a local image the library source already has (LibraryItem.art_path, or the first of its art_candidates that is a
     file -- a source that lists without touching the disk leaves that check to here), used in place;
  3. TMDB's poster, downloaded once into the item's work directory.

Returns None when neither is available, which renders a chart-only report.
'''
import logging
import os
import shutil
from typing import Optional

import requests

from pipeline.library.source import LibraryItem
from pipeline.publish.art import fetch_poster

logger = logging.getLogger('library_artwork')


def resolve_art(item: LibraryItem, meta: dict, art_dir: Optional[str]) -> Optional[str]:
    '''
    :param meta: the entry's final metadata; its `poster` is TMDB's path fragment.
    :param art_dir: a durable per-item directory to download into; without one tier 3 is skipped, since a
        temp file would not outlive the run.
    :return: a local image path, or None. A failed download is logged and treated as "no artwork" -- it must
        never cost an item its extraction and design.
    '''
    local = next((c for c in (item.art_path, *item.art_candidates) if c and os.path.isfile(c)), None)
    if local:
        if art_dir is None:
            return local
        # kept in the title's own folder, as a downloaded poster is: the work directory is what a reviewer on another
        # machine shares, and a media path means nothing there (R6)
        dest = os.path.join(art_dir, 'poster' + os.path.splitext(local)[1].lower())
        try:
            os.makedirs(art_dir, exist_ok=True)
            if os.path.abspath(local) != os.path.abspath(dest):
                shutil.copyfile(local, dest)
            return dest
        except OSError as error:
            logger.warning('Unable to copy the artwork of %s into its work folder: %s', item.id, error)
            return local
    poster = meta.get('poster')
    if not poster or art_dir is None:
        return None
    try:
        os.makedirs(art_dir, exist_ok=True)
        downloaded = fetch_poster(poster, dest_dir=art_dir)
        # a fixed name, so a later re-download replaces rather than accumulates files
        dest = os.path.join(art_dir, 'poster' + os.path.splitext(downloaded)[1])
        os.replace(downloaded, dest)
        return dest
    except (requests.RequestException, OSError) as error:
        logger.warning('Unable to download the TMDB poster for %s: %s', item.id, error)
        return None
