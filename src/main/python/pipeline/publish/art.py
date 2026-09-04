'''
pipeline/publish/art.py: poster fetch -- design/pipeline-implementation-plan.md
phase 3 (item 8).

TMDB's poster_path (pipeline.metadata.BeqMetadata.poster) is a path
fragment, not a URL (design/api-headless-pipeline.md §6) -- the image base
URL and a size have to be supplied. The GUI today has the user paste a full
URL themselves (SaveReportDialog.imageURL); this is that step done
headlessly, combining TMDB's documented image base URL with the fragment.

Qt-free equivalent of SaveReportDialog.__download_image (model/report.py:792),
minus the QMessageBox on failure -- this raises instead.
'''
import os
import tempfile
from typing import Optional

import requests

TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/'


def poster_url(poster_path: str, size: str = 'original', base_url: str = TMDB_IMAGE_BASE_URL) -> str:
    ''' :return: a full TMDB image URL for a poster_path fragment (e.g. "/abc.jpg"). '''
    return f"{base_url.rstrip('/')}/{size}/{poster_path.lstrip('/')}"


def fetch_poster(poster_path: str, dest_dir: Optional[str] = None, size: str = 'original',
                 base_url: str = TMDB_IMAGE_BASE_URL) -> str:
    '''
    Downloads the poster for a TMDB poster_path fragment.
    :param poster_path: BeqMetadata.poster -- a path fragment, not a URL.
    :param dest_dir: directory to write into; a temp dir if not given.
    :param size: the TMDB image size variant (e.g. 'original', 'w500').
    :return: the path to the downloaded file.
    :raises requests.HTTPError: on a failed download.
    '''
    url = poster_url(poster_path, size=size, base_url=base_url)
    suffix = os.path.splitext(poster_path)[1] or '.jpg'
    fd, path = tempfile.mkstemp(suffix=suffix, dir=dest_dir)
    os.close(fd)
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        with open(path, 'wb') as f:
            f.write(resp.content)
    except Exception:
        os.remove(path)
        raise
    return path
