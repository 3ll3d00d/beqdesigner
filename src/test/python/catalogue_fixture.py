'''
A fake beqcatalogue for tests: `compare/titles.json` and per-title `compare/<key>.json`, answered as `requests.get` would
(ETags, 304 on If-None-Match, 404), with every request recorded.
'''
import json

import requests

TITLES = [{'key': 'film-alien_348', 'title': 'Alien', 'year': '1979', 'content_type': 'film', 'authors': ['aron7awol']},
          {'key': 'film-brazil_68', 'title': 'Brazil', 'year': '1985', 'content_type': 'film', 'authors': ['mobe1969']},
          {'key': 'tv-alien_9999', 'title': 'Alien', 'year': '1979', 'content_type': 'tv', 'authors': ['x']}]
ALIEN = [{'author': 'aron7awol', 'audioTypes': ['DTS-HD MA 5.1'], 'content_type': 'film', 'season': '', 'episode': '',
          'edition': '', 'catalogue_url': 'https://beqcatalogue.readthedocs.io/en/latest/aron7awol/1/#alien',
          'filters': [{'type': 'LowShelf', 'freq': 20.0, 'gain': 5.0, 'q': 0.9, 'count': 2}]},
         {'author': 'mobe1969', 'audioTypes': ['Atmos'], 'content_type': 'film', 'season': '', 'episode': '',
          'edition': "Director's Cut", 'filters': [{'type': 'PeakingEQ', 'freq': 30.0, 'gain': 3.0, 'q': 2.0}]}]


class Answer:
    def __init__(self, status, body=b'', etag=None):
        self.status_code, self._body, self.headers = status, body, {'ETag': etag} if etag else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code}', response=self)

    def iter_content(self, chunk_size):
        yield self._body


class FakeCatalogue:
    def __init__(self, files=None):
        self.files = {'titles.json': TITLES, 'film-alien_348.json': ALIEN} if files is None else files
        self.asked = []
        self.down = False

    def __call__(self, url, headers=None, timeout=None, stream=False):
        name = url.rsplit('/compare/', 1)[-1]
        self.asked.append((name, dict(headers or {})))
        if self.down:
            raise requests.ConnectionError('no route to host')
        if name not in self.files:
            return Answer(404)
        etag = f'"{name}-v1"'
        if (headers or {}).get('If-None-Match') == etag:
            return Answer(304)
        body = self.files[name]
        return Answer(200, body if isinstance(body, bytes) else json.dumps(body).encode(), etag)
