'''
A fake 1.6 designer for the redesign-scope tests (design/redesign-scope.md): a real HTTP server answering `GET /health`
and `POST /impact` as designer-interface.md §7.1 says, so the caller is tested against the wire, not a mock. beqforge's
own /impact is not built yet (its TODO 34).

Each request to /impact is checked against docs/schema/http_designer_impact_request.schema.json and kept in `asked`;
`response_errors` checks an answer against the response schema.
A record's answer is `tiers[path]` (a tier, or a whole result to send as it is), else `unaffected`; a path the designer
does not hold is `unknown`, as a missing record is.
'''
import http.server
import json
import os
import threading
from typing import Dict, List, Optional

from jsonschema import Draft202012Validator

_SCHEMAS = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'schema')
_SCHEMA = os.path.join(_SCHEMAS, 'http_designer_impact_request.schema.json')
CHANGE = {'design_version': 7, 'id': 'fit-choice', 'summary': 'the fitter chooses among near-equal fits',
          'moved': 'swapped designed/declined, changed strategy, or moved > 1 dB at 5-50 Hz',
          'basis': [{'class': 'declined_after_fit', 'measured': 34, 'moved': 18},
                    {'class': 'designed_all_passing', 'measured': 15, 'moved': 2}]}
_SCORES = {'will_move': 1.0, 'likely': 0.7, 'possible': 0.2, 'unaffected': 0.0}


def response_errors(answer: dict) -> List[str]:
    ''' What the 1.6 response schema says of `answer`. '''
    with open(os.path.join(_SCHEMAS, 'http_designer_impact_response.schema.json')) as f:
        return [e.message for e in Draft202012Validator(json.load(f)).iter_errors(answer)]


class FakeImpactDesigner:
    def __init__(self, design_version: int = 7, design_settings: Optional[str] = 's0', impact: bool = True,
                 tiers: Optional[Dict[str, object]] = None, held: Optional[set] = None):
        self.health = {'contract_version': '1.6', 'shared_root': True, 'impact': impact,
                       'design_version': design_version, 'design_settings': design_settings,
                       'designer_build': 'fake+src:000000000000'}
        self.tiers = dict(tiers or {})
        self.held = held          # None: every path is held
        self.asked: List[dict] = []
        self.status: Optional[int] = None   # set to answer every /impact with this status instead
        self.mangle = None                  # set to a function of the answer, to send something out of shape
        with open(_SCHEMA) as f:
            self.validator = Draft202012Validator(json.load(f))
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        ''' Its design endpoint, as a profile names it. '''
        return f'http://127.0.0.1:{self.server.server_address[1]}/design'

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.thread.join(timeout=5)

    def answer(self, body: dict) -> dict:
        results = []
        for record in body['records']:
            path = record['path']
            if self.held is not None and path not in self.held:
                results.append({'id': record['id'], 'tier': 'unknown', 'score': None, 'from_design_version': None,
                                'unknown_reason': 'not_found', 'reasons': [{'change': None, 'feature': None, 'detail': 'record not found'}]})
                continue
            tier = self.tiers.get(path, 'unaffected')
            if isinstance(tier, dict):
                results.append({'id': record['id'], **tier})
                continue
            reasons = [] if tier == 'unaffected' else [{'change': 'fit-choice', 'feature': 'fit.choice',
                                                        'detail': f'{path} relied on the fit choice'}]
            results.append({'id': record['id'], 'tier': tier, 'score': _SCORES[tier], 'from_design_version': 6,
                            'reasons': reasons})
        out = {'contract_version': '1.6', **{k: self.health[k] for k in ('design_version', 'design_settings',
                                                                          'designer_build')},
               'changes': [CHANGE], 'results': results}
        return self.mangle(out) if self.mangle else out

    def _handler(self):
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _send(self, status, body):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == '/health':
                    self._send(200, fake.health)
                else:
                    self._send(404, {'detail': 'not found'})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path != '/impact':
                    self._send(404, {'detail': 'not found'})
                    return
                errors = [e.message for e in fake.validator.iter_errors(body)]
                ids = [r.get('id') for r in body.get('records') or []]
                if errors or len(set(ids)) != len(ids):
                    self._send(400, {'detail': errors or ['two records share an id']})
                    return
                fake.asked.append(body)
                if fake.status:
                    self._send(fake.status, {'detail': 'no shared root is configured'})
                    return
                self._send(200, fake.answer(body))

            def log_message(self, *_):
                pass
        return Handler
