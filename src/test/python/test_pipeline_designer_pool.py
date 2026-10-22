'''
One designer served by several endpoints (pipeline.designer.pool, designer-interface.md §7.2): each used while it is up,
each sent no more than its `parallel` designs at once, and a design that cannot reach one sent to another.
'''
import threading
import time

import pytest
import requests

from pipeline.designer.contract import DesignResponse
from pipeline.designer.http_binding import DesignerUnavailable, HttpDesignerError, declared_endpoints, \
    register_declared_designers
from pipeline.designer.pool import Activity, DesignerPool, PoolEndpoint, cannot_reach, designing
from pipeline.designer.registry import designer_capacity, get_designer, takes_sources, unregister_designer
from pipeline.library.failure import unavailable_reason


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeEndpoint:
    ''' A designer at `url`: up or down, recording each design it is sent and how many it had at once. '''

    def __init__(self, url, up=True, fails_with=None, hold=None, reports=None):
        self.url, self.up, self.fails_with, self.hold, self.reports = url, up, fails_with, hold, reports
        self.designed, self.health_asked, self.sources = [], 0, []
        self.active = self.most_at_once = 0
        self.lock = threading.Lock()

    def health(self):
        self.health_asked += 1
        if not self.up:
            raise DesignerUnavailable(f'GET {self.url}/health failed: refused')
        return {'status': 'ok'} if self.reports is None else {'status': 'ok', 'parallel': self.reports}

    def call(self, request, sources=None):
        with self.lock:
            self.active += 1
            self.most_at_once = max(self.most_at_once, self.active)
        try:
            if self.hold is not None:
                assert self.hold.wait(5)
            if self.fails_with is not None:
                raise self.fails_with
            self.designed.append(request)
            self.sources.append(sources)
            return DesignResponse(contract_version='1.0', decline_reason='no_rolloff_detected')
        finally:
            with self.lock:
                self.active -= 1

    def endpoint(self, parallel=None, takes_sources=False):
        return PoolEndpoint(self.url, self.call, self.health, parallel=parallel, takes_sources=takes_sources)


def _request():
    return object()   # the pool passes the request through untouched


def _unreachable(url):
    ''' What http_designer raises when a POST cannot connect. '''
    try:
        raise requests.ConnectionError('Connection refused')
    except requests.ConnectionError as e:
        try:
            raise HttpDesignerError(f'POST {url} failed: {e}') from e
        except HttpDesignerError as wrapped:
            return wrapped


def test_capacity_is_the_sum_of_the_endpoints_parallel():
    a, b = FakeEndpoint('http://a'), FakeEndpoint('http://b')
    assert DesignerPool('d', [a.endpoint(parallel=1), b.endpoint(parallel=3)]).capacity == 4


def test_an_endpoint_without_parallel_takes_what_its_health_reports_and_follows_a_restart():
    clock = Clock()
    reporting, old = FakeEndpoint('http://new', reports=2), FakeEndpoint('http://old')   # 0.2.4 and an older build
    pool = DesignerPool('d', [reporting.endpoint(), old.endpoint()], clock=clock, recheck_seconds=60)
    assert pool.capacity == 2   # not asked yet: 1 each

    pool(_request())
    assert pool.capacity == 3 and {s.url: s.parallel for s in pool.states()} == {'http://new': 2, 'http://old': 1}

    reporting.reports = 4       # restarted with --workers 4
    clock.now += 61
    pool(_request())
    assert pool.capacity == 5


def test_a_profile_parallel_overrides_what_the_designer_reports():
    reporting = FakeEndpoint('http://a', reports=4)
    pool = DesignerPool('d', [reporting.endpoint(parallel=1)], clock=Clock())
    pool(_request())
    assert pool.capacity == 1


@pytest.mark.parametrize('body, expected', [
    ({'parallel': 2}, 2), ({'parallel': 8}, 8), ({}, None), ({'parallel': 0}, None), ({'parallel': 9}, None),
    ({'parallel': True}, None), ({'parallel': '2'}, None), (None, None), ([], None)])
def test_only_a_sensible_reported_parallel_is_used(body, expected):
    from pipeline.designer.http_binding import reported_parallel
    assert reported_parallel(body) == expected


def test_capacity_counts_only_the_endpoints_that_are_up_but_never_drops_below_one():
    up, down = FakeEndpoint('http://up', reports=2), FakeEndpoint('http://down', up=False, reports=3)
    pool = DesignerPool('d', [up.endpoint(), down.endpoint(parallel=3)], clock=Clock())
    assert pool.capacity == 4   # neither asked yet
    pool(_request())
    assert pool.capacity == 2

    gone = DesignerPool('d', [FakeEndpoint('http://x', up=False).endpoint()], clock=Clock())
    with pytest.raises(DesignerUnavailable):
        gone(_request())
    assert gone.capacity == 1


def test_a_down_endpoint_is_left_out_and_the_designs_go_to_the_live_one():
    up, down = FakeEndpoint('http://up'), FakeEndpoint('http://down', up=False)
    pool = DesignerPool('d', [down.endpoint(), up.endpoint()], clock=Clock())

    for _ in range(3):
        pool(_request())

    assert len(up.designed) == 3 and down.designed == []
    assert down.health_asked == 1   # asked once, then left alone until the recheck is due
    states = {s.url: s for s in pool.states()}
    assert states['http://down'].up is False and 'refused' in states['http://down'].detail
    assert states['http://up'].up is True


def test_an_endpoint_that_comes_back_is_used_after_the_recheck():
    clock = Clock()
    a, b = FakeEndpoint('http://a'), FakeEndpoint('http://b', up=False)
    pool = DesignerPool('d', [a.endpoint(), b.endpoint()], clock=clock, recheck_seconds=60)
    pool(_request())
    b.up = True

    clock.now += 30
    pool(_request())
    assert b.designed == []   # not asked again yet

    clock.now += 31
    b_hold = threading.Event()
    a.hold = b_hold           # a is busy: the next design can only go to b, now that it answers
    worker = threading.Thread(target=pool, args=(_request(),))
    worker.start()
    time.sleep(0.1)
    pool(_request())
    b_hold.set()
    worker.join(5)
    assert len(b.designed) == 1


def test_a_design_that_cannot_reach_its_endpoint_is_sent_to_another():
    clock = Clock()
    gone, live = FakeEndpoint('http://gone'), FakeEndpoint('http://live')
    gone.fails_with = _unreachable('http://gone')
    pool = DesignerPool('d', [gone.endpoint(parallel=4), live.endpoint()], clock=clock)

    response = pool(_request())

    assert response.decline_reason == 'no_rolloff_detected' and len(live.designed) == 1
    assert {s.url: s.up for s in pool.states()} == {'http://gone': False, 'http://live': True}


def test_a_titles_own_failure_is_not_retried_elsewhere():
    ''' A 500, or a refused request, is about the design: another endpoint would only repeat it. '''
    broken, other = FakeEndpoint('http://a'), FakeEndpoint('http://b')
    broken.fails_with = HttpDesignerError('POST http://a failed: 500 Server Error: the fit diverged')
    pool = DesignerPool('d', [broken.endpoint(parallel=2), other.endpoint()], clock=Clock())

    with pytest.raises(HttpDesignerError, match='diverged'):
        pool(_request())

    assert other.designed == [] and all(s.up for s in pool.states())


def test_with_every_endpoint_down_the_design_is_unavailable_as_before():
    a, b = FakeEndpoint('http://a', up=False), FakeEndpoint('http://b', up=False)
    pool = DesignerPool('d', [a.endpoint(), b.endpoint()], clock=Clock())

    with pytest.raises(DesignerUnavailable, match='no endpoint of designer .d. is up') as raised:
        pool(_request())

    assert 'http://a' in str(raised.value) and 'http://b' in str(raised.value)
    assert unavailable_reason(raised.value) is not None   # a run counts it towards stop_after_unavailable


def test_every_endpoint_unreachable_mid_design_is_unavailable():
    a, b = FakeEndpoint('http://a'), FakeEndpoint('http://b')
    a.fails_with, b.fails_with = _unreachable('http://a'), _unreachable('http://b')
    pool = DesignerPool('d', [a.endpoint(), b.endpoint()], clock=Clock())

    with pytest.raises(Exception) as raised:
        pool(_request())

    assert unavailable_reason(raised.value) is not None


def test_no_endpoint_is_sent_more_than_its_parallel_at_once():
    release = threading.Event()
    one, three = FakeEndpoint('http://one', hold=release), FakeEndpoint('http://three', hold=release)
    pool = DesignerPool('d', [one.endpoint(parallel=1), three.endpoint(parallel=3)], clock=Clock())
    workers = [threading.Thread(target=pool, args=(_request(),)) for _ in range(6)]
    for w in workers:
        w.start()
    time.sleep(0.3)
    assert (one.active, three.active) == (1, 3)   # the other two wait in the pool, not at a designer
    release.set()
    for w in workers:
        w.join(5)

    assert one.most_at_once == 1 and three.most_at_once == 3
    assert len(one.designed) + len(three.designed) == 6


def test_states_say_what_each_endpoint_is_designing_while_it_is_out():
    release = threading.Event()
    one, two = FakeEndpoint('http://one', hold=release), FakeEndpoint('http://two', hold=release)
    pool = DesignerPool('d', [one.endpoint(parallel=1), two.endpoint(parallel=1)], clock=Clock(), wall_clock=lambda: 5.0)

    def design(title_id, title):
        with designing(title_id, title):
            pool(_request())

    workers = [threading.Thread(target=design, args=(f'id{i}', f'Title {i}')) for i in range(2)]
    for w in workers:
        w.start()
    time.sleep(0.3)
    out = {s.url: s.active for s in pool.states()}
    assert sorted((a for v in out.values() for a in v), key=lambda a: a.title_id) == [Activity('id0', 'Title 0', 5.0), Activity('id1', 'Title 1', 5.0)]
    assert all(len(v) == 1 for v in out.values())   # one each: each endpoint takes one at once
    release.set()
    for w in workers:
        w.join(5)

    assert all(s.active == () for s in pool.states())   # nothing is out once the designs come back


def test_a_design_not_named_is_out_but_not_listed():
    release = threading.Event()
    one = FakeEndpoint('http://one', hold=release)
    pool = DesignerPool('d', [one.endpoint(parallel=1)], clock=Clock())
    worker = threading.Thread(target=pool, args=(_request(),))
    worker.start()
    time.sleep(0.3)
    [state] = pool.states()
    assert (state.in_use, state.active) == (1, ())
    release.set()
    worker.join(5)


def test_a_design_moved_to_another_endpoint_is_listed_only_where_it_is():
    down = FakeEndpoint('http://down', fails_with=_unreachable('http://down'))
    seen = {}
    pool = None

    class Watching(FakeEndpoint):
        def call(self, request, sources=None):
            seen.update({s.url: s.active for s in pool.states()})
            return super().call(request, sources)

    up = Watching('http://up')
    pool = DesignerPool('d', [down.endpoint(parallel=5), up.endpoint(parallel=1)], clock=Clock(), wall_clock=lambda: 7.0)
    with designing('id', 'Alien'):
        pool(_request())

    assert seen == {'http://down': (), 'http://up': (Activity('id', 'Alien', 7.0),)}


def test_sources_go_only_to_an_endpoint_that_takes_references():
    shared, inline = FakeEndpoint('http://shared'), FakeEndpoint('http://inline')
    pool = DesignerPool('d', [shared.endpoint(takes_sources=True), inline.endpoint()], clock=Clock())
    assert pool.takes_sources

    release = threading.Event()
    shared.hold = release
    worker = threading.Thread(target=pool, args=(_request(),), kwargs={'sources': 'the wavs'})
    worker.start()
    time.sleep(0.1)
    pool(_request(), sources='the wavs')   # shared is busy: this one goes inline
    release.set()
    worker.join(5)

    assert shared.sources == ['the wavs'] and inline.sources == [None]


@pytest.mark.parametrize('error, reachable', [
    (DesignerUnavailable('down'), False),
    (requests.ConnectionError('refused'), False),
    (requests.ConnectTimeout('slow to connect'), False),
    (requests.ReadTimeout('the design ran over'), True),
    (HttpDesignerError('POST failed: 422'), True),
    (ValueError('bad body'), True),
])
def test_only_a_connection_failure_or_a_gateway_moves_a_design(error, reachable):
    assert cannot_reach(error) is not reachable


def test_a_gateway_answer_is_down_and_a_server_error_is_not():
    def http_error(status):
        response = requests.Response()
        response.status_code = status
        return requests.HTTPError(f'{status}', response=response)
    assert cannot_reach(http_error(503)) and cannot_reach(http_error(502))
    assert not cannot_reach(http_error(500)) and not cannot_reach(http_error(404))


# --- the profile's forms ---------------------------------------------------------------------------------------------

def test_a_single_url_is_one_endpoint_taking_run_parallelism_design():
    [only] = declared_endpoints('b', {'url': 'http://x/design', 'timeout': 600, 'by_reference': True}, 2)
    assert (only.url, only.parallel, only.timeout, only.by_reference) == ('http://x/design', 2, 600.0, True)


def test_endpoints_each_say_their_own_parallel_and_share_the_designers_timeout_and_headers():
    a, b = declared_endpoints('b', {'timeout': 900, 'headers': {'Authorization': 'Bearer t'}, 'endpoints': [
        {'url': 'http://a/design', 'by_reference': True},
        {'url': 'http://b/design', 'parallel': 3, 'headers': {'X-Box': 'two'}}]}, default_parallel=4)

    assert (a.parallel, a.by_reference, a.headers) == (None, True, {'Authorization': 'Bearer t'})   # what /health says
    assert (b.parallel, b.by_reference, b.headers) == (3, False, {'Authorization': 'Bearer t', 'X-Box': 'two'})
    assert a.timeout == b.timeout == 900.0


@pytest.mark.parametrize('spec, message', [
    ({'url': 'http://a', 'endpoints': [{'url': 'http://b'}]}, 'either url or endpoints'),
    ({'endpoints': []}, 'non-empty list'),
    ({'endpoints': [{'parallel': 2}]}, 'endpoint 1 needs a url'),
    ({'endpoints': [{'url': 'http://a', 'parallel': 0}]}, 'parallel must be an integer'),
    ({'endpoints': [{'url': 'http://a', 'parallel': 9}]}, 'parallel must be an integer'),
    ({'endpoints': [{'url': 'http://a', 'by_reference': 'yes'}]}, 'by_reference must be true or false'),
    ({'timeout': 5}, 'needs a url'),
])
def test_a_malformed_designer_is_refused_naming_what_is_wrong(spec, message):
    with pytest.raises(ValueError, match=message):
        declared_endpoints('b', spec)


def test_a_declared_pool_is_registered_under_its_name_with_its_capacity(monkeypatch):
    from pipeline.designer import http_binding
    made = {}
    monkeypatch.setattr(http_binding, 'http_designer',
                        lambda url, **options: made.setdefault(url, options) and None or (lambda request, sources=None: None))
    try:
        register_declared_designers({'beqforge': {'timeout': 100, 'endpoints': [
            {'url': 'http://a/design', 'by_reference': True},
            {'url': 'http://b/design', 'parallel': 2}]}}, shared_root='/work')

        assert designer_capacity('beqforge') == 3 and takes_sources('beqforge')   # a: 1 until its /health is read
        assert made['http://a/design'] == {'timeout': 100.0, 'headers': None, 'shared_root': '/work'}
        # a designer that answers one at a time queues what it is sent at once: its timeout allows for that queue
        assert made['http://b/design'] == {'timeout': 200.0, 'headers': None}
        assert isinstance(get_designer('beqforge'), DesignerPool)
    finally:
        unregister_designer('beqforge')


def test_a_run_is_sized_to_its_designers_capacity_not_run_parallelism_design(tmp_path):
    from pipeline.library.setup import register_designers, run_config_from_values
    values = {'work_dir': str(tmp_path / 'w'), 'queue_dir': str(tmp_path / 'q'), 'designer': 'beqforge',
              'parallelism': {'design': 1}}
    config = {'designers': {'beqforge': {'endpoints': [{'url': 'http://a/design', 'parallel': 2},
                                                       {'url': 'http://b/design', 'parallel': 3}]}}}
    try:
        register_designers(values, config)
        assert run_config_from_values(values, config).design_parallelism == 5
        config = {'designers': {'beqforge': 'http://a/design'}}
        values['parallelism'] = {'design': 2}
        register_designers(values, config)
        assert run_config_from_values(values, config).design_parallelism == 2   # one url: as before
    finally:
        unregister_designer('beqforge')
