'''
The work directory's join inbox and handing work to another process's run --
design/archive/library-sync/worklist-feedback.md F5 (pipeline.library.inbox, pipeline.library.handoff).
'''
import os
import threading

from pipeline.library.handoff import JOINED, WITHDRAWN, hand_off
from pipeline.library.inbox import PENDING, TAKEN, WITHDRAWN as WITHDRAWN_STATE, WorkDirInbox, inbox_path
from pipeline.library.join import JoinQueue, JoinRequest
from pipeline.library.selection import Selection
from pipeline.service.lease import WorkDirLease, read_lease, run_lease


def test_a_posted_request_is_claimed_once_whole_and_oldest_first(tmp_path):
    inbox = WorkDirInbox(str(tmp_path))
    first = JoinRequest(Selection(ids=('a', 'b'), kind='movie', year='>=2000'), 'extract', retry_failed=True)
    second = JoinRequest(Selection(needs=('extract',)))
    inbox.post(first)
    os.utime(os.path.join(inbox_path(str(tmp_path)), f'{first.id}.json'), (1, 1))
    inbox.post(second)

    assert inbox.state(first.id) == PENDING
    assert inbox.claim() == [first, second]
    assert inbox.claim() == [] and inbox.state(first.id) == TAKEN
    assert not inbox.withdraw(first.id)   # a claimed request cannot be taken back


def test_a_withdrawn_request_is_never_claimed_and_forget_tidies_up(tmp_path):
    inbox = WorkDirInbox(str(tmp_path))
    request = JoinRequest(Selection(ids=('a',)))
    inbox.post(request)

    assert inbox.withdraw(request.id) and inbox.state(request.id) == WITHDRAWN_STATE
    assert inbox.claim() == []
    inbox.forget(request.id)
    assert inbox.state(request.id) is None and os.listdir(inbox_path(str(tmp_path))) == []


def test_a_request_that_cannot_be_read_is_set_aside_not_claimed(tmp_path):
    inbox = WorkDirInbox(str(tmp_path))
    os.makedirs(inbox_path(str(tmp_path)))
    with open(os.path.join(inbox_path(str(tmp_path)), 'broken.json'), 'w') as f:
        f.write('{ not json')

    assert inbox.claim() == [] and os.listdir(inbox_path(str(tmp_path))) == ['broken.bad']


def test_the_inbox_is_a_source_of_a_join_queue(tmp_path):
    inbox = WorkDirInbox(str(tmp_path))
    join = JoinQueue(sources=[inbox.claim])
    request = JoinRequest(Selection(ids=('a',)))
    inbox.post(request)

    assert join.take() == [request]
    join.close()
    inbox.post(JoinRequest(Selection(ids=('b',))))
    assert join.take() == []   # a closed queue claims nothing: the poster can still withdraw it


def test_handing_off_waits_for_the_run_that_claimed_it_to_end(tmp_path):
    work = str(tmp_path)
    lease = WorkDirLease(work, 'job-1', host='nas', pid=1).__enter__()
    request = JoinRequest(Selection(ids=('a',)))
    seen = []

    def other_run():
        inbox = WorkDirInbox(work)
        while not inbox.claim():
            threading.Event().wait(0.01)
        seen.append('claimed')
        lease.__exit__(None, None, None)

    thread = threading.Thread(target=other_run)
    thread.start()
    assert hand_off(work, request, poll_seconds=0.01, on_claimed=lambda: seen.append('told')) == JOINED
    thread.join()
    assert seen[0] == 'claimed' and 'told' in seen and WorkDirInbox(work).state(request.id) is None


def test_handing_off_takes_the_request_back_when_the_run_ends_without_claiming_it(tmp_path):
    work = str(tmp_path)
    lease = WorkDirLease(work, 'job-1', host='nas', pid=1).__enter__()
    request = JoinRequest(Selection(ids=('a',)))
    timer = threading.Timer(0.05, lambda: lease.__exit__(None, None, None))
    timer.start()

    assert hand_off(work, request, poll_seconds=0.01) == WITHDRAWN
    timer.join()
    assert WorkDirInbox(work).state(request.id) is None and read_lease(work) is None


def test_the_lease_says_who_holds_it():
    from pipeline.service.lease import LeaseHolder
    assert LeaseHolder('desk', 1, 'worklist-abc', 0).who() == 'the work list on desk'
    assert LeaseHolder('desk', 1, 'cli-abc', 0).describe() == \
        'a command-line run on desk is running in this work directory; try again when it has finished'
    assert LeaseHolder('nas', 1, 'job-12345678', 0).who() == 'the pipeline service on nas (job-1234)'


def test_a_run_lease_is_named_for_its_kind(tmp_path):
    with run_lease(str(tmp_path), 'cli'):
        assert read_lease(str(tmp_path)).job_id.startswith('cli-')
    assert read_lease(str(tmp_path)) is None


def test_what_became_of_handed_off_titles_depends_on_how_far_they_were_to_go():
    from types import SimpleNamespace
    from pipeline.library.handoff import outcomes

    def row(title_id, needs, extract_state='current', design_state='none'):
        return SimpleNamespace(id=title_id, needs=needs, detail=needs, extract_state=extract_state,
                               design_state=design_state)

    index = SimpleNamespace(titles=lambda ids: [row('a', 'design'), row('b', 'review', design_state='current'),
                                                row('c', 'extract', extract_state='failed')])
    ids = ['a', 'b', 'c', 'gone']

    assert {i: o.failed for i, o in outcomes(index, ids, 'extract').items()} == \
        {'a': False, 'b': False, 'c': True, 'gone': True}
    assert {i: o.failed for i, o in outcomes(index, ids, 'design').items()} == \
        {'a': True, 'b': False, 'c': True, 'gone': True}
