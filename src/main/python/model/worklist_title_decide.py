'''
Deciding a title on the title page -- design/archive/library-sync/workflow-rework/design.md §12.10, chunks 27a and 27b: **Accept**, **Skip** and
**Reject**, the rule for which title is "next", and what is checked before a decision is written. A mixin of
`model.worklist_title.TitlePage` (which owns the widgets, the entry and the signals), split out of it so each file stays readable.

The queue entry is the truth for a decision: `pipeline.library.decide.decide()` (shared with the pipeline service) reads it again
before it is written, and nothing is written if its status no longer allows the decision, its candidates are not the ones on screen
(it was redesigned), or -- for an accept -- its metadata is incomplete.
Accepting a design the designer rejected (contract 1.1) is an override: the page's `confirm_override` asks first, with the reasons.
'''
import logging
from typing import Mapping

from model.worklist_title_text import override_question
from pipeline.library.decide import DECISION_FROM, DECISION_STATUS, DecisionRefused, decide, decision_blocked, \
    next_waiting_id, offered_digest, sent_back  # noqa: F401 (DECISION_FROM: re-exported for the title page)
from pipeline.library.index import TitleRow
from pipeline.review import read_entry

logger = logging.getLogger('worklist')


class TitleDecisions:
    '''
    The mixin: it uses the page's `_queue_dir`, `_title_id`, `_entry`, `_picked`, `_rows`, `_ids`, `_running`, `_problems`, `_defaults`,
    `_decided_here`, `_revised_here`, `_hooks`, `rightTabs`, `_metadata`, `_flush_for_move()`, `_flushed_or_discarded()`, `_say()`,
    `reload()`, `show_title()` and the signal `decided`.
    '''

    def _probably_waiting(self, title_id: str, rows: Mapping[str, TitleRow]) -> bool:
        '''
        From the rows alone (no file read): the index says a person is to review it (not a pending entry whose design is
        out of date or failed, which is not offered) and it was not decided on this page.
        '''
        row = rows.get(title_id)
        return title_id not in self._decided_here and not sent_back(self._revised_here.get(title_id, '')) \
            and row is not None and row.review_state == 'pending' and row.needs == 'review'

    def _waiting(self, title_id: str, rows: Mapping[str, TitleRow]) -> bool:
        ''' Whether the title needs a decision now: the rows narrow it down, and the entry has the last word. '''
        if not self._probably_waiting(title_id, rows):
            return False
        try:
            return read_entry(self._queue_dir(), title_id).status == 'pending'
        except Exception:   # unreadable or gone: the person can look at it by stepping to it, but it is not "next"
            return False

    # --- decisions --------------------------------------------------------------------------------------------------------

    def accept(self) -> bool:
        '''
        Accepts the highlighted candidate and goes to the next title waiting for a decision. A design the designer rejected is
        accepted only if `confirm_override` says so: nothing is written otherwise.
        '''
        return self._decide('accept')

    def skip(self) -> bool:
        '''
        Goes to the next title waiting for a decision, leaving this one as it is: Skip is moving on, not a decision (it used to
        write `skipped`, which took the title off the list of those waiting).
        :return: False, and stays, if no other title is waiting or what is being edited could not be saved.
        '''
        rows = self._rows()
        following = next_waiting_id(self._ids, self._title_id, lambda i: self._waiting(i, rows))
        if following is None:
            self._say('No other title in this list is waiting for a decision.')
            return False
        return self.show_title(following)

    def _can_skip(self, rows: Mapping[str, TitleRow]) -> bool:
        ''' From the rows alone: another listed title probably waits for a decision. '''
        return any(title_id != self._title_id and self._probably_waiting(title_id, rows) for title_id in self._ids)

    def reject(self) -> bool:
        return self._decide('reject')

    def _decide(self, decision: str) -> bool:
        '''
        :return: True if the decision was written. What is being edited is saved first (a decision that cannot save it is not
        made). The entry is read again, and nothing is written if it is not in a state this decision applies to, its
        candidates are not the ones on screen (it was redesigned), or -- for an accept -- its metadata is incomplete: the page
        then shows what is there now.
        '''
        queue_dir, title_id, seen = self._queue_dir(), self._title_id, self._entry
        if not queue_dir or seen is None:
            return False
        picked, word = self._picked, DECISION_STATUS[decision]
        # An edit is never lost, and no decision can leave a person stuck behind one: Accept needs the edit saved (its metadata is
        # what is being accepted), so it waits; Skip and Reject do not, so if the edit cannot be saved they ask whether to drop it.
        if not (self._flush_for_move() if decision == 'accept' else self._flushed_or_discarded()):
            return False
        shown = self._entry
        if shown is None or shown.offered != seen.offered:   # saving the edit found a redesign, and showed it
            self._say(f'Not {word}: the design of this title changed while it was open. Look at it again.', problem=True)
            return False
        blocked = decision_blocked(decision, shown, self._rows().get(title_id), title_id in self._running(),
                                   self._problems, self._revised_here.get(title_id, ''), self._hooks.redo)
        if blocked:
            self._say(blocked, problem=True)
            if decision == 'accept' and self._problems:
                # show what is missing, but leave the keyboard where it is: the key that was refused (A) must not be followed by
                # the same key typed into the box it was refused for
                self.rightTabs.setCurrentIndex(1)
                self._metadata.highlight_problem()
            return False
        if decision == 'accept' and len(shown.candidates) <= picked < len(shown.offered) and \
                not self.confirm_override(override_question(picked, shown.offered[picked])):
            self._say('Not accepted: the designer rejected this design, and it was not overridden.')
            return False
        try:
            # the override was asked above, so the write half (pipeline.library.decide) is told it was given
            decide(queue_dir, title_id, decision, seen_digest=offered_digest(shown), picked=picked,
                   row=self._rows().get(title_id), running=title_id in self._running(),
                   revised=self._revised_here.get(title_id, ''), redo=self._hooks.redo, meta_defaults=self._defaults(),
                   override_rejection=True)
        except DecisionRefused as refused:
            if refused.kind == 'blocked':
                self._say(refused.reason, problem=True)
                return False
            return self._changed(refused.reason)
        except Exception as error:   # a full disk, a permission, a file damaged since it was read
            logger.exception('Could not %s %s', decision, title_id)
            self._say(f'Not saved: {type(error).__name__}: {error}', problem=True)
            return False
        self._decided_here.add(title_id)
        self.decided.emit(title_id, DECISION_STATUS[decision])
        rows = self._rows()
        following = next_waiting_id(self._ids, title_id, lambda i: self._waiting(i, rows))
        if following is not None:
            self.show_title(following)
        else:
            self.reload()
            self._say('No other title in this list is waiting for a decision.')
        return True

    def _changed(self, message: str) -> bool:
        self.reload()
        self._say(message, problem=True)
        return False
