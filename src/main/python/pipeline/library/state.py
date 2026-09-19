'''
What a title needs next -- design/library-sync/workflow-rework/design.md §12.6.

A title has a state per stage (extract, design, review, publish, commit) and some orthogonal flags. What the work
list shows -- and what `--needs` selects -- is the single next thing the title needs, derived from those by
derive_needs(). It is a **pure function**: nothing here reads a file, a queue entry or git; discovery (status.py,
index.py) reads the outputs, fills a StageStates in and calls it, so "what would run" is decided in one place and
can be tested with no disk at all.

Stage states (strings, so they go into the index unchanged):

    extract  none | current | stale | failed
    design   none | current | stale | failed | protected      (protected: accepted or published)
    review   none | pending | accepted | skipped | rejected   (none: no queue entry yet)
    publish  none | not_written | written | out_of_date
    commit   none | uncommitted | committed | pushed | unknown  (committed means committed but not pushed)

`failed` is a *remembered* failure that still applies: the source and settings it failed against are unchanged.
'''
from dataclasses import dataclass
from typing import Optional, Tuple

EXTRACT_STATES = ('none', 'current', 'stale', 'failed')
DESIGN_STATES = ('none', 'current', 'stale', 'failed', 'protected')
REVIEW_STATES = ('none', 'pending', 'accepted', 'skipped', 'rejected')
PUBLISH_STATES = ('none', 'not_written', 'written', 'out_of_date')
COMMIT_STATES = ('none', 'uncommitted', 'committed', 'pushed', 'unknown')

# `needs` values, in the order the pipeline strip shows them; the CLI's --needs vocabulary
NEEDS = ('attention', 'review', 'extract', 'design', 'publish', 'commit', 'done')
TIERS = ('attention', 'human', 'machine', 'done')
TIER_OF_NEEDS = {'attention': 'attention', 'review': 'human', 'extract': 'machine', 'design': 'machine',
                 'publish': 'machine', 'commit': 'machine', 'done': 'done'}
TIER_ORDER = {tier: rank for rank, tier in enumerate(TIERS)}  # the work list's sort: attention first, done last

# the orthogonal flags, by the name a work list shows
FLAG_IGNORED = 'Ignored'
FLAG_SHADOWED = 'Shadowed'
FLAG_GONE = 'Gone'
FLAG_DUPLICATE = 'Possible duplicate'
FLAG_IN_CATALOGUE = 'Already in catalogue'


@dataclass(frozen=True)
class StageStates:
    ''' Everything derive_needs() looks at. The defaults are a brand new title nothing has happened to. '''
    extract: str = 'none'
    design: str = 'none'
    review: str = 'none'
    publish: str = 'none'
    commit: str = 'none'
    # flags that take a title out of the work altogether
    ignored: str = ''                       # why (the rule, or "ignored by you"), '' if not ignored
    shadowed_by: str = ''                   # the id of the title that owns this file, '' if not shadowed
    gone: bool = False                      # the item has left its source but its outputs remain
    # facts
    failure: str = ''                       # the remembered failure's message, for extract/design == 'failed'
    project_conflict: bool = False          # the mono and multichannel projects were edited and now disagree
    source_changed: bool = False            # the source differs from the one an accepted/published title was designed on
    metadata_problems: Tuple[str, ...] = ()  # pipeline.metadata.validate(), for a title with an entry
    decline: str = ''                       # the designer's decline reason, if it declined
    confidence: Optional[float] = None      # the top candidate's confidence
    candidates: int = 0
    out_of_date: str = ''                   # why a published title's catalogue copy is out of date
    commit_detail: str = ''                 # why the commit state is what it is, where that is not obvious


@dataclass(frozen=True)
class Needs:
    needs: str    # one of NEEDS
    tier: str     # one of TIERS
    detail: str   # the one-line reason a work list shows next to it


def _needs(needs: str, detail: str) -> Needs:
    return Needs(needs, TIER_OF_NEEDS[needs], detail)


def derive_needs(s: StageStates) -> Needs:
    '''
    The first matching row, top down (design.md §12.6). A title that is *not for this catalogue* (ignored, shadowed
    by another source's copy of the file, gone from its source) is done whatever its stages say.

    A title that has been decided (accepted, published, skipped, rejected) is not made stale by a *settings*
    change -- only by its source changing, which is `attention` -- so extract and design staleness only count for a
    title still in play (no entry yet, or pending).
    '''
    if s.ignored:
        return _needs('done', s.ignored)
    if s.shadowed_by:
        return _needs('done', f'shadowed: the same file is title {s.shadowed_by}')
    if s.gone:
        return _needs('done', 'gone from source')

    # attention: something a person must look at, because the machine cannot get past it
    for stage in ('extract', 'design'):
        if getattr(s, stage) == 'failed':
            return _needs('attention', f'{stage} failed: {s.failure}' if s.failure else f'{stage} failed')
    if s.project_conflict:
        return _needs('attention', 'the mono and multichannel projects were edited independently and disagree')
    if s.source_changed and s.review == 'accepted':
        return _needs('attention', 'source changed since accepted' if s.publish in ('none', 'not_written')
                      else 'source changed since published')

    in_play = s.review in ('none', 'pending')

    # human
    problems = '; '.join(s.metadata_problems)
    if s.review == 'pending' and s.design == 'current':
        if s.decline:
            detail = f'designer declined: {s.decline}'
        else:
            detail = (f'conf {s.confidence:.2f} - ' if s.confidence is not None else '') + \
                     f'{s.candidates} candidate{"" if s.candidates == 1 else "s"}'
        return _needs('review', f'{detail} - metadata incomplete: {problems}' if problems else detail)
    if s.review == 'accepted' and problems:
        return _needs('review', f'metadata incomplete: {problems}')

    # machine
    if in_play and s.extract in ('none', 'stale'):
        return _needs('extract', 'new' if s.extract == 'none' and s.review == 'none' else
                      'not extracted' if s.extract == 'none' else 'source or settings changed')
    if in_play and s.design in ('none', 'stale'):
        return _needs('design', 'new' if s.design == 'none' else 'source or settings changed')
    if s.review == 'accepted' and s.publish in ('none', 'not_written'):
        return _needs('publish', 'accepted, not written to the repository')
    if s.publish == 'out_of_date':
        return _needs('publish', s.out_of_date or 'changed since published')
    if s.publish == 'written' and s.commit in ('uncommitted', 'committed', 'unknown'):
        return _needs('commit', {'uncommitted': 'written, not committed',
                                 'committed': 'committed, not pushed',
                                 'unknown': s.commit_detail or 'cannot tell whether it is committed and pushed'
                                 }[s.commit])

    # done
    if s.review in ('skipped', 'rejected'):
        return _needs('done', s.review)
    if s.commit == 'pushed':
        return _needs('done', 'pushed')
    return _needs('done', '')
