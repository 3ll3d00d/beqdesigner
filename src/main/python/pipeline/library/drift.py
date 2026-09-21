'''
Which accepted or published titles were designed under other settings than the ones in force now --
design/library-sync/workflow-rework/design.md §12.6 ("Done titles and staleness") and §12.8.

An accepted or published title is *protected*: a run never redesigns it, and a change to the designer, the analysis or the
coverage deliberately does not put it back in the work list (it would put every done title there at once). The work list
instead says, once, "settings changed since N titles were designed" and offers to revise them; this module is what counts
them. It reads the index's rows and the queue entries and changes nothing, and the index schema has no column for it (it is
frozen), so it is worked out when asked.

A title counts when the fingerprint its entry recorded when it was designed (`QueueEntry.design_fingerprint`) is not the
one a design run would record for it *now*, with the settings given and **the source it was designed from** (the entry's
recorded `source_fingerprint`). Taking the source from the entry, not from the library, is what keeps this about the
*settings*: a source that changed since is a different thing (the title is `attention`, "source changed since accepted")
and is not counted here. What it cannot tell, it does not claim:

- a title whose entry has no design fingerprint or no source fingerprint (designed before they were recorded) -- no false
  alarms, as the index treats them;
- a TV season built from several episodes: its design was made from the joined track, whose fingerprint is not kept with
  the entry, so a season is never counted (its `unit` is `season`);
- a row that came from outputs alone (the index has no items for it).
'''
import os
from typing import List, Optional, Sequence

from pipeline.library.design_cache import PROTECTED_STATUSES, design_fingerprint
from pipeline.library.extract_cache import read_manifest
from pipeline.library.index import LibraryIndex, TitleRow
from pipeline.library.status import EntryFacts, ScanSettings, read_entry_facts


def _multichannel_variants(settings: ScanSettings, item_dir: str) -> List[bool]:
    '''
    Whether the design used a multichannel extraction, as discovery decides it (`status.Evaluator.evaluate`). Only a setting that
    keeps the multichannel extraction can make it multichannel, so the manifest is not read (one small file per title) when it is
    off: the answer is `[False]` whatever the manifest says.
    '''
    if not settings.keep_multichannel:
        return [False]
    try:
        manifest = read_manifest(item_dir) if os.path.isdir(item_dir) else {}
    except (OSError, ValueError):   # an unreadable manifest: discovery would call the extraction stale; here it says nothing
        manifest = {}
    count = manifest.get('source_channel_count')
    kept = bool(settings.keep_multichannel and 'multichannel_source_fingerprint' in manifest and count != 1)
    return [kept] + ([not kept] if count is None and kept else [])


def designed_under_other_settings(index: LibraryIndex, settings: ScanSettings,
                                  rows: Optional[Sequence[TitleRow]] = None) -> List[str]:
    '''
    :param settings: what a scan and a run are given now (the designer, the analysis, the coverage, the directories).
    :param rows: the index's rows if the caller has them (default: all of them).
    :return: the ids of the accepted or published titles whose design was made under other settings, in the rows' order.
    '''
    rows = list(index.titles() if rows is None else rows)
    candidates = [row for row in rows if row.review_state == 'accepted' and row.design_state == 'protected'
                  and row.unit == 'item' and not (row.ignored or row.shadowed_by or row.gone)]
    if not candidates or not settings.queue_dir:
        return []
    ids = [row.id for row in candidates]
    units = index.units(ids)
    # what the last scan read of each entry: an entry whose file has not changed since is not parsed again (its curve is most
    # of it), which is what keeps this cheap over thousands of accepted titles on every read of the index
    summaries = index.entry_summaries(ids)
    found: List[str] = []
    for row in candidates:
        item = units.get(row.id)
        cached = EntryFacts.from_json(summaries[row.id]) if row.id in summaries else None
        facts = read_entry_facts(settings.queue_dir, row.id, cached)
        if item is None or facts is None or facts.status not in PROTECTED_STATUSES \
                or not facts.design_fingerprint or not facts.source_fingerprint:
            continue
        variants = _multichannel_variants(settings, os.path.join(settings.work_dir, row.id) if settings.work_dir else '')
        now = {design_fingerprint(item, settings.designer, settings.config, settings.coverage, multichannel=mc,
                                  source=facts.source_fingerprint) for mc in variants}
        if facts.design_fingerprint not in now:
            found.append(row.id)
    return found
