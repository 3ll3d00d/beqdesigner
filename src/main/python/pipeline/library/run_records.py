'''
Giving designs made before contract 1.6 their run record (design/redesign-scope.md, step 1 of the live check).

A design made by beqforge 0.2.10 or earlier has no `run_record` on its queue entry, so `/impact` cannot be asked about it,
but beqforge named the record in the design's commentary, as an absolute path on the machine that designed it
(`/work/records/designer-<digest>.run.json.gz` from the image, `\\\\HOST\\beq\\work\\records\\...` from the exe). The
record is in the work directory's `records/` either way, so the entry's `run_record` is `records/<that name>`: relative to
the work directory, the designer's shared root (§7.1). beqforge checked that each record holds its entry's current design.

Only the current design is given one: an earlier run's record has since been overwritten by the same request designed
again. An entry that names no record, names two, already has one, or whose record is not in the work directory is left
as it is, and says why. Run it with no run in progress, then rescan so the index carries the records.

    PYTHONPATH=src/main/python python -m pipeline.library.run_records --queue-dir Q --work-dir W [--apply]
'''
import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

from pipeline.review import read_entry, update_entry

RECORDS = 'records'
# the record's file name, as beqforge names it in commentary: by audio digest before 0.2.11, then also request and version
_NAME = re.compile(r'designer-[0-9a-f]+(?:-[0-9a-f]+)*(?:-v\d+)?\.run\.json\.gz')


@dataclass(frozen=True)
class Found:
    id: str
    run_record: Optional[str]   # the path to set; None when it is left as it is
    why: str                    # what was done, or why not, in words


def named_records(entry: dict) -> List[str]:
    ''' The record file names an entry's current design gives, in its own fields: an earlier run's (variants) are not its. '''
    return sorted(set(_NAME.findall(json.dumps({k: v for k, v in entry.items() if k != 'variants'}))))


def record_for(entry: dict, work_dir: str, exists: Callable[[str], bool] = os.path.isfile) -> Found:
    ''' The run record `entry` (a queue entry as stored) should name, or why it is left as it is. '''
    entry_id = entry.get('id', '')
    if entry.get('run_record'):
        return Found(entry_id, None, 'already has a run record')
    if entry.get('design_version') is None and not entry.get('candidates') and not entry.get('decline_reason'):
        return Found(entry_id, None, 'not designed')
    names = named_records(entry)
    if not names:
        return Found(entry_id, None, 'names no run record')
    if len(names) > 1:
        return Found(entry_id, None, f"names {len(names)} run records: {', '.join(names)}")
    if not exists(os.path.join(work_dir, RECORDS, names[0])):
        return Found(entry_id, None, f'{names[0]} is not in {os.path.join(work_dir, RECORDS)}')
    return Found(entry_id, f'{RECORDS}/{names[0]}', 'given its run record')


def entry_ids(queue_dir: str) -> List[str]:
    return sorted(n[:-len('.json')] for n in os.listdir(queue_dir) if n.endswith('.json') and not n.startswith('.'))


def backfill(queue_dir: str, work_dir: str, *, apply: bool = False, ids: Optional[Iterable[str]] = None,
             on_entry: Optional[Callable[[Found], None]] = None) -> List[Found]:
    '''
    Works out each entry's run record and, with `apply`, writes it. Nothing else in an entry changes.
    :param ids: the entries to look at; every entry in the queue by default.
    '''
    out = []
    for entry_id in (entry_ids(queue_dir) if ids is None else ids):
        with open(os.path.join(queue_dir, f'{entry_id}.json'), encoding='utf-8') as f:
            found = record_for(json.load(f), work_dir)
        if apply and found.run_record:
            if read_entry(queue_dir, entry_id).run_record:   # given one since it was read: a design finished meanwhile
                found = Found(entry_id, None, 'already has a run record')
            else:
                update_entry(queue_dir, entry_id, run_record=found.run_record)
        out.append(found)
        if on_entry:
            on_entry(found)
    return out


def summary(found: List[Found]) -> Counter:
    ''' How many entries came out each way: the reasons that name a file are counted under their common words. '''
    return Counter(re.sub(r'designer-\S+|/\S+', '…', f.why) for f in found)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Give designs made before contract 1.6 the run record their '
                                                 'commentary names (design/redesign-scope.md).')
    parser.add_argument('--queue-dir', required=True)
    parser.add_argument('--work-dir', required=True, help="the work directory: the designer's shared root")
    parser.add_argument('--apply', action='store_true', help='write them; without it, only say what would be written')
    parser.add_argument('--list', action='store_true', help='print every entry, not just the counts')
    args = parser.parse_args(argv)
    found = backfill(args.queue_dir, args.work_dir, apply=args.apply,
                     on_entry=(lambda f: print(f'{f.id}\t{f.run_record or "-"}\t{f.why}')) if args.list else None)
    for why, n in summary(found).most_common():
        print(f'{n:6d}  {why}')
    print(('written' if args.apply else 'dry run: nothing written') + f'; {len(found)} entries')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
