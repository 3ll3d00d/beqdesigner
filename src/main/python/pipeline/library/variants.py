'''
A title's design variants (design/web-parity.md WP7): every design run of it is kept until a person discards it, so a
redesign -- with adjustments or without -- can be compared with what it replaced, and the earlier one taken back.

The queue entry's own design fields are the **current** variant (`QueueEntry.variant_id`): what is reviewed, decided,
published and charted, as before, so nothing that reads an entry has to know about variants. `QueueEntry.variants` holds
the others, each those same fields as a dict (`VARIANT_FIELDS`), newest last. A redesign keeps the current one there
before it replaces it (`kept_variants()`, called by `design_cache.design_if_needed`); `use_variant()` swaps one back in;
`discard_variant()` forgets one. The adjustments a design was made with (`adjusted_with`) go with its variant, so they
are never lost; using a variant asks them again of the next design.
'''
import re
from dataclasses import asdict, fields, replace
from typing import List, Optional

from pipeline.review import CandidateSummary, QueueEntry, read_entry, write_queue_entry

# what one design run is: the entry's fields a redesign replaces
VARIANT_FIELDS = ('variant_id', 'designed_at', 'fs', 'curve', 'peak_curve', 'audio_stream', 'candidates', 'rejected',
                  'decline_reason', 'decline_message', 'design_fingerprint', 'source_fingerprint', 'bass_management',
                  'designer', 'designer_build', 'design_version', 'design_settings', 'adjusted_with',
                  'adjustment_outcomes', 'run_record')
DECIDED = ('accepted', 'published')


class VariantRefused(ValueError):
    ''' Why a variant cannot be used or discarded now, in words. '''


def _number(variant_id: Optional[str]) -> int:
    match = re.fullmatch(r'v(\d+)', variant_id or '')
    return int(match[1]) if match else 0


def next_variant_id(entry: Optional[QueueEntry]) -> str:
    ''' The id a new design run of the title gets: one more than any it has had. '''
    if entry is None:
        return 'v1'
    current = entry.variant_id or ('v1' if designed(entry) else None)
    return f'v{max([_number(current)] + [_number(v.get("variant_id")) for v in entry.variants]) + 1}'


def designed(entry: QueueEntry) -> bool:
    ''' Whether the entry holds a design run at all (an entry is only ever written by one, but be sure). '''
    return bool(entry.curve) and (bool(entry.candidates) or entry.declined)


def variant_of(entry: QueueEntry) -> dict:
    ''' The entry's current design run as a variant; one designed before variants were kept is named v1. '''
    d = asdict(entry)
    variant = {name: d[name] for name in VARIANT_FIELDS}
    variant['variant_id'] = entry.variant_id or 'v1'
    return variant


def kept_variants(existing: Optional[QueueEntry]) -> List[dict]:
    ''' What a redesign of `existing` keeps: its variants, and the current design run too. '''
    if existing is None:
        return []
    return [*existing.variants, variant_of(existing)] if designed(existing) else list(existing.variants)


def _with_variant(entry: QueueEntry, variant: dict) -> QueueEntry:
    ''' `entry` with `variant`'s design fields in place of its own (the person's fields -- metadata, artwork, note -- kept). '''
    values = {name: variant.get(name) for name in VARIANT_FIELDS}
    values['candidates'] = [CandidateSummary(**_known(c)) for c in values['candidates'] or []]
    values['rejected'] = [CandidateSummary(**_known(c)) for c in values['rejected'] or []]
    values['adjusted_with'] = list(values['adjusted_with'] or [])
    values['adjustment_outcomes'] = list(values['adjustment_outcomes'] or [])
    # what the next design is asked is what this one was made with: so it is current, and a redesign asks the same
    return replace(entry, adjustments=list(values['adjusted_with']), **values)


_CANDIDATE_FIELDS = None


def _known(candidate: dict) -> dict:
    global _CANDIDATE_FIELDS
    if _CANDIDATE_FIELDS is None:
        _CANDIDATE_FIELDS = {f.name for f in fields(CandidateSummary)}
    return {k: v for k, v in candidate.items() if k in _CANDIDATE_FIELDS}


def variant_offered(variant: dict):
    ''' A kept design run's candidates and rejected designs, as the entry's own (CandidateSummary). '''
    return ([CandidateSummary(**_known(c)) for c in variant.get('candidates') or []],
            [CandidateSummary(**_known(c)) for c in variant.get('rejected') or []])


def variant_view(entry: QueueEntry, variant_id: str) -> QueueEntry:
    '''
    The entry as it would be with variant `variant_id` current -- for charting an earlier design run beside this one.
    :raises KeyError: if the title has no such variant.
    '''
    if variant_id == (entry.variant_id or 'v1') and designed(entry):
        return entry
    variant = next((v for v in entry.variants if v.get('variant_id') == variant_id), None)
    if variant is None:
        raise KeyError(f'{entry.id} has no variant {variant_id}')
    return replace(_with_variant(entry, variant), status='pending', chosen_candidate_index=None)


def use_variant(queue_dir: str, entry_id: str, variant_id: str) -> QueueEntry:
    '''
    Makes variant `variant_id` the current one again, keeping the current one as a variant. The title waits for a person
    to decide again (`pending`).
    :raises VariantRefused: for an accepted or published title (reopen it first: its catalogue files stand for the
        current one), or a variant it does not have.
    '''
    entry = read_entry(queue_dir, entry_id)
    if entry.status in DECIDED:
        raise VariantRefused(f'{entry_id} is {entry.status}: reopen it for review before using another variant')
    if variant_id == (entry.variant_id or 'v1'):
        return entry
    others = [v for v in entry.variants if v.get('variant_id') != variant_id]
    if len(others) == len(entry.variants):
        raise VariantRefused(f'{entry_id} has no variant {variant_id}')
    chosen = next(v for v in entry.variants if v.get('variant_id') == variant_id)
    swapped = replace(_with_variant(entry, chosen), status='pending', chosen_candidate_index=None,
                      variants=[*others, variant_of(entry)] if designed(entry) else others)
    write_queue_entry(queue_dir, swapped)
    return swapped


def discard_variant(queue_dir: str, entry_id: str, variant_id: str) -> QueueEntry:
    '''
    Forgets variant `variant_id`. The current one cannot be discarded: use another first.
    :raises VariantRefused: for the current variant, or one the title does not have.
    '''
    entry = read_entry(queue_dir, entry_id)
    if variant_id == (entry.variant_id or 'v1'):
        raise VariantRefused(f'{variant_id} is the current design of {entry_id}: use another before discarding it')
    kept = [v for v in entry.variants if v.get('variant_id') != variant_id]
    if len(kept) == len(entry.variants):
        raise VariantRefused(f'{entry_id} has no variant {variant_id}')
    updated = replace(entry, variants=kept)
    write_queue_entry(queue_dir, updated)
    return updated


def save_edited(queue_dir: str, entry_id: str, source: int, specs) -> QueueEntry:
    '''
    A person's edit of design `source` (an index into the current run's offered designs, a decline's flat one included):
    a new design run, current, whose one design is the edit (`method` 'edited', `commentary['edited_from']` saying which),
    the current one kept as a variant. It keeps the run's measured curves, designer and identity -- the design is still
    of that audio, by that designer, then edited -- so it is not out of date; its fit figures, which were the designer's,
    are dropped. The title waits for a person to decide.
    :param specs: `contract.BiquadSpec`s, checked by `convert.specs_to_complete_filter`.
    :raises VariantRefused: for an accepted or published title, or no such design.
    :raises ContractViolation: for filters a designer could not have returned.
    '''
    from datetime import datetime, timezone
    from pipeline.designer.convert import specs_to_complete_filter
    entry = read_entry(queue_dir, entry_id)
    if entry.status in DECIDED:
        raise VariantRefused(f'{entry_id} is {entry.status}: reopen it for review before editing a design')
    if not 0 <= source < len(entry.offered):
        raise VariantRefused(f'{entry_id} offers {len(entry.offered)} design(s), not {source + 1}')
    picked = entry.offered[source]
    filters = specs_to_complete_filter(list(specs), entry.fs)
    edited = CandidateSummary(filters=filters.to_json(), confidence=picked.confidence, method='edited',
                              mv_adjust_db=picked.mv_adjust_db, gain_reduction_db=picked.gain_reduction_db,
                              commentary={'edited_from': f'{entry.variant_id or "v1"} design {source + 1}'})
    updated = replace(entry, candidates=[edited], rejected=[], decline_reason=None, decline_message=None,
                      adjustment_outcomes=[], run_record=None, variant_id=next_variant_id(entry), status='pending',
                      chosen_candidate_index=None, designed_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                      variants=kept_variants(entry))
    write_queue_entry(queue_dir, updated)
    return updated
