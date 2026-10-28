'''
pipeline/library/variants.py -- every design run of a title kept until a person discards it (design/web-parity.md WP7):
a redesign keeps the one it replaces, with its adjustments; an earlier one can be used again or discarded.
'''
import pytest

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import AdjustmentOutcome, BiquadSpec, DesignCandidate, DesignResponse
from pipeline.designer.registry import register_designer, unregister_designer
from pipeline.library.design_cache import design_if_needed
from pipeline.library.source import LibraryItem
from pipeline.library.variants import VariantRefused, discard_variant, discard_variants, next_variant_id, use_variant, \
    variant_view
from pipeline.orchestrate import Session
from pipeline.review import read_entry, update_entry
from review_entry_fixture import write_entry
from test_pipeline_orchestrate import _write_mono_wav

DESIGNER = 'test.variants.designer'
ASKED = [{'id': 'a1', 'kind': 'max_boost', 'params': {'db': 10}, 'text': None}]


@pytest.fixture(autouse=True)
def _designer():
    gains = iter([4.0, 6.0, 8.0, 10.0])

    def design(request):
        return DesignResponse(contract_version=request.contract_version, candidates=[DesignCandidate(
            filters=[BiquadSpec(type='low_shelf', freq_hz=18.0, gain_db=next(gains), q=0.7)], confidence=0.6,
            mv_adjust_db=4.0, method='fitted')],
            adjustment_outcomes=[AdjustmentOutcome(a.id, 'honoured') for a in request.adjustments or ()] or None)
    register_designer(DESIGNER, design)
    yield
    unregister_designer(DESIGNER)


@pytest.fixture
def title(tmp_path):
    wav = str(tmp_path / 'mono.wav')
    _write_mono_wav(wav)
    source = tmp_path / 'heat.mkv'
    source.write_bytes(b'x')
    item = LibraryItem(id='heat', source_path=str(source), display_name='Heat', fingerprint='s')
    queue = str(tmp_path / 'q')

    def redesign(adjustments=None):
        if adjustments is not None:
            update_entry(queue, 'heat', adjustments=adjustments)
        return design_if_needed(Session(), item, wav, DESIGNER, queue, AnalysisConfig(), force=True).entry
    return queue, redesign


def _gain(candidate):
    return candidate['filters']['filters'][0]['gain'] if isinstance(candidate, dict) else candidate.filters['filters'][0]['gain']


def test_each_redesign_keeps_the_one_it_replaces_with_its_adjustments(title):
    queue, redesign = title
    first = redesign()
    assert (first.variant_id, first.variants, first.designed_at is not None) == ('v1', [], True)
    second = redesign(ASKED)
    assert second.variant_id == 'v2' and [v['variant_id'] for v in second.variants] == ['v1']
    assert second.adjustments == second.adjusted_with == ASKED and second.adjustment_outcomes[0]['outcome'] == 'honoured'
    assert second.variants[0]['adjusted_with'] == [] and _gain(second.variants[0]['candidates'][0]) == 4.0
    third = redesign([])
    assert [v['variant_id'] for v in third.variants] == ['v1', 'v2'] and third.variants[1]['adjusted_with'] == ASKED


def test_an_earlier_variant_can_be_used_again_and_the_current_one_is_kept(title):
    queue, redesign = title
    redesign()
    redesign(ASKED)
    entry = use_variant(queue, 'heat', 'v1')
    assert entry.variant_id == 'v1' and entry.adjustments == [] and _gain(entry.candidates[0]) == 4.0
    assert [v['variant_id'] for v in entry.variants] == ['v2'] and entry.variants[0]['adjusted_with'] == ASKED
    assert read_entry(queue, 'heat').variant_id == 'v1'
    assert redesign().variant_id == 'v3'                                      # numbers are never reused


def test_a_decided_title_is_reopened_before_another_variant_is_used(title):
    queue, redesign = title
    redesign()
    redesign()
    update_entry(queue, 'heat', status='accepted', chosen_candidate_index=0)
    with pytest.raises(VariantRefused, match='reopen it'):
        use_variant(queue, 'heat', 'v1')


def test_a_variant_is_discarded_only_when_it_is_not_the_current_one(title):
    queue, redesign = title
    redesign()
    redesign()
    with pytest.raises(VariantRefused, match='current design'):
        discard_variant(queue, 'heat', 'v2')
    assert discard_variant(queue, 'heat', 'v1').variants == []
    with pytest.raises(VariantRefused, match='no variant v1'):
        discard_variant(queue, 'heat', 'v1')


def test_several_or_all_earlier_variants_are_discarded_at_once_and_the_current_one_is_kept(title):
    queue, redesign = title
    for _ in range(4):
        redesign()
    entry = read_entry(queue, 'heat')
    current, earlier = entry.variant_id, [v['variant_id'] for v in entry.variants]
    assert len(earlier) == 3

    with pytest.raises(VariantRefused, match='current design'):
        discard_variants(queue, 'heat', [earlier[0], current])
    with pytest.raises(VariantRefused, match='no variant nope'):
        discard_variants(queue, 'heat', [earlier[0], 'nope'])
    assert len(read_entry(queue, 'heat').variants) == 3   # a refusal discards nothing

    assert [v['variant_id'] for v in discard_variants(queue, 'heat', [earlier[1]]).variants] == [earlier[0], earlier[2]]
    kept = discard_variants(queue, 'heat')
    assert kept.variants == [] and kept.variant_id == current and kept.candidates == entry.candidates
    assert discard_variants(queue, 'heat').variants == []   # nothing left: nothing to do


def test_an_earlier_variant_can_be_viewed_without_changing_anything(title):
    queue, redesign = title
    redesign()
    redesign(ASKED)
    entry = read_entry(queue, 'heat')
    view = variant_view(entry, 'v1')
    assert _gain(view.candidates[0]) == 4.0 and view.adjustments == [] and read_entry(queue, 'heat').variant_id == 'v2'
    assert variant_view(entry, 'v2') is entry
    with pytest.raises(KeyError):
        variant_view(entry, 'v9')


def test_an_entry_designed_before_variants_were_kept_becomes_v1(tmp_path):
    entry = write_entry(str(tmp_path), 'old')
    assert entry.variant_id is None and next_variant_id(entry) == 'v2' and next_variant_id(None) == 'v1'


def test_a_redesign_starts_with_no_review_and_the_run_it_replaces_keeps_its_own(title):
    ''' contract 1.6 (§3, "Reviewed designs"): a new design was made at the new identity, so nothing about it was kept '''
    queue, redesign = title
    redesign()
    update_entry(queue, 'heat', reviewed_at=[{'design_version': 7, 'design_settings': None, 'at': 'x', 'by': None}])
    second = redesign()
    assert second.reviewed_at == [] and second.variants[0]['reviewed_at'][0]['design_version'] == 7


def test_the_entry_on_disk_keeps_its_history_while_the_projects_are_written(title, tmp_path, monkeypatch):
    ''' A redesign wrote the new design, then spent tens of seconds writing its projects before it added the earlier runs,
    the note and the fingerprints back: a reader in between (the review page) saw no earlier design runs. '''
    import pipeline.publish.project as project
    from pipeline.library.source import LibraryItem
    queue, redesign = title
    redesign()
    update_entry(queue, 'heat', reviewer_note='a person wrote this')
    seen = []

    def projects(*args, **kwargs):
        on_disk = read_entry(queue, 'heat')
        seen.append((on_disk.variant_id, [v['variant_id'] for v in on_disk.variants], on_disk.reviewer_note,
                     on_disk.design_fingerprint is not None))
        return {'mono': True, 'multichannel': None}

    monkeypatch.setattr(project, 'write_title_projects_if_safe', projects)
    item = LibraryItem(id='heat', source_path=str(tmp_path / 'heat.mkv'), display_name='Heat', fingerprint='s')
    project_dir = tmp_path / 'work' / 'heat'
    project_dir.mkdir(parents=True)
    result = design_if_needed(Session(), item, str(tmp_path / 'mono.wav'), DESIGNER, queue, AnalysisConfig(), force=True,
                              project_dir=str(project_dir))

    assert seen == [('v2', ['v1'], 'a person wrote this', True)]
    assert result.entry.variant_id == 'v2' and read_entry(queue, 'heat').variant_id == 'v2'
