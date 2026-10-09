"""Library run concurrency-limit defaults and validation."""
import pytest

from pipeline.library.run import LibraryRunConfig, stage_parallelism


def test_stage_parallelism_defaults_and_validates_each_independent_limit():
    assert stage_parallelism() == {'extract': 1, 'design': 1}
    assert stage_parallelism({'extract': 4, 'design': 2}) == {'extract': 4, 'design': 2}
    assert stage_parallelism({'extract': 16, 'design': 4}) == {'extract': 16, 'design': 4}   # extraction: one per disk


@pytest.mark.parametrize('value', [
    'extract:2',
    {'publish': 2},
    {'extract': 0},
    {'design': 5},
    {'extract': 17},
    {'extract': True},
    {'design': 1.5},
])
def test_stage_parallelism_rejects_invalid_profile_values(value):
    with pytest.raises(ValueError, match='run.parallelism'):
        stage_parallelism(value)


@pytest.mark.parametrize('field, value', [('extract_parallelism', 0), ('extract_parallelism', 17),
                                          ('extract_parallelism', True), ('design_parallelism', 0),
                                          ('design_parallelism', 5), ('design_parallelism', True)])
def test_library_run_config_rejects_invalid_worker_counts(field, value):
    with pytest.raises(ValueError, match=f'run.parallelism.{field.removesuffix("_parallelism")}'):
        LibraryRunConfig(work_dir='/work', queue_dir='/queue', designer='test', **{field: value})
