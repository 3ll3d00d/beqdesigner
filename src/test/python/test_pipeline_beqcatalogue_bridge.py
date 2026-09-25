'''Exercise the installed sibling BEQCatalogue checkout's real JSON ingestion path.'''
import csv
import importlib
import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

from model.catalogue import CatalogueEntry
from model.iir import PeakingEQ
from pipeline.metadata import BeqMetadata
from pipeline.publish.catalogue_json import aggregate, filter_record


def test_filter_record_reaches_public_catalogue_and_designer(tmp_path, monkeypatch):
    checkout = Path(os.environ.get('BEQCATALOGUE_CHECKOUT', Path(__file__).resolve().parents[3].parent / 'beqcatalogue'))
    if not (checkout / 'beqcatalogue' / '__init__.py').is_file():
        pytest.skip('set BEQCATALOGUE_CHECKOUT to run the cross-repository contract test')
    monkeypatch.syspath_prepend(str(checkout / 'beqcatalogue'))
    monkeypatch.syspath_prepend(str(checkout))
    catalogue = importlib.import_module('beqcatalogue')

    source = tmp_path / 'source'
    (source / 'filters').mkdir(parents=True)
    record = filter_record([PeakingEQ(48000, 40, 0.7071, -3)],
                           BeqMetadata(title='Bridge Fixture', year='2024', audio_types=['DTS-HD MA 5.1'],
                                       author='fixture'), now=100)
    (source / 'filters' / 'bridge.json').write_text(json.dumps(record), encoding='utf-8')
    (source / 'database.json').write_bytes(aggregate({'filters/bridge.json': record}))

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'docs').mkdir()
    catalogue.error_files = defaultdict(list)
    catalogue.source_record_times = {}
    catalogue.json_catalogue = []
    catalogue.times = {'fixture': {}}
    with (tmp_path / 'database.csv').open('w', newline='') as handle:
        catalogue.db_writer = csv.writer(handle)
        extracted = catalogue.extract_filter_records(str(source), 'fixture')
        assert len(extracted) == 1  # aggregate is skipped
        catalogue.process_content_from_repo('fixture', extracted, [], 'film', [])
    assert catalogue.error_files['fixture'] == []
    for item in catalogue.json_catalogue:
        catalogue.finalise_catalogue_entry(item)
    public = tmp_path / 'docs' / 'database.json'
    public.write_text(json.dumps(catalogue.json_catalogue), encoding='utf-8')
    (published,) = json.loads(public.read_text(encoding='utf-8'))
    entry = CatalogueEntry('fixture-bridge', published)
    assert entry.title == record['title']
    assert entry.filters == record['filters']
    assert entry.digest == record['digest']
    assert entry.beqc_url.startswith('https://beqcatalogue.readthedocs.io/')
    assert published['audioCodecs'] == ['DTS-HD MA']
    assert published['audioChannelCounts'] == ['5.1']
    assert published['filterAuthor'] == 'human'
