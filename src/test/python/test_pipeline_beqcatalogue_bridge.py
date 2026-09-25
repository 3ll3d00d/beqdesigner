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
    author, relative_path = catalogue.RECORD_REPO_CONFIGS[0]
    assert (author, relative_path) == ('3ll3d00d', '.input/3ll3d00d/beqfilters')

    source = tmp_path / relative_path
    (source / 'movies').mkdir(parents=True)
    record = filter_record([PeakingEQ(48000, 40, 0.7071, -3)],
                           BeqMetadata(title='Bridge Fixture', year='2024', audio_types=['DTS-HD MA 5.1'],
                                       author=author), now=100)
    (source / 'movies' / 'bridge.json').write_text(json.dumps(record), encoding='utf-8')
    (source / 'database.json').write_bytes(aggregate({'movies/bridge.json': record}))

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'docs').mkdir()
    monkeypatch.setattr(catalogue, 'error_files', defaultdict(list), raising=False)
    monkeypatch.setattr(catalogue, 'source_record_times', {})
    monkeypatch.setattr(catalogue, 'json_catalogue', [], raising=False)
    monkeypatch.setattr(catalogue, 'times', {author: {}}, raising=False)
    with (tmp_path / 'database.csv').open('w', newline='') as handle:
        monkeypatch.setattr(catalogue, 'db_writer', csv.writer(handle), raising=False)
        extracted = catalogue.extract_filter_records(relative_path, author)
        assert len(extracted) == 1  # aggregate is skipped
        catalogue.process_content_from_repo(author, extracted, [], 'film', [])
    assert catalogue.error_files[author] == []
    for item in catalogue.json_catalogue:
        catalogue.finalise_catalogue_entry(item)
    public = tmp_path / 'docs' / 'database.json'
    public.write_text(json.dumps(catalogue.json_catalogue), encoding='utf-8')
    (published,) = json.loads(public.read_text(encoding='utf-8'))
    entry = CatalogueEntry('3ll3d00d-bridge', published)
    assert entry.title == record['title']
    assert entry.filters == record['filters']
    assert entry.digest == record['digest']
    assert entry.beqc_url.startswith('https://beqcatalogue.readthedocs.io/')
    assert published['audioCodecs'] == ['DTS-HD MA']
    assert published['audioChannelCounts'] == ['5.1']
    assert published['filterAuthor'] == 'human'
