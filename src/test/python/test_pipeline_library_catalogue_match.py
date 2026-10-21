'''
pipeline/library/catalogue_match.py's filter building: a catalogue entry's filters as biquads, from database.json's shape
and from beqcatalogue's per-title compare/ files, whose `count` repeats a filter (design/web-parity.md WP0/WP3).
'''
import numpy as np
import pytest

from model.catalogue import CatalogueEntry
from model.iir import LowShelf, PeakingEQ
from pipeline.library.catalogue_match import catalogue_biquads, catalogue_filter

_FILTERS = [{'type': 'LowShelf', 'freq': 22.0, 'gain': 5.0, 'q': 0.9, 'count': 3},
            {'type': 'PeakingEQ', 'freq': 13.0, 'gain': 0.8, 'q': 3.0}]


def test_a_count_repeats_its_filter_in_order():
    biquads = catalogue_biquads(_FILTERS, fs=48000)
    assert [type(b) for b in biquads] == [LowShelf, LowShelf, LowShelf, PeakingEQ]
    assert [b.id for b in biquads] == [0, 1, 2, 3] and all(b.fs == 48000 for b in biquads)
    assert (biquads[0].freq, biquads[0].q, biquads[0].gain) == (22.0, 0.9, 5.0)


def test_an_unknown_type_is_refused():
    with pytest.raises(ValueError, match='Unknown filt_type AllPass'):
        catalogue_biquads([{'type': 'AllPass', 'freq': 1, 'gain': 0, 'q': 1}])


def test_the_browse_catalogue_entry_builds_the_same_biquads():
    entry = CatalogueEntry('0', {'title': 'x', 'filters': [dict(f, count=1) for f in _FILTERS]})
    assert [(type(b), b.freq, b.q, b.gain) for b in entry.iir_filters()] == \
           [(type(b), b.freq, b.q, b.gain) for b in catalogue_biquads([dict(f, count=1) for f in _FILTERS], 96000)]


def test_the_complete_filter_boosts_where_the_shelves_do():
    response = catalogue_filter(_FILTERS).get_transfer_function().get_magnitude()
    low = response.y[np.searchsorted(response.x, 5.0)]
    assert low == pytest.approx(15.0, abs=1.5) and abs(response.y[np.searchsorted(response.x, 150.0)]) < 0.5
