import json

from model.iir import LowShelf, PeakingEQ
from pipeline.metadata import BeqMetadata
from pipeline.publish.catalogue_json import aggregate, filter_record


def test_filter_record_matches_the_catalogue_biquad_shape_and_unrolls_shelves():
    record = filter_record([PeakingEQ(48000, 40, 0.7071, -3), LowShelf(48000, 20, 0.9, 4, 2)],
                           BeqMetadata(title='Example', year='2020', audio_types=['DTS-HD MA 5.1'], author='me'),
                           now=10)

    assert record['content_type'] == 'film'
    assert [f['type'] for f in record['filters']] == ['PeakingEQ', 'LowShelf', 'LowShelf']
    assert record['filters'][0]['biquads']['96000'] == {
        'b': ['0.9993588401736285', '-1.9956024707176718', '0.9962504693854236'],
        'a': ['1.9956024707176718', '-0.9956093095590519']}
    assert record['filters'][1]['count'] == 1
    assert record['created_at'] == record['updated_at'] == 10


def test_filter_record_preserves_creation_time_and_aggregate_is_path_sorted():
    meta = BeqMetadata(title='Example', year='2020', audio_types=['DTS'], author='me')
    first = filter_record([PeakingEQ(48000, 40, 1, -3)], meta, now=10)
    changed = filter_record([PeakingEQ(48000, 40, 1, -4)], meta, existing=first, now=20)

    assert (changed['created_at'], changed['updated_at']) == (10, 20)
    assert [record['digest'] for record in json.loads(aggregate({'z.json': changed, 'a.json': first}))] == \
           [first['digest'], changed['digest']]
