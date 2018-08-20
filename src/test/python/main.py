import os

import ffmpeg
from ffmpeg.nodes import FilterNode, filter_operator

os.environ['PATH'] = os.environ['PATH'] + ';C:\\Users\\mattk\\apps\\ffmpeg-20180816-fe06ed2-win64-static\\bin'

file = 'd:/Despicable Me 3.mkv'

# merge to mono
i1 = ffmpeg.input(file)['1']
f1 = i1.filter('pan', **{'mono|c0': '0.5*c0+0.5*c1'})
s1 = f1.filter('aresample', '1000', resampler='soxr')\
    .output('d:/junk/test.wav', acodec='pcm_s24le')
print(s1.compile())

@filter_operator()
def join(*streams, **kwargs):
    return FilterNode(streams, join.__name__, kwargs=kwargs, max_inputs=None).stream()

i1 = ffmpeg.input(file)

s1 = i1['1:0'].join(i1['1:1'], inputs=2, channel_layout='stereo', map='0.0-FL|1.0-FR').output('d:/junk/test_join.wav', acodec='pcm_s24le')
print(s1.compile())
