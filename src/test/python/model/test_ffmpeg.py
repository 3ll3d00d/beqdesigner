from model.ffmpeg import find_missing_ffmpeg_tools, describe_missing_binary


def test_find_missing_ffmpeg_tools_none_missing(monkeypatch):
    monkeypatch.setattr('shutil.which', lambda name: f"/usr/bin/{name}")
    assert find_missing_ffmpeg_tools() == []


def test_find_missing_ffmpeg_tools_all_missing(monkeypatch):
    monkeypatch.setattr('shutil.which', lambda name: None)
    assert find_missing_ffmpeg_tools() == ['ffmpeg', 'ffprobe']


def test_find_missing_ffmpeg_tools_partial(monkeypatch):
    monkeypatch.setattr('shutil.which', lambda name: '/usr/bin/ffmpeg' if name.startswith('ffmpeg') else None)
    assert find_missing_ffmpeg_tools() == ['ffprobe']


def test_describe_missing_binary_includes_filename():
    e = FileNotFoundError(2, 'No such file or directory')
    e.filename = 'ffprobe'
    msg = describe_missing_binary(e)
    assert 'ffprobe' in msg
    assert 'Preferences' in msg


def test_describe_missing_binary_without_filename():
    e = FileNotFoundError(2, 'No such file or directory')
    e.filename = None
    msg = describe_missing_binary(e)
    assert 'ffmpeg/ffprobe' in msg
