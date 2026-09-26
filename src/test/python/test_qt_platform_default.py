import os


def test_the_suite_runs_offscreen_unless_told_otherwise():
    assert os.environ.get('QT_QPA_PLATFORM')
