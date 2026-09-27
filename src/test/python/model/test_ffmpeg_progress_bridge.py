import socket

from model.ffmpeg import FfmpegProgressBridge


def test_stopping_progress_bridge_releases_its_udp_port():
    bridge = FfmpegProgressBridge(lambda key, value: None, port=0, auto=True)
    port = bridge._FfmpegProgressBridge__server.server_address[1]

    bridge.stop()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as replacement:
        replacement.bind(('127.0.0.1', port))


def test_stopping_hands_over_a_report_that_arrived_before_the_stop():
    '''
    ffmpeg sends its last report (the final out_time, progress=end) as it exits, so run_sync can stop the bridge while that
    report is still waiting on the socket. serve_forever stops without reading it; the bridge must read it before closing.
    Here the first report's handler keeps the loop busy while the second arrives and stop() is asked for.
    '''
    import threading
    import time
    seen = []
    first_handled = threading.Event()

    def handler(key, value):
        seen.append((key, value))
        if key == 'first':
            first_handled.set()
            time.sleep(0.5)

    bridge = FfmpegProgressBridge(handler, port=0, auto=True)
    port = bridge._FfmpegProgressBridge__server.server_address[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as ffmpeg:
        ffmpeg.sendto(b'first=1', ('127.0.0.1', port))
        assert first_handled.wait(10)
        ffmpeg.sendto(b'out_time_ms=984000\nprogress=end', ('127.0.0.1', port))

    bridge.stop()

    assert seen == [('first', '1'), ('out_time_ms', '984000'), ('progress', 'end')]


_HOLD_A_PORT = '''
import socket, sys
from model.ffmpeg import get_next_port
port = get_next_port()
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as bridge:
    bridge.bind(('127.0.0.1', port))
    print(port, flush=True)
    sys.stdin.readline()
'''


def test_two_processes_are_never_given_the_same_progress_port():
    '''
    Each process used to count up from 12000, so two at once (parallel test workers, the app beside the pipeline service) both
    bound 12001 and one failed with "Address already in use".
    '''
    import os
    import pathlib
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=str((pathlib.Path(__file__).parents[3] / 'main' / 'python').resolve()))
    first = subprocess.Popen([sys.executable, '-c', _HOLD_A_PORT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
    try:
        held = int(first.stdout.readline())
        second = subprocess.run([sys.executable, '-c', _HOLD_A_PORT], input='\n', capture_output=True, text=True,
                                timeout=60, env=env)
        assert second.returncode == 0, second.stderr
        assert int(second.stdout) != held
    finally:
        first.communicate('\n', timeout=60)
