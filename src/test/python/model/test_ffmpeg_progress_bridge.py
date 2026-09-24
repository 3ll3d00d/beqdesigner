import socket

from model.ffmpeg import FfmpegProgressBridge


def test_stopping_progress_bridge_releases_its_udp_port():
    bridge = FfmpegProgressBridge(lambda key, value: None, port=0, auto=True)
    port = bridge._FfmpegProgressBridge__server.server_address[1]

    bridge.stop()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as replacement:
        replacement.bind(('127.0.0.1', port))
