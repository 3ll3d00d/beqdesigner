'''Exercise the built image through real ffmpeg extraction, HTTP design and the review queue.'''
import argparse
import json
import math
import os
import pathlib
import struct
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Designer(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert request['mono_mix']['data_base64'] and request['fs'] > 0
        response = json.dumps({'contract_version': '1.0', 'decline_reason': 'no_rolloff_detected',
                               'decline_message': 'smoke fixture has no BEQ to apply'}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *args):
        pass


def fixture(root: pathlib.Path, designer_port: int) -> None:
    for name in ('config', 'work', 'queue', 'media'):
        (root / name).mkdir()
    with wave.open(str(root / 'media' / 'Smoke Movie.wav'), 'wb') as out:
        out.setnchannels(6)
        out.setsampwidth(2)
        out.setframerate(48000)
        samples = (int(9000 * math.sin(2 * math.pi * 40 * i / 48000)) for i in range(48000))
        out.writeframes(b''.join(struct.pack('<6h', *([sample] * 6)) for sample in samples))
    profile = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': ['/media']}],
               'designers': {'smoke': f'http://host.docker.internal:{designer_port}/design'},
               'run': {'work_dir': '/work', 'queue_dir': '/queue', 'designer': 'smoke'}}
    (root / 'config' / 'profile.yaml').write_text(json.dumps(profile))
    (root / 'config' / 'service.yaml').write_text('listen: {host: 0.0.0.0, port: 8080}\n')


def request(port: int, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {'Authorization': 'Bearer smoke-token'}
    if data is not None:
        headers['Content-Type'] = 'application/json'
    with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data, headers=headers),
                                timeout=5) as response:
        return json.load(response)


def wait_for(predicate, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise RuntimeError('timed out waiting for the service')


def smoke(image: str, port: int) -> None:
    server = ThreadingHTTPServer(('0.0.0.0', 0), Designer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    container = None
    try:
        with tempfile.TemporaryDirectory(prefix='beq-image-smoke-') as directory:
            root = pathlib.Path(directory)
            fixture(root, server.server_port)
            container = subprocess.check_output([
                'docker', 'run', '--detach', '--rm', '--add-host', 'host.docker.internal:host-gateway',
                '--publish', f'127.0.0.1:{port}:8080', '--user', f'{os.getuid()}:{os.getgid()}',
                '--env', 'BEQ_SERVICE_TOKEN=smoke-token',
                '--volume', f'{root / "config"}:/config:ro', '--volume', f'{root / "work"}:/work',
                '--volume', f'{root / "queue"}:/queue', '--volume', f'{root / "media"}:/media:ro', image,
            ], text=True).strip()
            assert wait_for(lambda: request(port, '/ready')['ready'])
            submitted = request(port, '/v1/jobs/run', {'filter': {'match': 'Smoke Movie'}, 'through': 'design'})
            job = wait_for(lambda: (state if (state := request(port, f'/v1/jobs/{submitted["id"]}'))['state']
                                    not in ('queued', 'running') else None))
            assert job['state'] == 'succeeded', job
            assert list((root / 'queue').glob('*.json')), 'design produced no review queue entry'
            print(f"image smoke passed: {submitted['id']}")
    finally:
        if container:
            subprocess.run(['docker', 'logs', container], check=False)
            subprocess.run(['docker', 'stop', container], check=False)
        server.shutdown()
        thread.join(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--port', type=int, default=18080)
    args = parser.parse_args()
    smoke(args.image, args.port)
