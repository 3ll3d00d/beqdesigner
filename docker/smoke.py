'''
Exercise the built image through real ffmpeg extraction, HTTP design and the review queue.

    python3 docker/smoke.py beqdesigner-pipeline:smoke [--designer-image beqforge-designer:smoke]

By default a stub designer on the host answers every request with a decline. With --designer-image the real designer
runs in a second container on a private network, with the same work directory mounted at its shared root, as
docker/compose.example.yaml runs them; the title's audio must then reach it by reference, not inline.
'''
import argparse
import json
import math
import os
import pathlib
import re
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


def fixture(root: pathlib.Path, designer_port: int = 0, *, designer_url: str = '', seconds: int = 1) -> None:
    '''
    :param designer_url: a designer that shares the work directory (`by_reference`); else the stub on `designer_port`.
    :param seconds: the title's length: long enough, by reference, that its audio inline would be megabytes.
    '''
    for name in ('config', 'work', 'queue', 'media'):
        (root / name).mkdir()
    with wave.open(str(root / 'media' / 'Smoke Movie.wav'), 'wb') as out:
        out.setnchannels(6)
        out.setsampwidth(2)
        out.setframerate(48000)
        samples = (int(9000 * math.sin(2 * math.pi * 40 * i / 48000)) for i in range(48000 * seconds))
        out.writeframes(b''.join(struct.pack('<6h', *([sample] * 6)) for sample in samples))
    designer = {'url': designer_url, 'by_reference': True} if designer_url else \
        f'http://host.docker.internal:{designer_port}/design'
    profile = {'sources': [{'name': 'disk', 'kind': 'filesystem', 'globs': ['/media']}],
               'designers': {'smoke': designer},
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


def page(port: int, path: str) -> str:
    ''' A page or file served without the token (the browser app's). '''
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=5) as response:
        return response.read().decode()


def check_browser_app(port: int) -> None:
    ''' The image serves the built browser app at /ui (design/web-app.md §5): its page, and the script the page loads. '''
    html = page(port, '/ui/')
    assert '<div id="root">' in html, f'/ui/ is not the browser app:\n{html[:300]}'
    script = re.search(r'src="(/ui/assets/[^"]+\.js)"', html)
    assert script, f'the page loads no script from /ui/assets:\n{html[:300]}'
    assert page(port, script.group(1)), f'{script.group(1)} is empty'


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


def by_reference(designer_log: str) -> bool:
    ''' Whether every request the designer logged arrived without its audio: by reference, not megabytes inline. '''
    bodies = re.findall(r'request timing: body ([0-9.]+) MB', designer_log)
    return bool(bodies) and all(float(size) < 0.1 for size in bodies)


def smoke(image: str, port: int, designer_image: str = '') -> None:
    server = thread = None
    if not designer_image:
        server = ThreadingHTTPServer(('0.0.0.0', 0), Designer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    user = f'{os.getuid()}:{os.getgid()}'
    network = f'beq-smoke-{port}'
    designer = f'beq-smoke-designer-{port}'
    container = None
    started_designer = False
    try:
        with tempfile.TemporaryDirectory(prefix='beq-image-smoke-') as directory:
            root = pathlib.Path(directory)
            if designer_image:
                fixture(root, designer_url=f'http://{designer}:8420/design', seconds=30)
                subprocess.run(['docker', 'network', 'create', network], check=True, capture_output=True)
                (root / 'cache').mkdir()
                subprocess.run(['docker', 'run', '--detach', '--rm', '--name', designer, '--network', network,
                                '--user', user, '--volume', f'{root / "work"}:/work',
                                '--volume', f'{root / "cache"}:/cache', designer_image], check=True)
                started_designer = True
            else:
                fixture(root, server.server_port)
            container = subprocess.check_output([
                'docker', 'run', '--detach', '--rm', '--add-host', 'host.docker.internal:host-gateway',
                *(['--network', network] if designer_image else []),
                '--publish', f'127.0.0.1:{port}:8080', '--user', user,
                '--env', 'BEQ_SERVICE_TOKEN=smoke-token',
                '--volume', f'{root / "config"}:/config:ro', '--volume', f'{root / "work"}:/work',
                '--volume', f'{root / "queue"}:/queue', '--volume', f'{root / "media"}:/media:ro', image,
            ], text=True).strip()
            assert wait_for(lambda: request(port, '/ready')['ready'])
            check_browser_app(port)
            if designer_image:   # the service says when it can reach the designer (/v1/status, TODO R2)
                assert wait_for(lambda: request(port, '/v1/status')['designer']['reachable'])
            submitted = request(port, '/v1/jobs/run', {'filter': {'match': 'Smoke Movie'}, 'through': 'design'})
            job = wait_for(lambda: (state if (state := request(port, f'/v1/jobs/{submitted["id"]}'))['state']
                                    not in ('queued', 'running') else None), seconds=300)
            assert job['state'] == 'succeeded', job
            entries = list((root / 'queue').glob('*.json'))
            assert entries, 'design produced no review queue entry'
            review = request(port, f'/v1/titles/{entries[0].stem}/review')   # what a person decides it on, over HTTP
            assert review['status'] == 'pending' and review['candidates'], review
            if designer_image:
                logged = subprocess.run(['docker', 'logs', designer], capture_output=True, text=True)
                log = logged.stdout + logged.stderr
                assert by_reference(log), f'the audio did not reach the designer by reference:\n{log}'
                assert 'beqforge' in entries[0].read_text(), 'the queue entry does not name the designer build'
            print(f"image smoke passed: {submitted['id']}" + (f' (designed by {designer_image}, by reference)'
                                                              if designer_image else ''))
    finally:
        if container:
            subprocess.run(['docker', 'logs', container], check=False)
            subprocess.run(['docker', 'stop', container], check=False)
        if started_designer:
            subprocess.run(['docker', 'logs', designer], check=False)
            subprocess.run(['docker', 'stop', designer], check=False)
        if designer_image:
            subprocess.run(['docker', 'network', 'rm', network], check=False, capture_output=True)
        if server is not None:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('image')
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--designer-image', default='', help='run this beqforge designer image beside the service')
    args = parser.parse_args()
    smoke(args.image, args.port, args.designer_image)
