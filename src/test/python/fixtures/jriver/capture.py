'''
Capture a real JRiver MCWS server's responses and write the sanitised fixture beside this file (TODO E1).

    PYTHONPATH=src/main/python uv run python src/test/python/fixtures/jriver/capture.py PROFILE RAW_DIR [--sanitise-only]

PROFILE is a library profile with a `jriver` source (its login is read from the file and never printed); RAW_DIR, outside
the repository, receives the raw responses: /Alive, Library/Fields, Browse/Children three levels down from the root and
Browse/Files of the source's browse node with exactly the fields JRiverLibrarySource requests. The sanitised fixture keeps:

* /Alive with the friendly name, access key and runtime GUID replaced (the field set and order are real);
* Library/Fields cut to the fields the source requests or reads;
* Browse/Children of the root and of each node on the way to the browse node, nothing else;
* Browse/Files rows chosen to cover every path form, Playback Info shape, image form and id combination seen, with
  `Description` dropped and any display-device id inside Playback Info replaced.

Titles, file and folder names under the server's media root are public catalogue data and kept as reported.
'''
import argparse
import json
import pathlib
import re
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
ROWS_PER_SHAPE = 2
ALIVE_REPLACEMENTS = {'FriendlyName': 'MEDIA-SERVER', 'AccessKey': 'ACCESSKEY',
                      'RuntimeGUID': '{00000000-0000-0000-0000-000000000000}'}


def capture(profile_path: str, raw: pathlib.Path) -> None:
    import requests
    from requests.auth import HTTPBasicAuth
    from pipeline.library.jriver import JRiverLibrarySource
    from pipeline.library.profile import read_config_file

    config = read_config_file(profile_path)
    source = next(s for s in config['sources'] if s.get('kind') == 'jriver')
    base = f"http{'s' if source.get('ssl') else ''}://{source['host']}:{source['port']}/MCWS/v1/"
    auth = HTTPBasicAuth(source['username'], source['password']) if source.get('username') else None
    raw.mkdir(parents=True, exist_ok=True)

    def get(path, params, name):
        response = requests.get(base + path, params=params, auth=auth, timeout=60)
        response.raise_for_status()
        (raw / name).write_bytes(response.content)
        return response

    get('Alive', None, 'alive.xml')
    get('Library/Fields', None, 'fields.xml')
    frontier = [(-1, 0)]
    while frontier:
        node, depth = frontier.pop(0)
        root = ET.fromstring(get('Browse/Children', {'Version': 2, 'ErrorOnMissing': 0, 'ID': node},
                                 f'children_{node}.xml').text)
        if depth < 2:
            frontier += [(int(item.text), depth + 1) for item in root.iter('Item') if (item.text or '').strip().isdigit()]
    library = JRiverLibrarySource(source['host'], source['port'], source['browse_node_id'],
                                  external_id_fields=source.get('external_id_fields'))
    fields = ['Key', 'Name', 'Media Type', 'Media Sub Type', 'Series', 'Season', 'Episode', 'Artist', 'Album',
              'Track #', 'Dimensions', 'HDR Format', 'Duration'] + library.requested_fields
    get('Browse/Files', {'ID': source['browse_node_id'], 'Action': 'JSON', 'Fields': ','.join(fields)}, 'files.json')
    (raw / 'request.json').write_text(json.dumps({'browse_node_id': source['browse_node_id'], 'fields': fields,
                                                  'browse_path': source.get('browse_path', '')}, indent=1))


def playback_parts(value: str):
    ''' The length-prefixed `(n:text)` records of a Playback Info value, or None if it does not parse. '''
    parts, i = [], 0
    while i < len(value):
        match = re.match(r'\((\d+):', value[i:])
        if not match:
            return None
        start = i + match.end()
        parts.append(value[start:start + int(match.group(1))])
        i = start + int(match.group(1)) + 1
    return parts


def _record(text: str) -> str:
    return f'({len(text)}:{text})'


def _scrub_playback(value: str) -> str:
    ''' A JRVRProfiles record names the display device (its hardware id): keep its nesting, not the id. '''
    parts = playback_parts(value)
    if parts is None or not any('DISPLAY#' in part for part in parts):
        return value
    device = _record('(1:0)' + _record(_record('DISPLAY#DEVICE') + _record('-1')))
    return ''.join(_record(device[device.index(':') + 1:-1] if 'DISPLAY#' in part else part) for part in parts)


def _shape(row: dict) -> tuple:
    filename = row.get('Filename', '').lower()
    form = 'bluray;' if ';' in filename else 'bdmv' if filename.endswith('index.bdmv') else filename.rsplit('.', 1)[-1]
    playback = playback_parts(row.get('Playback Info', '')) if row.get('Playback Info') else []
    image = 'none' if not row.get('Image File') else 'INTERNAL' if row['Image File'] == 'INTERNAL' else 'file'
    ids = tuple(sorted(k for k in ('IMDb ID', 'TheMovieDB Movie ID') if row.get(k)))
    return (form, row.get('Filename', '')[:1].islower(), tuple(playback[1::2]) if playback else (),
            image, ids, bool(row.get('Series')), int(row.get('Audio Streams') or 0) > 1)


def sanitise(raw: pathlib.Path, out: pathlib.Path) -> dict:
    request = json.loads((raw / 'request.json').read_text())
    alive = (raw / 'alive.xml').read_text(encoding='utf-8')
    for name, value in ALIVE_REPLACEMENTS.items():
        alive = re.sub(rf'(<Item Name="{name}">)[^<]*(</Item>)', rf'\g<1>{value}\g<2>', alive)
    (out / 'alive.xml').write_text(alive, encoding='utf-8')

    wanted = set(request['fields']) | {'Date (year)', 'Filename', 'Image File'}
    fields = ET.parse(raw / 'fields.xml').getroot()
    container = fields.find('Fields')
    for field in list(container):
        if field.attrib.get('Name') not in wanted:
            container.remove(field)
    ET.ElementTree(fields).write(out / 'fields.xml', encoding='UTF-8', xml_declaration=True)

    # the root, and each node on the way to the browse node: no other branch of the library
    path_names = [n.strip() for n in request.get('browse_path', '').split('>') if n.strip()]
    node, kept = -1, {}
    for name in [None, *path_names]:
        if name is not None:
            match = next((i for i in ET.parse(raw / f'children_{node}.xml').getroot().iter('Item')
                          if i.attrib.get('Name') == name), None)
            if match is None:
                break
            node = int(match.text)
        if (raw / f'children_{node}.xml').is_file():
            kept[node] = (raw / f'children_{node}.xml').read_text(encoding='utf-8')
    for node_id, text in kept.items():
        (out / f'children_{node_id}.xml').write_text(text, encoding='utf-8')

    rows = json.loads((raw / 'files.json').read_text(encoding='utf-8'))
    by_shape: dict = {}
    for row in sorted(rows, key=lambda r: r['Key']):
        by_shape.setdefault(_shape(row), [])
        if len(by_shape[_shape(row)]) < ROWS_PER_SHAPE:
            by_shape[_shape(row)].append(row)
    chosen = sorted((row for group in by_shape.values() for row in group), key=lambda r: r['Key'])
    sanitised = []
    for row in chosen:
        row = {k: v for k, v in row.items() if k != 'Description'}
        if row.get('Playback Info'):
            row['Playback Info'] = _scrub_playback(row['Playback Info'])
        sanitised.append(row)
    (out / 'files.json').write_text(json.dumps(sanitised, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    summary = {'request': request, 'nodes_kept': sorted(kept), 'rows_captured': len(rows), 'rows_kept': len(sanitised),
               'shapes': len(by_shape)}
    (out / 'capture.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('profile')
    parser.add_argument('raw', type=pathlib.Path)
    parser.add_argument('--sanitise-only', action='store_true', help='rewrite the fixture from an earlier capture')
    args = parser.parse_args()
    if not args.sanitise_only:
        capture(args.profile, args.raw)
    print(json.dumps(sanitise(args.raw, HERE), indent=1))
