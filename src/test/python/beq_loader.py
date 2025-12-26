from dataclasses import asdict, dataclass, is_dataclass
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import requests
from scipy.signal import unit_impulse

from model.catalogue import CatalogueEntry, load_catalogue
from model.iir import CompleteFilter
from model.signal import Signal


@dataclass
class BEQFilter:
    mag_freqs: np.ndarray
    mag_db: np.ndarray
    entry: CatalogueEntry


def convert(entry: CatalogueEntry, fs=1000) -> BEQFilter | None:
    u_i = unit_impulse(fs * 4, 'mid') * 23453.66
    f = CompleteFilter(fs=fs, filters=entry.iir_filters(fs=fs), description=f'{entry}')
    signal = Signal('test', u_i, fs=fs)
    try:
        f_signal = signal.sosfilter(f.get_sos(), filtfilt=False)
        x, y = f_signal.mag_response
        return BEQFilter(x, y, entry)
    except Exception as e:
        print(f'Unable to process entry {entry.title}')
        return None


class CatalogueEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, CatalogueEntry):
            return obj.for_search
        return super().default(obj)


class CatalogueDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(dct):
        if 'mag_freqs' in dct and 'mag_db' in dct and 'entry' in dct:
            return BEQFilter(np.array(dct['mag_freqs']), np.array(dct['mag_db']), CatalogueEntry('0', dct['entry']))
        return dct


def load() -> list[BEQFilter]:
    a = time.time()

    try:
        with open('database.bin', 'r') as f:
            data: list[dict] = json.load(f, cls=CatalogueDecoder)['data']
    except Exception as e:
        try:
            r = requests.get('https://raw.githubusercontent.com/3ll3d00d/beqcatalogue/master/docs/database.json',
                             allow_redirects=True)
            r.raise_for_status()
            entries: list[CatalogueEntry] = load_catalogue(r.content)
        except requests.exceptions.HTTPError as e:
            entries: list[CatalogueEntry] = load_catalogue('/home/matt/.beq/database.json')
        with ProcessPoolExecutor() as executor:
            data: list[BEQFilter] = list(executor.map(convert, [e for e in entries if e.filters]))
        with open('database.bin', 'w') as f:
            json.dump({'data': data}, f, cls=CatalogueEncoder)

    b = time.time()
    print(f'Loaded catalogue in {b - a:.3g}s')

    return data
