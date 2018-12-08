import logging
import os
from uuid import uuid4

import numpy as np

from model.iir import Gain

logger = logging.getLogger('codec')


def signaldata_to_json(signal):
    '''
    Converts the signal to a json compatible format.
    :return: a dict to write to json.
    '''
    avg = signal.raw[0]
    peak = signal.raw[1]
    out = {
        '_type': signal.__class__.__name__,
        'name': signal.name,
        'fs': signal.fs,
        'data': {
            'avg': xydata_to_json(avg),
            'peak': xydata_to_json(peak),
        },
    }
    if signal.filter is not None:
        out['filter'] = signal.filter.to_json()
    if signal.master is not None:
        out['master_name'] = signal.master.name
    if len(signal.slaves) > 0:
        out['slave_names'] = [s.name for s in signal.slaves]
    if signal.duration_seconds is not None:
        out['duration_seconds'] = signal.duration_seconds
        out['start_seconds'] = signal.start_seconds
    if signal.signal is not None and signal.signal.metadata is not None:
        out['metadata'] = signal.signal.metadata
    return out


def signalmodel_from_json(input):
    '''
    Reassembles all signals from the json including master/slave relationships.
    :param input: the input, a list of json dicts.
    :return: the signals
    '''
    signals = [signaldata_from_json(x) for x in input]
    for x in input:
        if 'slave_names' in x:
            master_name = x['name']
            logger.debug(f"Reassembling slaves for {master_name}")
            master = next((s for s in signals if s.name == master_name), None)
            if master is not None:
                for slave_name in x['slave_names']:
                    slave = next((s for s in signals if s.name == slave_name), None)
                    if slave is not None:
                        master.enslave(slave)
                    else:
                        logger.error(f"Bad json encountered, slave not decoded ({master_name} -> {slave_name})")
            else:
                logger.error(f"Bad json encountered, master {master_name} not decoded")
    return signals


def signaldata_from_json(o):
    '''
    Converts the given dict to a SignalData if it is compatible.
    :param o: the dict (from json).
    :return: the SignalData (or an error)
    '''
    from model.signal import SingleChannelSignalData
    if '_type' not in o:
        raise ValueError(f"{o} is not SignalData")
    elif o['_type'] == SingleChannelSignalData.__name__ or o['_type'] == 'SignalData':
        filt = o.get('filter', None)
        if filt is not None:
            filt = filter_from_json(filt)
        data = o['data']
        avg = xydata_from_json(data['avg'])
        peak = xydata_from_json(data['peak'])
        metadata = o.get('metadata', None)
        signal = None
        if metadata is not None:
            try:
                if os.path.isfile(metadata['src']):
                    from model.signal import readWav
                    signal = readWav(o['name'], metadata['src'], channel=metadata['channel'],
                                     start=metadata['start'], end=metadata['end'],
                                     target_fs=o['fs'])
            except:
                logger.exception(f"Unable to load signal from {metadata['src']}")
        if 'duration_seconds' in o:
            signal_data = SingleChannelSignalData(o['name'], o['fs'], [avg, peak], filter=filt,
                                                  duration_seconds=o.get('duration_seconds', None),
                                                  start_seconds=o.get('start_seconds', None),
                                                  signal=signal)
        elif 'duration_hhmmss' in o:
            h, m, s = o['duration_hhmmss'].split(':')
            duration_seconds = (int(h) * 3600) + int(m) * (60 + float(s))
            h, m, s = o['start_hhmmss'].split(':')
            start_seconds = (int(h) * 3600) + int(m) * (60 + float(s))
            signal_data = SingleChannelSignalData(o['name'], o['fs'], [avg, peak], filter=filt,
                                                  duration_seconds=duration_seconds,
                                                  start_seconds=start_seconds,
                                                  signal=signal)
        else:
            signal_data = SingleChannelSignalData(o['name'], o['fs'], [avg, peak], filter=filt,
                                                  signal=signal)
        return signal_data
    raise ValueError(f"{o._type} is an unknown signal type")


def filter_from_json(o):
    '''
    Converts a dict (parsed from json) to a filter.
    :param o: the dict.
    :return: the filter.
    '''
    from model.iir import Passthrough, PeakingEQ, LowShelf, HighShelf, FirstOrder_LowPass, \
        FirstOrder_HighPass, SecondOrder_LowPass, SecondOrder_HighPass, AllPass, CompleteFilter, ComplexLowPass, \
        FilterType, ComplexHighPass

    filt = None
    if '_type' not in o:
        raise ValueError(f"{o} is not a filter")
    if o['_type'] == Passthrough.__name__:
        if 'fs' in o:
            filt = Passthrough(fs=int(o['fs']))
        else:
            filt = Passthrough()
    elif o['_type'] == Gain.__name__:
        filt = Gain(o['fs'], o['gain'])
    elif o['_type'] == PeakingEQ.__name__:
        filt = PeakingEQ(o['fs'], o['fc'], o['q'], o['gain'])
    elif o['_type'] == LowShelf.__name__:
        filt = LowShelf(o['fs'], o['fc'], o['q'], o['gain'], o['count'])
    elif o['_type'] == HighShelf.__name__:
        filt = HighShelf(o['fs'], o['fc'], o['q'], o['gain'], o['count'])
    elif o['_type'] == FirstOrder_LowPass.__name__:
        filt = FirstOrder_LowPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == FirstOrder_HighPass.__name__:
        filt = FirstOrder_HighPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == SecondOrder_LowPass.__name__:
        filt = SecondOrder_LowPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == SecondOrder_HighPass.__name__:
        filt = SecondOrder_HighPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == AllPass.__name__:
        filt = AllPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == CompleteFilter.__name__:
        filt = CompleteFilter(filters=[filter_from_json(x) for x in o['filters']], description=o['description'])
    elif o['_type'] == ComplexLowPass.__name__:
        filt = ComplexLowPass(FilterType(o['filter_type']), o['order'], o['fs'], o['fc'])
    elif o['_type'] == ComplexHighPass.__name__:
        filt = ComplexHighPass(FilterType(o['filter_type']), o['order'], o['fs'], o['fc'])
    if filt is None:
        raise ValueError(f"{o._type} is an unknown filter type")
    else:
        if filt.id == -1:
            filt.id = uuid4()
        return filt


def xydata_from_json(o):
    '''
    Converts a json dict to an XYData.
    :param o: the dict.
    :return: the XYData (or an error).
    '''
    from model.iir import XYData
    if '_type' not in o:
        raise ValueError(f"{o} is not XYData")
    elif o['_type'] == XYData.__name__:
        x_json = o['x']
        x_vals = np.linspace(x_json['min'], x_json['max'], num=x_json['count'], dtype=np.float64)
        description = o['description'] if 'description' in o else ''
        return XYData(o['name'], description, x_vals, np.array(o['y']), colour=o.get('colour', None),
                      linestyle=o.get('linestyle', '-'))
    raise ValueError(f"{o._type} is an unknown data type")


def xydata_to_json(data):
    '''
    A json compatible rendering of the xy data.
    :return: a dict.
    '''
    return {
        '_type': data.__class__.__name__,
        'name': data.internal_name,
        'description': data.internal_description,
        'x': {
            'count': data.x.size,
            'min': data.x[0],
            'max': data.x[-1]
        },
        'y': np.around(data.y, decimals=6).tolist(),
        'colour': data.colour,
        'linestyle': data.linestyle
    }


def minidspxml_to_filt(file):
    ''' Extracts a set of filters from the provided minidsp file '''
    from model.iir import PeakingEQ, LowShelf, HighShelf

    filts = __extract_filters(file)
    output = []
    for filt_tup, count in filts.items():
        filt_dict = dict(filt_tup)
        if filt_dict['type'] == 'SL':
            filt = LowShelf(48000, float(filt_dict['freq']), float(filt_dict['q']), float(filt_dict['boost']),
                            count=count)
            output.append(filt)
        elif filt_dict['type'] == 'SH':
            filt = HighShelf(48000, float(filt_dict['freq']), float(filt_dict['q']), float(filt_dict['boost']),
                             count=count)
            output.append(filt)
        elif filt_dict['type'] == 'PK':
            for i in range(0, count):
                filt = PeakingEQ(48000, float(filt_dict['freq']), float(filt_dict['q']), float(filt_dict['boost']))
                output.append(filt)
        else:
            logger.info(f"Ignoring unknown filter type {filt_dict}")
    return output


def __extract_filters(file):
    import xml.etree.ElementTree as ET
    from collections import Counter

    ignore_vals = ['hex', 'dec']
    tree = ET.parse(file)
    root = tree.getroot()
    filts = {}
    for child in root:
        if child.tag == 'filter':
            if 'name' in child.attrib:
                inner_filt = None
                filter_tokens = child.attrib['name'].split('_')
                if len(filter_tokens) == 3:
                    if filter_tokens[0] == 'PEQ':
                        if filter_tokens[1] not in filts:
                            filts[filter_tokens[1]] = {}
                        filt = filts[filter_tokens[1]]
                        if filter_tokens[2] not in filt:
                            filt[filter_tokens[2]] = {}
                        inner_filt = filt[filter_tokens[2]]
                        for val in child:
                            if val.tag not in ignore_vals:
                                inner_filt[val.tag] = val.text
                if inner_filt is not None and 'bypass' in inner_filt and inner_filt['bypass'] == '1':
                    del filts[filter_tokens[1]]
    final_filt = None
    # if 1 and 2 are identical then throw one away
    if '1' in filts and '2' in filts:
        filt_1 = filts['1']
        filt_2 = filts['2']
        if filt_1 == filt_2:
            final_filt = list(filt_1.values())
    elif '1' in filts:
        final_filt = list(filts['1'])
    elif '2' in filts:
        final_filt = list(filts['2'])
    else:
        raise ValueError(f"Multiple active filters found in {file}")
    return Counter([tuple(f.items()) for f in final_filt])
