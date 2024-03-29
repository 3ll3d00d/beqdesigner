import logging
import os
from uuid import uuid4

import numpy as np

from model.iir import Gain

logger = logging.getLogger('codec')


def bassmanagedsignaldata_to_json(signal):
    '''
    Converts the bass managed signal to a json compatible format.
    :param signal: the signal.
    :return: a dict to write to json.
    '''
    out = {
        '_type': signal.__class__.__name__,
        'name': signal.name,
        'channels': [signaldata_to_json(x) for x in signal.channels],
        'bm': {
            'lpf_fs': signal.bm_lpf_fs,
            'lpf_position': signal.bm_lpf_position,
            'headroom_type': signal.bm_headroom_type
        },
        'clip': {
            'before': signal.clip_before,
            'after': signal.clip_after
        },
        'offset': f"{signal.offset:g}"
    }
    return out


def signaldata_to_json(signal):
    '''
    Converts the signal to a json compatible format.
    :return: a dict to write to json.
    '''
    avg = signal.current_unfiltered[0]
    peak = signal.current_unfiltered[1]
    median = signal.current_unfiltered[2] if len(signal.current_unfiltered) == 3 else None
    out = {
        '_type': signal.__class__.__name__,
        'name': signal.name,
        'fs': signal.fs,
        'data': {
            'avg': xydata_to_json(avg),
            'peak': xydata_to_json(peak),
        },
        'offset': f"{signal.offset:g}"
    }
    if median is not None:
        out['data']['median'] = xydata_to_json(median)
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


def signalmodel_from_json(input, preferences):
    '''
    Reassembles all signals from the json including master/slave relationships.
    :param input: the input, a list of json dicts.
    :return: the signals
    '''
    signals = [signaldata_from_json(x, preferences) for x in input]
    from model.signal import SingleChannelSignalData, BassManagedSignalData
    single_signals = [x for x in signals if isinstance(x, SingleChannelSignalData)]
    bm_channels = [y for x in signals if isinstance(x, BassManagedSignalData) for y in x.channels]
    channels = single_signals + bm_channels
    for x in input:
        if x['_type'] == BassManagedSignalData.__name__:
            for signal_json in x['channels']:
                if 'slave_names' in signal_json:
                    __link_master_slave(channels, signal_json)
        else:
            if 'slave_names' in x:
                __link_master_slave(channels, x)
    return signals


def __link_master_slave(channels, signal_json):
    master_name = signal_json['name']
    logger.debug(f"Reassembling slaves for {master_name}")
    master = next((s for s in channels if s.name == master_name), None)
    if master is not None:
        for slave_name in signal_json['slave_names']:
            slave = next((s for s in channels if s.name == slave_name), None)
            if slave is not None:
                master.enslave(slave)
            else:
                logger.error(f"Bad json encountered, slave not decoded ({master_name} -> {slave_name})")
    else:
        logger.error(f"Bad json encountered, master {master_name} not decoded")


def signaldata_from_json(o, preferences):
    '''
    Converts the given dict to a SignalData if it is compatible.
    :param o: the dict (from json).
    :return: the SignalData (or an error)
    '''
    from model.signal import SingleChannelSignalData, BassManagedSignalData
    if '_type' not in o:
        raise ValueError(f"{o} is not SignalData")
    elif o['_type'] == BassManagedSignalData.__name__:
        channels = [signaldata_from_json(c, preferences) for c in o['channels']]
        lpf_fs = float(o['bm']['lpf_fs'])
        lpf_position = o['bm']['lpf_position']
        offset = float(o['offset'])
        bmsd = BassManagedSignalData(channels, lpf_fs, lpf_position, preferences, offset=offset)
        bmsd.bm_headroom_type = o['bm']['headroom_type']
        bmsd.clip_before = o['clip']['before']
        bmsd.clip_after = o['clip']['after']
        return bmsd
    elif o['_type'] == SingleChannelSignalData.__name__ or o['_type'] == 'SignalData':
        filt = o.get('filter', None)
        if filt is not None:
            filt = filter_from_json(filt)
        data = o['data']
        avg = xydata_from_json(data['avg'])
        peak = xydata_from_json(data['peak'])
        median = xydata_from_json(data['median']) if 'median' in data else None
        xy_data = [avg, peak] if median is None else [avg, peak, median]
        metadata = o.get('metadata', None)
        offset = float(o.get('offset', 0.0))
        signal = None
        if metadata is not None:
            try:
                if os.path.isfile(metadata['src']):
                    from model.signal import readWav
                    signal = readWav(o['name'], preferences, input_file=metadata['src'], channel=metadata['channel'],
                                     start=metadata['start'], end=metadata['end'], target_fs=o['fs'], offset=offset)
            except:
                logger.exception(f"Unable to load signal from {metadata['src']}")
        if 'duration_seconds' in o:
            signal_data = SingleChannelSignalData(o['name'], o['fs'], xy_data=xy_data, filter=filt,
                                                  duration_seconds=o.get('duration_seconds', None),
                                                  start_seconds=o.get('start_seconds', None),
                                                  signal=signal,
                                                  offset=offset)
        elif 'duration_hhmmss' in o:
            h, m, s = o['duration_hhmmss'].split(':')
            duration_seconds = (int(h) * 3600) + int(m) * (60 + float(s))
            h, m, s = o['start_hhmmss'].split(':')
            start_seconds = (int(h) * 3600) + int(m) * (60 + float(s))
            signal_data = SingleChannelSignalData(o['name'], o['fs'], xy_data=xy_data, filter=filt,
                                                  duration_seconds=duration_seconds,
                                                  start_seconds=start_seconds,
                                                  signal=signal,
                                                  offset=offset)
        else:
            signal_data = SingleChannelSignalData(o['name'], o['fs'], xy_data=xy_data, filter=filt, signal=signal,
                                                  offset=offset)
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
        filt = FirstOrder_LowPass(o['fs'], o['fc'])
    elif o['_type'] == FirstOrder_HighPass.__name__:
        filt = FirstOrder_HighPass(o['fs'], o['fc'])
    elif o['_type'] == SecondOrder_LowPass.__name__:
        filt = SecondOrder_LowPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == SecondOrder_HighPass.__name__:
        filt = SecondOrder_HighPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == AllPass.__name__:
        filt = AllPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == CompleteFilter.__name__:
        kwargs = {}
        if 'fs' in o:
            kwargs['fs'] = o['fs']
        filt = CompleteFilter(filters=[filter_from_json(x) for x in o['filters']], description=o['description'], **kwargs)
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
    Converts a json dict to a MagnitudeData.
    :param o: the dict.
    :return: the MagnitudeData (or an error).
    '''
    from model.xy import MagnitudeData
    if '_type' not in o:
        raise ValueError(f"{o} is not MagnitudeData")
    elif o['_type'] == MagnitudeData.__name__ or o['_type'] == 'XYData':
        x_json = o['x']
        if 'count' in x_json:
            x_vals = np.linspace(x_json['min'], x_json['max'], num=x_json['count'], dtype=np.float64)
        else:
            x_vals = np.array(x_json)
        description = o['description'] if 'description' in o else ''
        return MagnitudeData(o['name'], description, x_vals, np.array(o['y']), colour=o.get('colour', None),
                             linestyle=o.get('linestyle', '-'))
    raise ValueError(f"{o['_type']} is an unknown data type")


def xydata_to_json(data):
    '''
    A json compatible rendering of the xy data.
    :return: a dict.
    '''
    return {
        '_type': data.__class__.__name__,
        'name': data.internal_name,
        'description': data.internal_description,
        'x': np.around(data.x, decimals=6).tolist(),
        'y': np.around(data.y, decimals=6).tolist(),
        'colour': data.colour,
        'linestyle': data.linestyle
    } if data else {}
