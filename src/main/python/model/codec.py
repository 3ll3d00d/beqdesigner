import numpy as np
import logging

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
    if signal.duration_hhmmss is not None:
        out['duration_hhmmss'] = signal.duration_hhmmss
        out['start_hhmmss'] = signal.start_hhmmss
        out['end_hhmmss'] = signal.end_hhmmss
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
    from model.signal import SignalData
    if '_type' not in o:
        raise ValueError(f"{o} is not SignalData")
    elif o['_type'] == SignalData.__name__:
        filt = o.get('filter', None)
        if filt is not None:
            filt = filter_from_json(filt)
        data = o['data']
        avg = xydata_from_json(data['avg'])
        peak = xydata_from_json(data['peak'])
        return SignalData(o['name'], o['fs'], [avg, peak], filter=filt, duration_hhmmss=o.get('duration_hhmmss', None),
                          start_hhmmss=o.get('start_hhmmss', None), end_hhmmss=o.get('end_hhmmss', None))
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

    if '_type' not in o:
        raise ValueError(f"{o} is not a filter")
    if o['_type'] == Passthrough.__name__:
        return Passthrough()
    elif o['_type'] == PeakingEQ.__name__:
        return PeakingEQ(o['fs'], o['fc'], o['q'], o['gain'])
    elif o['_type'] == LowShelf.__name__:
        return LowShelf(o['fs'], o['fc'], o['q'], o['gain'], o['count'])
    elif o['_type'] == HighShelf.__name__:
        return HighShelf(o['fs'], o['fc'], o['q'], o['gain'], o['count'])
    elif o['_type'] == FirstOrder_LowPass.__name__:
        return FirstOrder_LowPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == FirstOrder_HighPass.__name__:
        return FirstOrder_HighPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == SecondOrder_LowPass.__name__:
        return SecondOrder_LowPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == SecondOrder_HighPass.__name__:
        return SecondOrder_HighPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == AllPass.__name__:
        return AllPass(o['fs'], o['fc'], o['q'])
    elif o['_type'] == CompleteFilter.__name__:
        return CompleteFilter(filters=[filter_from_json(x) for x in o['filters']], description=o['description'])
    elif o['_type'] == ComplexLowPass.__name__:
        return ComplexLowPass(FilterType(o['filter_type']), o['order'], o['fs'], o['fc'])
    elif o['_type'] == ComplexHighPass.__name__:
        return ComplexHighPass(FilterType(o['filter_type']), o['order'], o['fs'], o['fc'])
    raise ValueError(f"{o._type} is an unknown filter type")


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
