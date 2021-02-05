import logging
import os
import xml.etree.ElementTree as et
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, Callable, Tuple

import qtawesome as qta
from qtpy.QtWidgets import QDialog, QFileDialog, QVBoxLayout, QLabel, QListWidget, QFrame, QHBoxLayout, QToolButton
from scipy.signal import unit_impulse

from model import iir
from model.iir import CompleteFilter, s_to_q, q_to_s
from model.limits import dBRangeCalculator, PhaseRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import BEQ_CONFIG_FILE, JRIVER_GEOMETRY, get_filter_colour, JRIVER_GRAPH_X_MIN, \
    JRIVER_GRAPH_X_MAX
from ui.jriver import Ui_jriverDspDialog
from ui.pipeline import Ui_jriverGraphDialog

USER_CHANNELS = ['User 1', 'User 2']
SHORT_USER_CHANNELS = ['U1', 'U2']
JRIVER_NAMED_CHANNELS = [None, None, 'Left', 'Right', 'Centre', 'Subwoofer', 'Surround Left', 'Surround Right',
                         'Rear Left', 'Rear Right', None] + USER_CHANNELS
JRIVER_SHORT_NAMED_CHANNELS = [None, None, 'L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR', None] + SHORT_USER_CHANNELS
JRIVER_CHANNELS = JRIVER_NAMED_CHANNELS + [f"Channel {i + 9}" for i in range(24)]
JRIVER_SHORT_CHANNELS = JRIVER_SHORT_NAMED_CHANNELS + [f"#{i + 9}" for i in range(24)]

logger = logging.getLogger('jriver')


def get_channel_name(idx: int, short: bool = False) -> str:
    '''
    Converts a channel index to a named channel.
    :param idx: the index.
    :param short: get the short name if true.
    :return: the name.
    '''
    channels = JRIVER_SHORT_CHANNELS if short else JRIVER_CHANNELS
    return channels[idx]


def get_channel_idx(name: str, short: bool = False) -> int:
    '''
    Converts a channel name to an index.
    :param name the name.
    :param short: search via short name if true.
    :return: the index.
    '''
    channels = JRIVER_SHORT_CHANNELS if short else JRIVER_CHANNELS
    return channels.index(name)


class JRiverDSPDialog(QDialog, Ui_jriverDspDialog):

    def __init__(self, parent, prefs):
        super(JRiverDSPDialog, self).__init__(parent)
        self.setupUi(self)
        self.limitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.fullRangeButton.setIcon(qta.icon('fa5s.expand'))
        self.subOnlyButton.setIcon(qta.icon('fa5s.compress'))
        self.findFilenameButton.setIcon(qta.icon('fa5s.folder-open'))
        self.__channel_controls: Dict[str, ChannelFilterControl] = {}
        self.prefs = prefs
        self.__dsp: Optional[JRiverDSP] = None
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, prefs,
                                                self.__get_data(), 'Filter', fill_primary=True,
                                                x_min_pref_key=JRIVER_GRAPH_X_MIN, x_max_pref_key=JRIVER_GRAPH_X_MAX,
                                                secondary_data_provider=self.__get_data('phase'),
                                                secondary_name='Phase', secondary_prefix='deg', fill_secondary=False,
                                                db_range_calc=dBRangeCalculator(30, expand=True),
                                                y2_range_calc=PhaseRangeCalculator(), show_y2_in_legend=False)
        self.__restore_geometry()
        self.showGraphButton.clicked.connect(self.__show_graph)

    def __show_graph(self):
        if self.__dsp is not None:
            peq_idx = self.blockSelector.currentIndex()
            in_c = self.__dsp.channel_names(short=True, output=False)
            out_c = self.__dsp.channel_names(short=True, output=True)
            JRiverFilterPipelineDialog(self.parent(), in_c, out_c, self.__dsp.peq(peq_idx)).exec()

    def redraw(self):
        self.__magnitude_model.redraw()

    def __restore_geometry(self):
        ''' loads the saved window size '''
        geometry = self.prefs.get(JRIVER_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def find_dsp_file(self):
        beq_config_file = self.prefs.get(BEQ_CONFIG_FILE)
        kwargs = {
            'caption': 'Select JRiver Media Centre DSP File',
            'filter': 'DSP (*.dsp)'
        }
        if beq_config_file is not None and len(beq_config_file) > 0 and os.path.exists(beq_config_file):
            kwargs['directory'] = str(Path(beq_config_file).parent.resolve())
        selected = QFileDialog.getOpenFileName(parent=self, **kwargs)
        if selected is not None and len(selected[0]) > 0:
            try:
                self.channelList.clear()
                self.__dsp = JRiverDSP(selected[0])
                for n in self.__dsp.channel_names():
                    self.channelList.addItem(n)
                self.filename.setText(selected[0])
                self.initialise_channel_filters()
                self.show_filters()
                self.channelList.selectAll()
            except Exception as e:
                logger.exception(f"Unable to parse {selected[0]}")
                from model.catalogue import show_alert
                show_alert('Unable to load DSP file', f"Invalid file\n\n{e}")

    def initialise_channel_filters(self):
        '''
        Creates and wires up the per channel filter display.
        '''
        for channel_name in self.__dsp.channel_names():
            channel_idx = get_channel_idx(channel_name)
            if channel_name not in self.__channel_controls:
                limits = self.__magnitude_model.limits
                self.__channel_controls[channel_name] = ChannelFilterControl(self, self.__dsp, channel_idx,
                                                                             self.blockSelector.currentIndex,
                                                                             lambda: (limits.x_min, limits.x_max))

    def show_filters(self):
        '''
        Displays the complete filter list for the selected PEQ block.
        '''
        if self.__dsp is not None:
            peq_idx = self.blockSelector.currentIndex()
            self.filterList.clear()
            for f in self.__dsp.peq(peq_idx):
                self.filterList.addItem(str(f))
            for c in self.__channel_controls.values():
                c.display(peq_idx)

    def show_channel_filters(self):
        '''
        Displays the selected per channel filters.
        '''
        selected_channels = [i.text() for i in self.channelList.selectedItems()]
        for name, c in self.__channel_controls.items():
            if name in selected_channels:
                c.show()
            else:
                c.hide()
        self.redraw()

    def __get_data(self, mode='mag'):
        return lambda *args, **kwargs: self.get_curve_data(mode, *args, **kwargs)

    def get_curve_data(self, mode, reference=None):
        ''' preview of the filter to display on the chart '''
        result = []
        if mode == 'mag' or self.showPhase.isChecked():
            for p in self.__channel_controls.values():
                f = p.as_filter()
                if f:
                    result.append(f.get_transfer_function().get_data(mode=mode, colour=get_filter_colour(len(result))))
        return result

    def show_limits(self):
        ''' shows the limits dialog for the filter chart. '''
        self.__magnitude_model.show_limits()

    def show_full_range(self):
        ''' sets the limits to full range. '''
        self.__magnitude_model.show_full_range()

    def show_sub_only(self):
        ''' sets the limits to sub only. '''
        self.__magnitude_model.show_sub_only()

    def show_phase_response(self):
        self.redraw()

    def closeEvent(self, QCloseEvent):
        ''' Stores the window size on close '''
        self.prefs.set(JRIVER_GEOMETRY, self.saveGeometry())
        super().closeEvent(QCloseEvent)


class JRiverDSP:

    def __init__(self, filename):
        self.__filename = filename
        config_txt = Path(self.__filename).read_text()
        i, o = get_available_channels(config_txt)
        self.__input_channel_indexes = i
        self.__output_channel_indexes = o
        self.__peqs: List[List[Filter]] = [self.__parse_peq(config_txt, 1), self.__parse_peq(config_txt, 2)]

    def channel_names(self, short=False, output=False):
        idxs = self.__output_channel_indexes if output else self.__input_channel_indexes
        return [get_channel_name(i, short=short) for i in idxs]

    @staticmethod
    def channel_name(i):
        return get_channel_name(i)

    def peq(self, idx):
        return self.__peqs[idx]

    def __parse_peq(self, xml, block):
        peq_block = get_peq_key_name(block)
        _, filt_element = extract_filters(xml, peq_block)
        filt_fragments = [v + ')' for v in filt_element.text.split(')') if v]
        if len(filt_fragments) < 2:
            raise ValueError('Invalid input file - Unexpected <Value> format')
        return [create_peq(d) for d in [self.__item_to_dicts(f) for f in filt_fragments[2:]] if d]

    @staticmethod
    def __item_to_dicts(frag) -> Optional[Dict[str, str]]:
        if frag.find(':') > -1:
            peq_xml = frag.split(':')[1][:-1]
            vals = {i.attrib['Name']: i.text for i in et.fromstring(peq_xml).findall('./Item')}
            if 'Enabled' in vals:
                if vals['Enabled'] != '0' and vals['Enabled'] != '1':
                    vals['Enabled'] = '1'
            else:
                vals['Enabled'] = '0'
            return vals
        return None

    def __repr__(self):
        return f"{self.__filename}"


class Filter:

    def __init__(self, vals, short_name):
        self.__vals = vals
        self.__short_name = short_name
        self.__enabled = vals['Enabled'] == '1'
        self.__type_code = vals['Type']

    @property
    def enabled(self):
        return self.__enabled

    def encode(self):
        return filt_to_xml(self.get_all_vals())

    def get_all_vals(self) -> Dict[str, str]:
        vals = {
            'Enabled': '1' if self.__enabled else '0',
            'Type': self.__type_code,
            **self.get_vals()
        }
        return vals

    def get_vals(self) -> Dict[str, str]:
        return {}

    def get_filter(self):
        return None

    def print_disabled(self):
        return '' if self.enabled else f" *** DISABLED ***"

    @property
    def short_name(self):
        return self.__short_name

    def is_mine(self, idx):
        return True


class ChannelFilter(Filter):

    def __init__(self, vals, short_name):
        super().__init__(vals, short_name)
        self.__channels = [int(c) for c in vals['Channels'].split(';')]
        self.__channel_names = [get_channel_name(i) for i in self.__channels]

    @property
    def channels(self):
        return self.__channels

    @property
    def channel_names(self):
        return self.__channel_names

    def get_vals(self) -> Dict[str, str]:
        return {
            'Channels': ';'.join([str(c) for c in self.channels]),
            **super().get_vals()
        }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.print_channel_names()}{self.print_disabled()}"

    def print_channel_names(self):
        return f"[{', '.join(self.channel_names)}]"

    def for_channel(self, idx):
        '''
        Copies the filter overriding the channel to the provided idx.
        :param idx: the channel idx.
        :return: the copy.
        '''
        vals = self.get_all_vals()
        vals['Channels'] = str(idx)
        return type(self)(vals)

    def is_mine(self, idx):
        return idx in self.channels


class GainQFilter(ChannelFilter):

    def __init__(self, vals, create_iir, short_name):
        super().__init__(vals, short_name)
        self.__create_iir = create_iir
        self.__gain = float(vals['Gain'])
        self.__frequency = float(vals['Frequency'])
        self.__q = self.from_jriver_q(float(vals['Q']), self.__gain)

    def get_vals(self) -> Dict[str, str]:
        return {
            'Slope': '12',
            'Q': f"{self.to_jriver_q(self.__q, self.__gain):.7g}",
            'Gain': f"{self.__gain:.7g}",
            'Frequency': f"{self.__frequency:.7g}",
            **super().get_vals()
        }

    def from_jriver_q(self,  q: float, gain: float):
        return q

    def to_jriver_q(self,  q: float, gain: float):
        return q

    def get_filter(self):
        return self.__create_iir(48000, self.__frequency, self.__q, self.__gain)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__gain:+.7g} dB Q={self.__q:.7g} at {self.__frequency:.7g} Hz {self.print_channel_names()}{self.print_disabled()}"


class Peak(GainQFilter):

    def __init__(self, vals):
        super().__init__(vals, iir.PeakingEQ, 'Peak')


class LowShelf(GainQFilter):

    def __init__(self, vals):
        super().__init__(vals, iir.LowShelf, 'LS')

    def from_jriver_q(self, q: float, gain: float):
        return s_to_q(q, gain)

    def to_jriver_q(self, q: float, gain: float):
        return q_to_s(q, gain)


class HighShelf(GainQFilter):

    def __init__(self, vals):
        super().__init__(vals, iir.HighShelf, 'HS')

    def from_jriver_q(self, q: float, gain: float):
        return s_to_q(q, gain)

    def to_jriver_q(self, q: float, gain: float):
        return q_to_s(q, gain)


class LowPass(GainQFilter):

    def __init__(self, vals):
        super().__init__(vals, iir.SecondOrder_LowPass, 'LPF')

    def from_jriver_q(self, q: float, gain: float):
        return q / 2**0.5

    def to_jriver_q(self, q: float, gain: float):
        return q * 2**0.5


class HighPass(GainQFilter):

    def __init__(self, vals):
        super().__init__(vals, iir.SecondOrder_HighPass, 'HPF')

    def from_jriver_q(self, q: float, gain: float):
        return q / 2**0.5

    def to_jriver_q(self, q: float, gain: float):
        return q * 2**0.5


class Gain(ChannelFilter):

    def __init__(self, vals):
        super().__init__(vals, 'GAIN')
        self.__gain = float(vals['Gain'])

    def get_vals(self) -> Dict[str, str]:
        return {
            'Gain': f"{self.__gain:.7g}",
            **super().get_vals()
        }

    def __repr__(self):
        return f"Gain {self.__gain:+.7g} dB {self.print_channel_names()}{self.print_disabled()}"

    def get_filter(self):
        return iir.Gain(48000, self.__gain)


class BitdepthSimulator(Filter):

    def __init__(self, vals):
        super().__init__(vals, 'BITDEPTH')
        self.__bits = int(vals['Bits'])
        self.__dither = vals['Dither']

    def get_vals(self) -> Dict[str, str]:
        return {
            'Bits': str(self.__bits),
            'Dither': self.__dither
        }

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.__bits} bits){self.print_disabled()}"


class Delay(ChannelFilter):

    def __init__(self, vals):
        super().__init__(vals, 'DELAY')
        self.__delay = float(vals['Delay'])

    def get_vals(self) -> Dict[str, str]:
        return {
            'Delay': f"{self.__delay:.7g}",
            **super().get_vals()
        }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__delay:+.7g} ms {self.print_channel_names()}{self.print_disabled()}"


class Divider(Filter):

    def __init__(self, vals):
        super().__init__(vals, '---')
        self.__text = vals['Text'] if 'Text' in vals else ''

    def get_vals(self) -> Dict[str, str]:
        return {
            'Text': self.__text
        }

    def __repr__(self):
        return self.__text


class LimiterMode(Enum):
    BRICKWALL = 0
    ADAPTIVE = 1


class Limiter(ChannelFilter):

    def __init__(self, vals):
        super().__init__(vals, 'LIMITER')
        self.__hold = vals['Hold']
        self.__mode = vals['Mode']
        self.__level = vals['Level']
        self.__release = vals['Release']
        self.__attack = vals['Attack']

    def get_vals(self) -> Dict[str, str]:
        return {
            'Hold': self.__hold,
            'Mode': self.__mode,
            'Level': self.__level,
            'Release': self.__release,
            'Attack': self.__attack,
            **super().get_vals()
        }

    def __repr__(self):
        return f"{LimiterMode(int(self.__mode)).name.capitalize()} {self.__class__.__name__} at {self.__level} dB {self.print_channel_names()}{self.print_disabled()}"


class LinkwitzTransform(ChannelFilter):

    def __init__(self, vals):
        super().__init__(vals, 'LT')
        self.__fp = float(vals['Fp'])
        self.__qp = float(vals['Qp'])
        self.__fz = float(vals['Fz'])
        self.__qz = float(vals['Qz'])
        self.__prevent_clipping = vals['PreventClipping']

    def get_vals(self) -> Dict[str, str]:
        return {
            'Fp': self.__fp,
            'Qp': self.__qp,
            'Fz': self.__fz,
            'Qz': self.__qz,
            'PreventClipping': self.__prevent_clipping,
            **super().get_vals()
        }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__fz:.7g} Hz / {self.__qz:.7g} -> {self.__fp:.7g} Hz / {self.__qp:.7} {self.print_channel_names()}{self.print_disabled()}"


class LinkwitzRiley(Filter):

    def __init__(self, vals):
        super().__init__(vals, 'LR')
        self.__freq = float(vals['Frequency'])

    def get_vals(self) -> Dict[str, str]:
        return {
            'Frequency': f"{self.__freq:.7g}"
        }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__freq:.7g} Hz{self.print_disabled()}"


class MidSideDecoding(Filter):

    def __init__(self, vals):
        super().__init__(vals, 'MS Decode')

    def get_vals(self) -> Dict[str, str]:
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}{self.print_disabled()}"


class MidSideEncoding(Filter):

    def __init__(self, vals):
        super().__init__(vals, 'MS Encode')

    def get_vals(self) -> Dict[str, str]:
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}{self.print_disabled()}"


class MixType(Enum):
    ADD = 0
    COPY = 1
    MOVE = 2
    SWAP = 3, 'and'
    SUBTRACT = 4, 'from'

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, modifier: str = 'to'):
        self.___modifier = modifier

    def __str__(self):
        return self.value

    @property
    def modifier(self):
        return self.___modifier


class Mix(Filter):

    def __init__(self, vals):
        super().__init__(vals, f"{MixType(int(vals['Mode'])).name.capitalize()}")
        self.__source = vals['Source']
        self.__destination = vals['Destination']
        self.__gain = float(vals['Gain'])
        # mode: 3 = swap, 1 = copy, 2 = move, 0 = add, 4 = subtract
        self.__mode = vals['Mode']

    @property
    def src_idx(self):
        return int(self.__source)

    @property
    def dst_idx(self):
        return int(self.__destination)

    @property
    def mix_type(self):
        return MixType(int(self.__mode)).name

    def get_vals(self) -> Dict[str, str]:
        return {
            'Source': self.__source,
            'Destination': self.__destination,
            'Gain': f"{self.__gain:.7g}",
            'Mode': self.__mode
        }

    def __repr__(self):
        mix_type = MixType(int(self.__mode))
        src_name = get_channel_name(int(self.__source), short=True)
        dst_name = get_channel_name(int(self.__destination), short=True)
        return f"{mix_type.name.capitalize()} {src_name} {mix_type.modifier} {dst_name} {self.__gain:+.7g} dB{self.print_disabled()}"

    def is_mine(self, idx):
        return self.src_idx == idx


class Order(Filter):

    def __init__(self, vals):
        super().__init__(vals, 'ORDER')
        self.__order = vals['Order'].split(',')
        self.__named_order = [get_channel_name(int(i)) for i in self.__order]

    def get_vals(self) -> Dict[str, str]:
        return {
            'Order': ','.join(self.__order)
        }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__named_order}{self.print_disabled()}"


class Mute(ChannelFilter):

    def __init__(self, vals):
        super().__init__(vals, 'MUTE')
        self.__gain = float(vals['Gain'])

    def get_vals(self) -> Dict[str, str]:
        return {
            'Gain': f"{self.__gain:.7g}",
            **super().get_vals()
        }


class Polarity(ChannelFilter):

    def __init__(self, vals):
        super().__init__(vals, 'INVERT')


class SubwooferLimiter(ChannelFilter):

    def __init__(self, vals):
        super().__init__(vals, 'SW Limiter')
        self.__level = float(vals['Level'])

    def get_vals(self) -> Dict[str, str]:
        return {
            'Level': f"{self.__level:.7g}",
            **super().get_vals()
        }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__level:+.7g} dB {self.print_channel_names()}{self.print_disabled()}"


def create_peq(vals: Dict[str, str]) -> Filter:
    '''
    :param vals: the vals from the encoded xml format.
    :param channels: the available channel names.
    :return: a filter type.
    '''
    peq_type = vals['Type']
    if peq_type == '3':
        return Peak(vals)
    elif peq_type == '11':
        return HighShelf(vals)
    elif peq_type == '10':
        return LowShelf(vals)
    elif peq_type == '4':
        return Gain(vals)
    elif peq_type == '13':
        return BitdepthSimulator(vals)
    elif peq_type == '7':
        return Delay(vals)
    elif peq_type == '20':
        return Divider(vals)
    elif peq_type == '9':
        return Limiter(vals)
    elif peq_type == '8':
        return LinkwitzTransform(vals)
    elif peq_type == '16':
        return LinkwitzRiley(vals)
    elif peq_type == '19':
        return MidSideDecoding(vals)
    elif peq_type == '18':
        return MidSideEncoding(vals)
    elif peq_type == '6':
        return Mix(vals)
    elif peq_type == '5':
        return Mute(vals)
    elif peq_type == '12':
        return Order(vals)
    elif peq_type == '1':
        return LowPass(vals)
    elif peq_type == '2':
        return HighPass(vals)
    elif peq_type == '15':
        return Polarity(vals)
    elif peq_type == '14':
        return SubwooferLimiter(vals)
    else:
        raise ValueError(f"Unknown type {vals}")


class ChannelFilterControl:

    def __init__(self, parent: JRiverDSPDialog, dsp: JRiverDSP, channel_idx: int, selected_block: Callable[[], int],
                 x_lim: Callable[[], Tuple[int, int]]):
        self.__channel_idx = channel_idx
        self.__selected_block_provider = selected_block
        self.__x_lim_provider = x_lim
        self.dsp = dsp
        self.parent = parent
        self.frame = QFrame(parent)
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Sunken)
        self.layout = QVBoxLayout(self.frame)
        self.paramLayout = QHBoxLayout()
        self.layout.addLayout(self.paramLayout)
        self.label = QLabel(self.frame)
        self.label.setText(get_channel_name(channel_idx))
        self.paramLayout.addWidget(self.label)
        self.editButton = QToolButton(self.frame)
        self.editButton.setIcon(qta.icon('fa5s.edit'))
        self.editButton.clicked.connect(self.edit_filter)
        self.paramLayout.addWidget(self.editButton)
        self.filterList = QListWidget(self.frame)
        self.layout.addWidget(self.filterList)
        parent.perChannelLayout.addWidget(self.frame)
        self.hide()
        self.peqs: List[List[ChannelFilter]] = [self.__make_channel_filters(idx) for idx in range(2)]

    def __make_channel_filters(self, idx) -> List[ChannelFilter]:
        return [self.__for_me(f) for f in self.dsp.peq(idx) if self.__is_mine(f)]

    def __for_me(self, f: Filter):
        if isinstance(f, ChannelFilter):
            return f.for_channel(self.__channel_idx)
        raise TypeError(f"Can only own ChannelFilter not {f.__class__.__name__}")

    def edit_filter(self):
        from model.filter import FilterDialog, FilterModel
        from model.signal import Signal, SingleChannelSignalData
        signal = SingleChannelSignalData(name='default',
                                         signal=Signal('default', unit_impulse(48000, 'mid'), self.parent.prefs,
                                                       fs=48000),
                                         filter=CompleteFilter(fs=48000))
        filter_model = FilterModel(None, self.parent.prefs)
        filter_model.filter = self.as_filter()
        FilterDialog(self.parent.prefs, signal, filter_model, self.parent.redraw,
                     # selected_filter=signal.filter[selection.selectedRows()[0].row()]
                     parent=self.parent, x_lim=self.__x_lim_provider()).show()

    def __is_mine(self, filt: Filter):
        if isinstance(filt, ChannelFilter):
            return filt.is_mine(self.__channel_idx)
        return False

    def display(self, idx: int):
        '''
        Displays the specified peq block
        :param idx: the peq block idx.
        '''
        self.filterList.clear()
        for f in self.peqs[idx]:
            self.filterList.addItem(str(f))

    def show(self):
        ''' Makes the control visible. '''
        self.frame.setVisible(True)

    def hide(self):
        ''' Makes the control invisible. '''
        self.frame.setVisible(False)

    def as_filter(self):
        '''
        converts to a beqd filter.
        :return: the filter.
        '''
        if self.frame.isVisible():
            return CompleteFilter(fs=48000, filters=self.__make_filters(),
                                  description=get_channel_name(self.__channel_idx))
        return None

    def __make_filters(self):
        idx = self.__selected_block_provider()
        return [p.get_filter() for p in self.peqs[idx]]


def convert_filter_to_mc_dsp(filt, target_channels):
    '''
    :param filt: a filter.
    :param target_channels: the channels to output to.
    :return: values to put in an xml fragment.
    '''
    from model.iir import PeakingEQ, LowShelf, Gain, BiquadWithQGain, q_to_s
    if isinstance(filt, BiquadWithQGain):
        return {
            'Enabled': 1,
            'Slope': 12,
            'Q': f"{filt.q if isinstance(filt, PeakingEQ) else q_to_s(filt.q, filt.gain):.7g}",
            'Type': 3 if isinstance(filt, PeakingEQ) else 10 if isinstance(filt, LowShelf) else 11,
            'Gain': f"{filt.gain:.7g}",
            'Frequency': f"{filt.freq:.7g}",
            'Channels': target_channels
        }
    elif isinstance(filt, Gain):
        return {
            'Enabled': 1,
            'Type': 4,
            'Gain': f"{filt.gain:.7g}",
            'Channels': target_channels
        }
    else:
        raise ValueError(f"Unsupported filter type {filt}")


def xpath_to_key_data_value(key_name, data_name):
    '''
    an ET compatible xpath to get the value from a DSP config via the path /Preset/Key/Data/Value for a given key and
    data.
    :param key_name:
    :param data_name:
    :return:
    '''
    return f"./Preset/Key[@Name=\"{key_name}\"]/Data/Name[.=\"{data_name}\"]/../Value"


def get_text_value(root, xpath):
    matches = root.findall(xpath)
    if matches:
        if len(matches) == 1:
            return matches[0].text
        else:
            raise ValueError(f"Multiple matches for {xpath}")
    else:
        raise ValueError(f"No matches for {xpath}")


def get_available_channels(config_txt) -> Tuple[List[int], List[int]]:
    '''
    :param config_txt: the dsp config.
    :return: the channel indexes in the config file as (input, output).
    '''
    root = et.fromstring(config_txt)

    def xpath_val(key):
        return get_text_value(root, xpath_to_key_data_value('Audio Settings', key))

    output_channels = int(xpath_val('Output Channels'))
    padding = int(xpath_val('Output Padding Channels'))
    layout = int(xpath_val('Output Channel Layout'))
    user_channel_indexes = [11, 12]
    if output_channels == 0:
        # source number of channels so assume 7.1
        channels = [i + 2 for i in range(8)]
        return channels, channels + user_channel_indexes
    elif output_channels == 3:
        # 2.1
        return [2, 3, 5], [2, 3, 5] + user_channel_indexes
    elif output_channels == 4 and layout == 15:
        # 3.1
        channels = [i+2 for i in range(4)]
        return channels, channels + user_channel_indexes
    elif output_channels == 2 and padding == 2:
        # 2 in 4 channel container
        return [2, 3], [i+2 for i in range(4)] + user_channel_indexes
    elif output_channels == 2 and padding == 4:
        # 2 in 5.1 channel container
        return [2, 3], [i+2 for i in range(6)] + user_channel_indexes
    elif output_channels == 6 and padding == 2:
        # 5.1 in 7.1 channel container
        return [i+2 for i in range(6)], [i+2 for i in range(8)] + user_channel_indexes
    elif output_channels == 4:
        # 4 channel
        channels = [i + 2 for i in range(4)]
        return channels, channels + user_channel_indexes
    elif output_channels == 6 or output_channels == 8:
        channels = [i + 2 for i in range(output_channels)]
        return channels, channels + user_channel_indexes
    else:
        excess = output_channels - 8
        if excess < 1:
            raise ValueError(f"Illegal combination [ch: {output_channels}, p: {padding}, l: {layout}")
        numbered = [i + len(JRIVER_SHORT_NAMED_CHANNELS) for i in range(excess + 1)]
        return [i+2 for i in range(8)] + numbered, [i+2 for i in range(8)] + user_channel_indexes + numbered


def write_dsp_file(root, file_name):
    '''
    :param root: the root element.
    :param file_name: the file to write to.
    '''
    tree = et.ElementTree(root)
    tree.write(file_name, encoding='UTF-8', xml_declaration=True)


def extract_filters(config_txt, key_name):
    '''
    :param config_txt: the xml text.
    :param key_name: the filter key name.
    :return: (root element, filter element)
    '''
    root = et.fromstring(config_txt)
    elements = root.findall(xpath_to_key_data_value(key_name, 'Filters'))
    if elements and len(elements) == 1:
        return root, elements[0]
    raise ValueError(f"No Filters in {key_name} found in {config_txt}")


def get_peq_key_name(block):
    '''
    :param block: 1 or 2.
    :return: the PEQ key name.
    '''
    if block == 1:
        return 'Parametric Equalizer'
    elif block == 2:
        return 'Parametric Equalizer 2'
    else:
        raise ValueError(f"Unknown PEQ block {block}")


def filt_to_xml(vals: Dict[str, str]) -> str:
    '''
    Formats key-value pairs into a jriver dsp config file compatible str fragment.
    :param vals: the key-value pairs.
    :return: the txt snippet.
    '''
    items = [f"<Item Name=\"{k}\">{v}</Item>" for k, v in vals.items()]
    catted_items = '\n'.join(items)
    prefix = '<XMLPH version="1.1">'
    suffix = '</XMLPH>'
    # 1 for each new line
    segment_length = len(prefix) + 1 + len(' '.join(items)) + 1 + len(suffix) + 1
    # 3 for the (:) formatting + 1 for another new line
    segment_length = segment_length + len(str(segment_length)) + 3 + 1
    xml_frag = f"({segment_length}:{prefix}\n{catted_items}\n{suffix})"
    return xml_frag.replace('<', '&lt;').replace('>', '&gt;')


class JRiverParser:

    def __init__(self, block=1, channels=('Subwoofer',)):
        self.__block = get_peq_key_name(block)
        self.__target_channels = ';'.join([str(JRIVER_CHANNELS.index(c)) for c in channels])

    def convert(self, dst, filt, **kwargs):
        from model.minidsp import flatten_filters
        flat_filts = flatten_filters(filt)
        config_txt = Path(dst).read_text()
        if len(flat_filts) > 0:
            logger.info(f"Copying {len(flat_filts)} to {dst}")
            # generate the xml formatted filters
            xml_filts = ''.join([filt_to_xml(convert_filter_to_mc_dsp(f, self.__target_channels)) for f in flat_filts])
            root, filt_element = extract_filters(config_txt, self.__block)
            # before_value, after_value, filt_section = extract_value_section(config_txt, self.__block)
            # separate the tokens, which are in (TOKEN) blocks, from within the Value element
            filt_fragments = [v + ')' for v in filt_element.text.split(')') if v]
            if len(filt_fragments) < 2:
                raise ValueError('Invalid input file - Unexpected <Value> format')
            # find the filter count and replace it with the new filter count
            filt_count = int(filt_fragments[1][1:-1].split(':')[1])
            new_filt_count = filt_count + len(flat_filts)
            filt_fragments[1] = f"({len(str(new_filt_count))}:{new_filt_count})"
            # append the new filters to any existing ones
            new_filt_section = ''.join(filt_fragments) + xml_filts
            # replace the value block in the original string
            filt_element.text = new_filt_section
            config_txt = et.tostring(root, encoding='UTF-8', xml_declaration=True).decode('utf-8')
        else:
            logger.warning(f"Nop for empty filter file {dst}")
        return config_txt, False

    @staticmethod
    def file_extension():
        return '.dsp'

    @staticmethod
    def newline():
        return '\r\n'


class Node:

    def __init__(self, id: str, filt: Optional[Filter], channel: Optional[str] = None):
        self.id = id
        self.filt = filt
        self.channel = channel
        if self.filt is None and self.channel is None:
            raise ValueError('Must have either filter or channel')
        self.edges: List[Node] = []

    def add(self, node):
        self.edges.append(node)

    def __repr__(self):
        detail = 'EMPTY'
        if self.filt:
            detail = str(self.filt)
        elif self.channel:
            detail = self.channel
        return f"{self.id} - {detail} -> {len(self.edges)} edges"


class GraphGenerator:

    def __init__(self, input_channels: List[str], output_channels: List[str], filts: List[Filter]):
        self.__filts = filts
        self.__output_channels = output_channels
        self.__input_channels = input_channels

    def generate(self, vertical: bool, selected_node: str = None) -> str:
        nodes_by_channel = self.__align_grid(self.__collapse_nodes(self.__link_nodes(self.__init_by_channel_grid())))
        gz, user_channel_clusters = self.__init_gz(nodes_by_channel, vertical, selected_node=selected_node)
        # add edges
        for channel, nodes in nodes_by_channel.items():
            gz += self.__append_edges(channel, nodes, user_channel_clusters)
        # calculate and add ranks
        gz += "\n"
        gz += self.__append_ranks(nodes_by_channel)
        gz += "}"
        return gz

    @staticmethod
    def __create_record(channels, prefix):
        return '|'.join([f"<{prefix}{c}> {c}" for c in channels if c not in SHORT_USER_CHANNELS])

    def __init_by_channel_grid(self):
        # create a channel/filter grid
        by_channel: Dict[str, List[Node]] = {c: [Node(f"IN:I{c}", None, c)] if c not in SHORT_USER_CHANNELS else [] for
                                             c in self.__output_channels}
        i = 0
        for idx, f in enumerate(self.__filts):
            if not isinstance(f, Divider):
                for k, v in by_channel.items():
                    channel_idx = get_channel_idx(k, short=True)
                    if f.is_mine(channel_idx):
                        if isinstance(f, Mix):
                            dst_channel = by_channel[get_channel_name(f.dst_idx, short=True)]
                            if len(dst_channel) < 2: # user channel or empty channel
                                dst_channel.append(Node(f"c{k}_f{i}", f))
                            v[-1].edges.append(dst_channel[-1])
                        else:
                            node = Node(f"c{k}_f{i}", f)
                            v.append(node)
                            i += 1
        # add output nodes
        for c, nodes in by_channel.items():
            if c not in SHORT_USER_CHANNELS:
                nodes.append(Node(f"OUT:O{c}", None, c))
        return by_channel

    @staticmethod
    def __append_ranks(nodes_by_channel):
        gz = ""
        ranks = []
        for channel, nodes in nodes_by_channel.items():
            for i, node in enumerate(nodes):
                if i <= len(ranks):
                    ranks.append([])
                if node.filt:
                    ranks[i].append(node.id)
        # for rank in ranks:
        #     if len(rank) > 0:
        #         gz += f"  {{rank = same; {'; '.join(rank)}}}"
        #         gz += "\n"
        return gz

    @staticmethod
    def __append_edges(channel, nodes, user_channel_clusters):
        output = ""
        for node in nodes:
            for edge in node.edges:
                output += f"  {node.id} -> {edge.id}"
                # if isinstance(edge.filt, Mix) and get_channel_name(edge.filt.src_idx, short=True) == channel:
                #     gz += f" [label=\"{edge.filt.mix_type}\"]"
                output += ";"
                output += "\n"
        if not nodes and channel not in SHORT_USER_CHANNELS:
            output += f"  IN:I{channel} -> OUT:O{channel};"
        if channel in SHORT_USER_CHANNELS:
            output = user_channel_clusters[channel] + output + "\n  }"
        output += "\n"
        return output

    @staticmethod
    def __create_io_record(name, definition):
        return f"  {name} [shape=record label=\"{definition}\"];"

    @staticmethod
    def __create_channel_nodes(channel, nodes, selected_node=None):
        to_append = ""
        indent = "    " if channel in SHORT_USER_CHANNELS else "  "
        for node in nodes:
            if node.filt:
                to_append += f"{indent}{node.id} [shape=box label=\"{node.filt.short_name}\""
                if selected_node and selected_node == node.id:
                    to_append += " style=filled fillcolor=lightgrey"
                to_append += "]\n"
        return to_append

    @staticmethod
    def __wrap_in_subcluster(name, nodes):
        output = f"  subgraph cluster_{name} {{"
        output += "\n"
        output += nodes
        output += f"    label = \"{name}\";"
        return output

    def __init_gz(self, by_channel, vertical, selected_node=None):
        gz = "digraph G {"
        gz += "\n"
        gz += f"  rankdir={'TB' if vertical else 'LR'};"
        gz += "\n"
        gz += self.__create_io_record('IN', self.__create_record(self.__input_channels, 'I'))
        gz += "\n"
        gz += self.__create_io_record('OUT', self.__create_record(self.__output_channels, 'O'))
        gz += "\n"
        user_channel_clusters = {}
        # add all nodes
        for c, nodes in by_channel.items():
            to_append = self.__create_channel_nodes(c, nodes, selected_node=selected_node)
            if c in SHORT_USER_CHANNELS:
                user_channel_clusters[c] = self.__wrap_in_subcluster(c, to_append)
            else:
                gz += to_append
        gz += "\n"
        return gz, user_channel_clusters

    @staticmethod
    def __link_nodes(by_channel):
        # link nodes into a chain
        for c, nodes in by_channel.items():
            upstream: Optional[Node] = None
            for node in nodes:
                if upstream:
                    upstream.add(node)
                upstream = node
        return by_channel

    @staticmethod
    def __collapse_nodes(by_channel):
        # TODO collapse contiguous Peak/LS/HS into a composite filter
        # TODO collapse contiguous Add nodes into a single node
        return by_channel

    @staticmethod
    def __align_grid(by_channel):
        # TODO ensure mix sections are aligned in the grid
        return by_channel


class JRiverFilterPipeline:

    def __init__(self, dsp: JRiverDSP):
        self.__dsp = dsp
        self.__in_c = self.__dsp.channel_names(short=True, output=False)
        self.__out_c = self.__dsp.channel_names(short=True, output=True)

    def as_dot(self, idx, vertical=True, selected_id=None) -> str:
        generator = GraphGenerator(self.__in_c, self.__out_c, self.__dsp.peq(idx))
        return generator.generate(vertical, selected_node=selected_id)

    def render(self, dot) -> bytes:
        from graphviz import Source
        return Source(dot, format='svg').pipe()


class JRiverFilterPipelineDialog(QDialog, Ui_jriverGraphDialog):

    def __init__(self, parent, input_channels: List[str], output_channels: List[str], peq: List[Filter]):
        super(JRiverFilterPipelineDialog, self).__init__(parent)
        self.setupUi(self)
        self.__selected_node = None
        self.__peq = peq
        self.__inputs = input_channels
        self.__outputs = output_channels
        self.svgView.signal.on_click.connect(self.__on_selected_node)
        self.__regen()
        self.direction.toggled.connect(self.__regen)
        self.source.textChanged.connect(self.__gen_svg)

    def __regen(self):
        dot = GraphGenerator(self.__inputs, self.__outputs, self.__peq).generate(self.direction.isChecked(),
                                                                                 selected_node=self.__selected_node)
        self.source.setPlainText(dot)
        self.__gen_svg()

    def __gen_svg(self):
        dot = self.source.toPlainText()
        from graphviz import Source
        self.__dot = Source(dot, format='svg')
        self.__svg = self.__dot.pipe()
        self.svgView.render_bytes(self.__svg)

    def __on_selected_node(self, node_name: str):
        self.__selected_node = node_name
        self.__regen()
