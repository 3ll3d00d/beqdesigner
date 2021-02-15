from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as et
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union

import qtawesome as qta
import time
from qtpy.QtCore import QPoint
from qtpy.QtGui import QColor, QPalette, QKeySequence
from qtpy.QtWidgets import QDialog, QFileDialog, QMenu, QAction, QListWidgetItem
from scipy.signal import unit_impulse

from model import iir
from model.iir import s_to_q, q_to_s
from model.limits import dBRangeCalculator, PhaseRangeCalculator
from model.log import to_millis
from model.magnitude import MagnitudeModel
from model.preferences import JRIVER_GEOMETRY, JRIVER_GRAPH_X_MIN, JRIVER_GRAPH_X_MAX, JRIVER_DSP_DIR, Preferences, \
    get_filter_colour
from model.signal import Signal
from model.xy import MagnitudeData
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


def short_to_long(short: str) -> str:
    return JRIVER_CHANNELS[JRIVER_SHORT_CHANNELS.index(short)]


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

    def __init__(self, parent, prefs: Preferences):
        super(JRiverDSPDialog, self).__init__(parent)
        self.__selected_node_names: List[str] = []
        self.__current_dot_txt = None
        self.prefs = prefs
        self.setupUi(self)
        self.limitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.fullRangeButton.setIcon(qta.icon('fa5s.expand'))
        self.subOnlyButton.setIcon(qta.icon('fa5s.compress'))
        self.findFilenameButton.setIcon(qta.icon('fa5s.folder-open'))
        self.showDotButton.setIcon(qta.icon('fa5s.info-circle'))
        self.showPhase.setIcon(qta.icon('mdi.cosine-wave'))
        self.pipelineView.signal.on_click.connect(self.__on_selected_node)
        self.pipelineView.signal.on_context.connect(self.__show_edit_menu)
        self.showDotButton.clicked.connect(self.__show_dot_dialog)
        self.direction.toggled.connect(self.__regen)
        self.viewSplitter.setSizes([100000, 100000])
        self.__dsp: Optional[JRiverDSP] = None
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, prefs,
                                                self.__get_data(), 'Filter', fill_primary=True,
                                                x_min_pref_key=JRIVER_GRAPH_X_MIN, x_max_pref_key=JRIVER_GRAPH_X_MAX,
                                                secondary_data_provider=self.__get_data('phase'),
                                                secondary_name='Phase', secondary_prefix='deg', fill_secondary=False,
                                                db_range_calc=dBRangeCalculator(30, expand=True),
                                                y2_range_calc=PhaseRangeCalculator(), show_y2_in_legend=False)
        self.__restore_geometry()

    def __show_edit_menu(self, node_name: str, pos: QPoint):
        menu = QMenu(self)
        act = QAction(f"&Insert after {node_name}", self)
        act.setShortcuts(QKeySequence.New)
        # act.setStatusTip(f"Insert Filter after {node_name}")
        # act.triggered.connect(self.__insert_node)
        menu.addAction(act)
        menu.exec(pos)

    def __show_dot_dialog(self):
        if self.__current_dot_txt:

            def on_change(txt):
                self.__current_dot_txt = txt
                self.__gen_svg()

            JRiverFilterPipelineDialog(self.__current_dot_txt, on_change, self).show()

    def __regen(self):
        self.__current_dot_txt = self.__dsp.as_dot(self.blockSelector.currentIndex(),
                                                   vertical=self.direction.isChecked(),
                                                   selected_nodes=self.__selected_node_names)
        self.__gen_svg()

    def __gen_svg(self):
        self.pipelineView.render_bytes(render_dot(self.__current_dot_txt))

    def __on_selected_node(self, node_name: str):
        if node_name in self.__selected_node_names:
            self.__selected_node_names.remove(node_name)
        else:
            if len(self.__selected_node_names) == 2:
                self.__selected_node_names.pop(0)
            self.__selected_node_names.append(node_name)
        self.__highlight_selected_nodes()
        self.__regen()

    def __highlight_selected_nodes(self):
        graph = self.__dsp.graph(self.blockSelector.currentIndex())
        filts = [graph.get_filter_at_node(n) for n in self.__selected_node_names if graph.get_filter_at_node(n) is not None]
        if filts:
            i = 0
            for f in self.__dsp.graph(self.blockSelector.currentIndex()).filters:
                if not isinstance(f, Divider):
                    item: QListWidgetItem = self.filterList.item(i)
                    if filts and f in filts:
                        filts.remove(f)
                        item.setSelected(True)
                    else:
                        item.setSelected(False)
                    i += 1
        else:
            self.filterList.clearSelection()

    def redraw(self):
        self.__magnitude_model.redraw()

    def __restore_geometry(self):
        ''' loads the saved window size '''
        geometry = self.prefs.get(JRIVER_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def find_dsp_file(self):
        dsp_dir = self.prefs.get(JRIVER_DSP_DIR)
        kwargs = {
            'caption': 'Select JRiver Media Centre DSP File',
            'filter': 'DSP (*.dsp)'
        }
        if dsp_dir is not None and len(dsp_dir) > 0 and os.path.exists(dsp_dir):
            kwargs['directory'] = dsp_dir
        selected = QFileDialog.getOpenFileName(parent=self, **kwargs)
        if selected is not None and len(selected[0]) > 0:
            try:
                self.channelList.clear()
                main_colour = QColor(QPalette().color(QPalette.Active, QPalette.Text)).name()
                highlight_colour = QColor(QPalette().color(QPalette.Active, QPalette.Highlight)).name()
                self.__dsp = JRiverDSP(selected[0], self.prefs, colours=(main_colour, highlight_colour))
                for i, n in enumerate(self.__dsp.channel_names(output=True)):
                    self.channelList.addItem(n)
                    item: QListWidgetItem = self.channelList.item(i)
                    item.setSelected(n not in USER_CHANNELS)
                self.filename.setText(os.path.basename(selected[0])[:-4])
                self.show_filters()
                self.prefs.set(JRIVER_DSP_DIR, os.path.dirname(selected[0]))
            except Exception as e:
                logger.exception(f"Unable to parse {selected[0]}")
                from model.catalogue import show_alert
                show_alert('Unable to load DSP file', f"Invalid file\n\n{e}")

    def show_filters(self):
        '''
        Displays the complete filter list for the selected PEQ block.
        '''
        if self.__dsp is not None:
            peq_idx = self.blockSelector.currentIndex()
            self.filterList.clear()
            self.__dsp.activate(peq_idx)
            for f in self.__dsp.active_graph.filters:
                if not isinstance(f, Divider):
                    self.filterList.addItem(str(f))
            self.__regen()
        self.redraw()

    def __get_data(self, mode='mag'):
        return lambda *args, **kwargs: self.get_curve_data(mode, *args, **kwargs)

    def get_curve_data(self, mode, reference=None):
        ''' preview of the filter to display on the chart '''
        result = []
        if mode == 'mag' or self.showPhase.isChecked():
            if self.__dsp:
                names = [n.text() for n in self.channelList.selectedItems()]
                for signal in self.__dsp.signals:
                    if short_to_long(signal.name) in names:
                        result.append(MagnitudeData(signal.name, None, *signal.avg,
                                                    colour=get_filter_colour(len(result))))
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

    def __init__(self, filename: str, prefs: Preferences, colours: Tuple[str, str] = None):
        self.__active_idx = 0
        self.__filename = filename
        self.__colours = colours
        self.__prefs = prefs
        start = time.time()
        config_txt = Path(self.__filename).read_text()
        peq_block_order = get_peq_block_order(config_txt)
        i, o = get_available_channels(config_txt)
        self.__input_channel_indexes = i
        self.__output_channel_indexes = o
        self.__graphs: List[FilterGraph] = []
        self.__signals: Dict[str, Signal] = {}
        for block in peq_block_order:
            out_names = self.channel_names(short=True, output=True)
            in_names = out_names if self.__graphs else self.channel_names(short=True, output=False)
            self.__graphs.append(FilterGraph(block, in_names, out_names, self.__parse_peq(config_txt, block)))
        end = time.time()
        logger.info(f"Parsed {filename} in {to_millis(start, end)}ms")

    def __init_signals(self) -> Dict[str, Signal]:
        names = [get_channel_name(c, short=True) for c in self.__output_channel_indexes]
        return {c: self.__default_signal(c, self.__prefs) for c in names}

    @property
    def signals(self) -> List[Signal]:
        return list(self.__signals.values()) if self.__signals else []

    @staticmethod
    def __default_signal(channel: str, prefs: Preferences):
        fs = 48000
        return Signal(channel, unit_impulse(fs*4, 'mid') * 23453.66, prefs, fs=fs)

    @property
    def graph_count(self) -> int:
        return len(self.__graphs)

    def graph(self, idx) -> FilterGraph:
        return self.__graphs[idx]

    def as_dot(self, idx, vertical=True, selected_nodes=None) -> str:
        renderer = GraphRenderer(self.__graphs[idx], colours=self.__colours)
        return renderer.generate(vertical, selected_nodes=selected_nodes)

    def channel_names(self, short=False, output=False):
        idxs = self.__output_channel_indexes if output else self.__input_channel_indexes
        return [get_channel_name(i, short=short) for i in idxs]

    @staticmethod
    def channel_name(i):
        return get_channel_name(i)

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

    def activate(self, active_idx: int):
        self.__active_idx = active_idx
        self.__generate_signals()

    def __generate_signals(self):
        start = time.time()
        signals: Dict[str, Signal] = self.__init_signals()
        for c, pipe in self.active_graph.filter_pipes_by_channel.items():
            while pipe is not None:
                signals[c] = pipe.op.apply(signals[c])
                pipe = pipe.next
        self.__signals = signals
        end = time.time()
        logger.info(f"Generated {len(signals)} signals in {to_millis(start, end)} ms")

    @property
    def active_graph(self):
        return self.__graphs[self.__active_idx]


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

    def get_filter(self) -> FilterOp:
        return NopFilterOp()

    def print_disabled(self):
        return '' if self.enabled else f" *** DISABLED ***"

    @property
    def short_name(self):
        return self.__short_name

    def short_desc(self):
        return self.short_name

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

    @property
    def freq(self) -> float:
        return self.__frequency

    @property
    def q(self) -> float:
        return self.__q

    @property
    def gain(self) -> float:
        return self.__gain

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

    def get_filter(self) -> FilterOp:
        sos = self.__create_iir(48000, self.__frequency, self.__q, self.__gain).get_sos()
        if sos:
            return SosFilterOp(sos)
        else:
            return NopFilterOp()

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

    def short_desc(self):
        return f"{self.short_name} {self.freq:g}Hz"


class HighPass(GainQFilter):

    def __init__(self, vals):
        super().__init__(vals, iir.SecondOrder_HighPass, 'HPF')

    def from_jriver_q(self, q: float, gain: float):
        return q / 2**0.5

    def to_jriver_q(self, q: float, gain: float):
        return q * 2**0.5

    def short_desc(self):
        return f"{self.short_name} {self.freq:g}Hz"


class Gain(ChannelFilter):

    def __init__(self, vals):
        super().__init__(vals, 'GAIN')
        self.__gain = float(vals['Gain'])

    @property
    def gain(self):
        return self.__gain

    def get_vals(self) -> Dict[str, str]:
        return {
            'Gain': f"{self.__gain:.7g}",
            **super().get_vals()
        }

    def __repr__(self):
        return f"Gain {self.__gain:+.7g} dB {self.print_channel_names()}{self.print_disabled()}"

    def get_filter(self) -> FilterOp:
        return GainFilterOp(self.__gain)

    def short_desc(self):
        return f"{self.__gain:+.7g} dB"


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

    @property
    def delay(self) -> float:
        return self.__delay

    def get_vals(self) -> Dict[str, str]:
        return {
            'Delay': f"{self.__delay:.7g}",
            **super().get_vals()
        }

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__delay:+.7g} ms {self.print_channel_names()}{self.print_disabled()}"

    def short_desc(self):
        return f"{self.__delay:+.7g}ms"

    def get_filter(self) -> FilterOp:
        return DelayFilterOp(self.__delay)


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

    def get_filter(self) -> FilterOp:
        sos = iir.LinkwitzTransform(48000, self.__fz, self.__qz, self.__fp, self.__qp).get_sos()
        if sos:
            return SosFilterOp(sos)
        else:
            return NopFilterOp()


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
        self.__mode = int(vals['Mode'])

    @property
    def src_idx(self):
        return int(self.__source)

    @property
    def dst_idx(self):
        return int(self.__destination)

    @property
    def mix_type(self) -> MixType:
        return MixType(int(self.__mode))

    def get_vals(self) -> Dict[str, str]:
        return {
            'Source': self.__source,
            'Destination': self.__destination,
            'Gain': f"{self.__gain:.7g}",
            'Mode': f"{self.__mode}"
        }

    def __repr__(self):
        mix_type = MixType(self.__mode)
        src_name = get_channel_name(int(self.__source), short=True)
        dst_name = get_channel_name(int(self.__destination), short=True)
        return f"{mix_type.name.capitalize()} {src_name} {mix_type.modifier} {dst_name} {self.__gain:+.7g} dB{self.print_disabled()}"

    def is_mine(self, idx):
        return self.src_idx == idx

    def short_desc(self):
        mix_type = MixType(self.__mode)
        if mix_type == MixType.MOVE or mix_type == MixType.MOVE:
            return f"{mix_type.name}\nto {get_channel_name(int(self.__destination), short=True)}"
        elif mix_type == MixType.SWAP:
            return f"{mix_type.name}\n{get_channel_name(int(self.__source), short=True)}-{get_channel_name(int(self.__destination), short=True)}"
        else:
            return super().short_desc()

    def get_filter(self) -> FilterOp:
        if self.mix_type == MixType.ADD:
            return AddFilterOp()
        elif self.mix_type == MixType.SUBTRACT:
            return SubtractFilterOp()
        else:
            return NopFilterOp()


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


def get_peq_block_order(config_txt):
    root = et.fromstring(config_txt)
    peq_blocks = []
    for e in root.findall('./Preset/Key[@Name]/Data/Name[.=\"Enabled\"]/../Value[.="1"]/../..'):
        if e.attrib['Name'] == 'Parametric Equalizer':
            peq_blocks.append(0)
        elif e.attrib['Name'] == 'Parametric Equalizer 2':
            peq_blocks.append(1)
    if not peq_blocks:
        raise ValueError(f"No Enabled Parametric Equalizers found in {config_txt}")
    if len(peq_blocks) > 1:
        order_elem = root.find('./Preset/Key[@Name="DSP Studio"]/Data/Name[.="Plugin Order"]/../Value')
        if order_elem is not None:
            block_order = [token for token in order_elem.text.split(')') if 'Parametric Equalizer' in token]
            if block_order:
                if block_order[0].endswith('Parametric Equalizer'):
                    return [0, 1]
                else:
                    return [1, 0]
    return peq_blocks


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
    :param block: 0 or 1.
    :return: the PEQ key name.
    '''
    if block == 0:
        return 'Parametric Equalizer'
    elif block == 1:
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

    def __init__(self, rank: int, name: str, filt: Optional[Filter], channel: str):
        self.name = name
        self.rank = rank
        self.filt = filt
        self.channel = channel
        self.visited = False
        if self.filt is None and self.channel is None:
            raise ValueError('Must have either filter or channel')
        self.__down_edges: List[Node] = []
        self.__up_edges: List[Node] = []

    def add(self, node: Node):
        if node not in self.downstream:
            self.downstream.append(node)
            node.upstream.append(self)

    @property
    def downstream(self) -> List[Node]:
        return self.__down_edges

    @property
    def upstream(self) -> List[Node]:
        return self.__up_edges

    def detach(self):
        '''
        Detaches this node from the upstream nodes.
        '''
        for u in self.upstream:
            u.downstream.remove(self)
        self.__up_edges = []

    def __repr__(self):
        return f"{self.name}{'' if self.__up_edges else ' - ROOT'} - {self.filt} -> {self.__down_edges}"

    @classmethod
    def swap(cls, n1: Node, n2: Node):
        '''
        Routes n1 to n2 downstream and vice versa.
        :param n1: first node.
        :param n2: second node.
        '''
        n2_target_upstream = [t for t in n1.upstream]
        n1_target_upstream = [t for t in n2.upstream]
        n1.detach()
        n2.detach()
        for to_attach in n1_target_upstream:
            to_attach.add(n1)
        for to_attach in n2_target_upstream:
            to_attach.add(n2)

    @classmethod
    def replace(cls, src: Node, dst: Node) -> List[Node]:
        '''
        Replaces the contents of dest with src.
        :param src: src.
        :param dst: dest.
        :returns detached downstream.
        '''
        # detach dst from its upstream node(s)
        dst.detach()
        # copy the downstream nodes from src and then remove them from src
        tmp_downstream = [d for d in src.downstream]
        src.downstream.clear()
        # attach the downstream nodes from dst to src
        for d in dst.downstream:
            d.detach()
            src.add(d)
        # remove those downstream nodes from dst
        dst.downstream.clear()
        # return any downstream nodes orphaned from the old src
        return tmp_downstream

    @classmethod
    def copy(cls, src: Node, dst: Node):
        '''
        Copies the contents of src to dst, same as replace but leaves the upstream intact on the src.
        :param src: src.
        :param dst: dest.
        '''
        # detach dst from its upstream node(s)
        dst.detach()
        # add dst a new downstream to src
        src.add(dst)


'''
    def edit_filter(self):
        filter_model = FilterModel(None, self.parent.prefs)
        filter_model.filter = self.as_filter()
        FilterDialog(self.parent.prefs, signal, filter_model, self.parent.redraw,
                     # selected_filter=signal.filter[selection.selectedRows()[0].row()]
                     parent=self.parent, x_lim=self.__x_lim_provider()).show()
'''


class FilterGraph:

    def __init__(self, stage: int, input_channels: List[str], output_channels: List[str], filts: List[Filter]):
        self.__stage = stage
        self.__filts = filts
        self.__output_channels = output_channels
        self.__input_channels = input_channels
        self.__filters_by_node_name: Dict[str, Filter] = {}
        self.__nodes_by_channel: Dict[str, Node] = self.__collapse(self.__link(self.__create_nodes()))
        self.__filter_pipes_by_channel: Dict[str, FilterPipe] = self.__generate_filter_paths()

    @property
    def filter_pipes_by_channel(self) -> Dict[str, FilterPipe]:
        return self.__filter_pipes_by_channel

    def get_filter_at_node(self, node_name: str) -> Optional[Filter]:
        return self.__filters_by_node_name.get(node_name, None)

    @property
    def filters(self):
        return self.__filts

    @property
    def nodes_by_channel(self) -> Dict[str, Node]:
        return self.__nodes_by_channel

    @property
    def input_channels(self):
        return self.__input_channels

    @property
    def output_channels(self):
        return self.__output_channels

    def __create_nodes(self) -> Dict[str, List[Node]]:
        '''
        transforms each filter into a node on each channel the filter affects.
        :return: nodes by channel name.
        '''
        # create a channel/filter grid
        by_channel: Dict[str, List[Node]] = {c: [Node(0, f"IN:{c}", None, c)] if c not in SHORT_USER_CHANNELS else []
                                             for c in self.__output_channels}
        i = 1
        for idx, f in enumerate(self.__filts):
            if not isinstance(f, Divider) and f.enabled:
                for channel_name, nodes in by_channel.items():
                    channel_idx = get_channel_idx(channel_name, short=True)
                    if f.is_mine(channel_idx):
                        # mix is added as a node to both channels
                        if isinstance(f, Mix):
                            dst_channel_name = get_channel_name(f.dst_idx, short=True)
                            by_channel[dst_channel_name].append(self.__make_node(i, dst_channel_name, f))
                            nodes.append(self.__make_node(i, channel_name, f))
                        else:
                            nodes.append(self.__make_node(i, channel_name, f))
                i += 1
        # add output nodes
        for c, nodes in by_channel.items():
            if c not in SHORT_USER_CHANNELS:
                nodes.append(Node(i * 100, f"OUT:{c}", None, c))
        return by_channel

    def __make_node(self, phase: int, channel_name: str, filt: Filter):
        node = Node(phase * 100, f"{channel_name}_{phase}00_{filt.short_name}", filt, channel_name)
        if node.name in self.__filters_by_node_name:
            print(f"Dupe!!! {node.name}")
        self.__filters_by_node_name[node.name] = filt
        return node

    def __link(self, by_channel: Dict[str, List[Node]]) -> Dict[str, Node]:
        '''
        Assembles edges between nodes. For all filters except Mix, this is just a link to the preceding. node.
        Copy, move and swap mixes change this relationship.
        :param by_channel: the nodes by channel
        :return: the linked nodes by channel.
        '''
        return self.__remix(self.__initialise_graph(by_channel))

    @staticmethod
    def __initialise_graph(by_channel):
        input_by_channel: Dict[str, Node] = {}
        for c, nodes in by_channel.items():
            upstream: Optional[Node] = None
            for node in nodes:
                if c not in input_by_channel:
                    input_by_channel[c] = node
                if upstream:
                    upstream.add(node)
                upstream = node
        return input_by_channel

    def __remix(self, by_channel: Dict[str, Node], orphaned_nodes: Dict[str, List[Node]] = None) -> Dict[str, Node]:
        if orphaned_nodes is None:
            orphaned_nodes = defaultdict(list)
        for c, node in by_channel.items():
            self.__remix_node(node, by_channel, orphaned_nodes)
        self.__remix_orphans(by_channel, orphaned_nodes)
        bad_nodes = [n for n in self.__collect_all_nodes(by_channel) if not self.__is_linked(n)]
        if bad_nodes:
            logger.warning(f"Found {len(bad_nodes)} badly linked nodes")
        return by_channel

    @staticmethod
    def __is_linked(node: Node):
        for u in node.upstream:
            if node not in u.downstream:
                logger.debug(f"Node not cross linked with upstream {u.name} - {node.name}")
                return False
        return True

    def __remix_orphans(self, by_channel, orphaned_nodes):
        while True:
            if orphaned_nodes:
                c_to_remove = []
                for c, orphans in orphaned_nodes.items():
                    new_root = orphans.pop(0)
                    by_channel[f"{c}:{new_root.rank}"] = new_root
                    if not orphans:
                        c_to_remove.append(c)
                orphaned_nodes = {k: v for k, v in orphaned_nodes.items() if k not in c_to_remove}
                self.__remix(by_channel, orphaned_nodes=orphaned_nodes)
            else:
                break

    def __remix_node(self, node: Node, by_channel: Dict[str, Node], orphaned_nodes: Dict[str, List[Node]]) -> None:
        downstream = [d for d in node.downstream]
        if not node.visited:
            node.visited = True
            channel_name = self.__extract_channel_name(node)
            f = node.filt
            if isinstance(f, Mix) and f.enabled:
                if f.mix_type == MixType.SWAP:
                    downstream = self.__swap_node(by_channel, channel_name, downstream, f, node, orphaned_nodes)
                elif f.mix_type == MixType.MOVE or f.mix_type == MixType.COPY:
                    downstream = self.__copy_or_replace_node(by_channel, channel_name, downstream, f, node,
                                                             orphaned_nodes)
                elif f.mix_type == MixType.ADD or f.mix_type == MixType.SUBTRACT:
                    self.__add_or_subtract_node(by_channel, f, node, orphaned_nodes)
        for d in downstream:
            self.__remix_node(d, by_channel, orphaned_nodes)

    @staticmethod
    def __extract_channel_name(node):
        if ':' not in node.channel:
            channel_name = node.channel
        else:
            channel_name = node.channel[0:node.channel.index(':')]
        return channel_name

    def __add_or_subtract_node(self, by_channel: Dict[str, Node], f: Mix, node: Node,
                               orphaned_nodes: Dict[str, List[Node]]):
        if f.is_mine(get_channel_idx(node.channel, short=True)):
            dst_channel_name = get_channel_name(f.dst_idx, short=True)
            try:
                dst_node = self.__find_owning_node_in_channel(by_channel[dst_channel_name], f, dst_channel_name)
                node.add(dst_node)
            except ValueError:
                dst_node = self.__find_owning_node_in_orphans(dst_channel_name, f, orphaned_nodes)
                if dst_node:
                    dst_node.visited = True
                    node.add(dst_node)
                else:
                    logger.debug(f"No match for {f} in {dst_channel_name}, presumed orphaned")
                    node.visited = False

    def __find_owning_node_in_orphans(self, dst_channel_name, f, orphaned_nodes):
        return next((n for n in orphaned_nodes.get(dst_channel_name, [])
                     if self.__owns_filter(n, f, dst_channel_name)), None)

    def __copy_or_replace_node(self, by_channel: Dict[str, Node], channel_name: str, downstream: List[Node], f: Mix,
                               node: Node, orphaned_nodes: Dict[str, List[Node]]) -> List[Node]:
        src_channel_name = get_channel_name(f.src_idx, short=True)
        if src_channel_name == channel_name and node.channel == channel_name:
            dst_channel_name = get_channel_name(f.dst_idx, short=True)
            try:
                dst_node = self.__find_owning_node_in_channel(by_channel[dst_channel_name], f, dst_channel_name)
                if f.mix_type == MixType.COPY:
                    Node.copy(node, dst_node)
                else:
                    new_downstream = Node.replace(node, dst_node)
                    if new_downstream:
                        if len(new_downstream) > 1:
                            txt = f"{channel_name} - {new_downstream}"
                            raise ValueError(f"Unexpected multiple downstream nodes on replace in channel {txt}")
                        if not new_downstream[0].name.startswith('OUT:'):
                            orphaned_nodes[channel_name].append(new_downstream[0])
                downstream = node.downstream
            except ValueError:
                logger.debug(f"No match for {f} in {dst_channel_name}, presumed orphaned")
                node.visited = False
        else:
            node.visited = False
        return downstream

    def __swap_node(self, by_channel: Dict[str, Node], channel_name: str, downstream: List[Node], f: Mix,
                    node: Node, orphaned_nodes: Dict[str, List[Node]]) -> List[Node]:
        src_channel_name = get_channel_name(f.src_idx, short=True)
        if src_channel_name == channel_name:
            dst_channel_name = get_channel_name(f.dst_idx, short=True)
            swap_channel_name = src_channel_name if channel_name == dst_channel_name else dst_channel_name
            try:
                swap_node = self.__find_owning_node_in_channel(by_channel[swap_channel_name], f, swap_channel_name)
            except ValueError:
                swap_node = self.__find_owning_node_in_orphans(swap_channel_name, f, orphaned_nodes)
            if swap_node:
                Node.swap(node, swap_node)
                downstream = node.downstream
            else:
                logger.debug(f"No match for {f} in {swap_channel_name}, presumed orphaned")
                node.visited = False
        else:
            node.visited = False
        return downstream

    @staticmethod
    def __find_owning_node_in_channel(node: Node, match: Filter, owning_channel_name: str) -> Node:
        if FilterGraph.__owns_filter(node, match, owning_channel_name):
            return node
        for d in node.downstream:
            return FilterGraph.__find_owning_node_in_channel(d, match, owning_channel_name)
        raise ValueError(f"No match for '{match}' in '{node}'")

    @staticmethod
    def __owns_filter(node: Node, match: Filter, owning_channel_name: str) -> bool:
        return node.filt and node.filt == match and node.channel == owning_channel_name

    def __collapse(self, by_channel: Dict[str, Node]) -> Dict[str, Node]:
        # TODO collapse contiguous Peak/LS/HS into a composite filter
        # TODO eliminate swap/move/copy nodes
        return by_channel

    def __generate_filter_paths(self) -> Dict[str, FilterPipe]:
        output_nodes = self.__get_output_nodes()
        filter_pipes: Dict[str, FilterPipe] = {}
        for channel_name, output_node in output_nodes.items():
            parent = self.__get_parent(output_node)
            filters: List[FilterOp] = []
            while parent is not None:
                if parent.filt:
                    filters.append(parent.filt.get_filter())
                parent = self.__get_parent(parent)
            filter_pipes[channel_name] = coalesce_ops(filters[::-1])
        return filter_pipes

    @staticmethod
    def __get_parent(node: Node) -> Optional[Node]:
        if node.upstream:
            if len(node.upstream) == 1:
                return node.upstream[0]
            else:
                if len(node.upstream) == 2:
                    u1, u2 = node.upstream
                    if FilterGraph.__is_inbound_mix(node, u1):
                        return u1
                    elif FilterGraph.__is_inbound_mix(node, u2):
                        return u2
                    else:
                        raise ValueError(f"Unexpected filter types upstream of {node} - {[n.name for n in node.upstream]}")
                else:
                    raise ValueError(f">2 upstream found! {node.name} -> {[n.name for n in node.upstream]}")
        else:
            return None

    @staticmethod
    def __is_inbound_mix(node: Node, upstream: Node):
        return isinstance(upstream.filt, Mix) \
                and (upstream.filt.mix_type == MixType.ADD or upstream.filt.mix_type == MixType.SUBTRACT) \
                and upstream.channel != node.channel

    def __get_output_nodes(self) -> Dict[str, Node]:
        output = {}
        for node in self.__collect_all_nodes(self.__nodes_by_channel):
            if node.name.startswith('OUT:'):
                output[node.channel] = node
        return output

    @staticmethod
    def __collect_all_nodes(nodes_by_channel: Dict[str, Node]) -> List[Node]:
        nodes = []
        for root in nodes_by_channel.values():
            collect_nodes(root, nodes)
        return nodes


class GraphRenderer:

    def __init__(self, graph: FilterGraph, colours: Tuple[str, str] = None):
        self.__graph = graph
        self.__colours = colours

    def generate(self, vertical: bool, selected_nodes: Optional[List[str]] = None) -> str:
        gz = self.__init_gz(vertical)
        node_defs, user_channel_clusters = self.__add_node_definitions(selected_nodes=selected_nodes)
        if node_defs:
            gz += node_defs
            gz += "\n"
        edges = self.__generate_edges()
        if edges:
            gz += self.__generate_edge_definitions(edges, user_channel_clusters)
        ranks = self.__generate_ranks()
        if ranks:
            gz += '\n'
            gz += ranks
        gz += "}"
        return gz

    @staticmethod
    def __generate_edge_definitions(edges: Dict[str, Tuple[str, str]], user_channel_clusters: Dict[str, str]) -> str:
        gz = ''
        output = defaultdict(str)
        for channel_edge in edges.values():
            output[channel_edge[0]] += f"{channel_edge[1]}\n"
        for c, v in output.items():
            if c in user_channel_clusters:
                gz += user_channel_clusters[c]
            gz += v
            gz += "\n"
        return gz

    def __generate_edges(self) -> Dict[str, Tuple[str, str]]:
        edges: Dict[str, Tuple[str, str]] = {}
        for channel, node in self.__graph.nodes_by_channel.items():
            self.__locate_edges(channel, node, edges)
        return edges

    @staticmethod
    def __create_record(channels):
        return '|'.join([f"<{c}> {c}" for c in channels if c not in SHORT_USER_CHANNELS])

    @staticmethod
    def __locate_edges(channel: str, start_node: Node, visited_edges: Dict[str, Tuple[str, str]]):
        for end_node in start_node.downstream:
            edge_txt = f"{start_node.name} -> {end_node.name}"
            if edge_txt not in visited_edges:
                indent = '    ' if end_node.channel in SHORT_USER_CHANNELS else '  '
                if end_node.name.startswith('OUT'):
                    target_channel = end_node.channel
                elif end_node.channel in SHORT_USER_CHANNELS:
                    target_channel = end_node.channel
                elif start_node.channel in SHORT_USER_CHANNELS:
                    target_channel = start_node.channel
                else:
                    target_channel = start_node.channel
                visited_edges[edge_txt] = (target_channel, f"{indent}{edge_txt};")
            GraphRenderer.__locate_edges(channel, end_node, visited_edges)

    @staticmethod
    def __create_io_record(name, definition):
        return f"  {name} [shape=record label=\"{definition}\"];"

    def __create_channel_nodes(self, channel, nodes, selected_nodes=None):
        to_append = ""
        label_prefix = f"{channel}\n" if channel in SHORT_USER_CHANNELS else ''
        for node in nodes:
            if node.filt:
                to_append += f"  {node.name} [label=\"{label_prefix}{node.filt.short_desc()}\""
                if selected_nodes and next((n for n in selected_nodes if n == node.name), None) is not None:
                    fill_colour = f"\"{self.__colours[1]}\"" if self.__colours else 'lightgrey'
                    to_append += f" style=filled fillcolor={fill_colour}"
                to_append += "]\n"
        return to_append

    def __init_gz(self, vertical) -> str:
        gz = "digraph G {\n"
        gz += f"  rankdir={'TB' if vertical else 'LR'};\n"
        gz += "  node [\n"
        gz += "    shape=\"box\"\n"
        if self.__colours:
            gz += f"    color=\"{self.__colours[0]}\"\n"
            gz += f"    fontcolor=\"{self.__colours[0]}\"\n"
        gz += "  ];\n"
        if self.__colours:
            gz += "  edge [\n"
            gz += f"    color=\"{self.__colours[0]}\"\n"
            gz += "  ];\n"
            gz += "  graph [\n"
            gz += f"    color=\"{self.__colours[0]}\"\n"
            gz += f"    fontcolor=\"{self.__colours[0]}\"\n"
            gz += "  ];"
            gz += "\n"
        gz += "\n"
        gz += self.__create_io_record('IN', self.__create_record(self.__graph.input_channels))
        gz += "\n"
        gz += self.__create_io_record('OUT', self.__create_record(self.__graph.output_channels))
        gz += "\n"
        gz += "\n"
        return gz

    def __add_node_definitions(self, selected_nodes: Optional[List[str]] = None) -> Tuple[str, Dict[str, str]]:
        user_channel_clusters = {}
        gz = ''
        # add all nodes
        for c, node in self.__graph.nodes_by_channel.items():
            to_append = self.__create_channel_nodes(c, collect_nodes(node, []), selected_nodes=selected_nodes)
            if to_append:
                if c in SHORT_USER_CHANNELS:
                    user_channel_clusters[c] = to_append
                else:
                    gz += to_append
                    gz += "\n"
        return gz, user_channel_clusters

    def __generate_ranks(self):
        nodes = []
        ranks = defaultdict(list)
        for root in self.__graph.nodes_by_channel.values():
            collect_nodes(root, nodes)
        for node in nodes:
            if node.filt and node not in ranks[node.rank]:
                ranks[node.rank].append(node)
        gz = ''
        for nodes in ranks.values():
            gz += f"  {{rank = same; {'; '.join([n.name for n in nodes])}}}"
            gz += "\n"
        return gz


class FilterOp(ABC):

    @abstractmethod
    def apply(self, input_signal: Signal) -> Signal:
        pass

    def __repr__(self):
        return self.__class__.__name__


class FilterPipe:

    def __init__(self, op: FilterOp):
        self.op = op
        self.next: Optional[FilterPipe] = None

    def __repr__(self):
        return f"{self.op} -> {self.next.op if self.next else 'END'}"


def coalesce_ops(ops: List[FilterOp]) -> Optional[FilterPipe]:
    root: Optional[FilterPipe] = None
    tmp: Optional[FilterPipe] = None
    last_sos: Optional[SosFilterOp] = None
    for op in ops:
        if isinstance(op, SosFilterOp):
            if last_sos:
                last_sos.extend(op)
            else:
                last_sos = op
        else:
            if last_sos:
                if root is None:
                    tmp = FilterPipe(last_sos)
                    root = tmp
                else:
                    tmp.next = FilterPipe(last_sos)
                    tmp = tmp.next
                last_sos = None
            if root is None:
                tmp = FilterPipe(op)
                root = tmp
            else:
                tmp.next = FilterPipe(op)
                tmp = tmp.next
    if last_sos:
        t = FilterPipe(last_sos)
        if root is None:
            root = t
        else:
            tmp.next = t
    return root


class NopFilterOp(FilterOp):

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal


class BranchPointOp(FilterOp):

    def __init__(self):
        self.__branch: Optional[Signal] = None

    def apply(self, input_signal: Signal) -> Signal:
        self.__branch = input_signal
        return input_signal

    @property
    def branch(self) -> Optional[Signal]:
        return self.__branch


class SosFilterOp(FilterOp):

    def __init__(self, sos: List[List[float]]):
        self.__sos = sos

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.sosfilter(self.__sos)

    def extend(self, more_sos: SosFilterOp):
        self.__sos += more_sos.__sos


class DelayFilterOp(FilterOp):

    def __init__(self, delay_millis: float, fs: int = 48000):
        self.__shift_samples = int((delay_millis / 1000) / (1.0 / fs))

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.shift(self.__shift_samples)


class AddFilterOp(FilterOp):

    def __init__(self):
        self.__other: Optional[Signal] = None

    def apply(self, input_signal: Signal) -> Signal:
        # return input_signal.add(samples)
        logger.error(f"TODO bypassing ADD for {input_signal}")
        return input_signal


class SubtractFilterOp(FilterOp):

    def __init__(self):
        self.__other: Optional[Signal] = None

    def apply(self, input_signal: Signal) -> Signal:
        # return input_signal.subtract(samples)
        logger.error(f"TODO bypassing ADD for {input_signal}")
        return input_signal


class GainFilterOp(FilterOp):
    def __init__(self, gain_db: float):
        self.__gain_db = gain_db

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.adjust_gain(10 ** (self.__gain_db / 20.0))


def collect_nodes(node: Node, arr: List[Node]) -> List[Node]:
    if node not in arr:
        arr.append(node)
    for d in node.downstream:
        collect_nodes(d, arr)
    return arr


class JRiverFilterPipelineDialog(QDialog, Ui_jriverGraphDialog):

    def __init__(self, dot: str, on_change, parent):
        super(JRiverFilterPipelineDialog, self).__init__(parent)
        self.setupUi(self)
        self.source.setPlainText(dot)
        if on_change:
            self.source.textChanged.connect(lambda: on_change(self.source.toPlainText()))


def render_dot(txt):
    from graphviz import Source
    return Source(txt, format='svg').pipe()
