import json
import logging
from collections import defaultdict
from typing import Callable, List, Optional, Any, Dict, Tuple, Type

import math
import qtawesome as qta
from PyQt5.QtCore import QModelIndex, QVariant, QAbstractItemModel, QSize
from PyQt5.QtGui import QShowEvent
from PyQt5.QtWidgets import QHeaderView, QDialogButtonBox, QListWidgetItem, QListWidget, QPushButton, QAbstractItemView
from qtpy.QtCore import Qt, QTimer, QAbstractTableModel
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QGridLayout, QLabel, QDoubleSpinBox, QComboBox, QSpinBox, QDialog, QSpacerItem, \
    QSizePolicy, QAbstractSpinBox, QFrame, QWidget, QHBoxLayout, QVBoxLayout, QCheckBox
from scipy.signal import unit_impulse

from model.iir import FilterType, ComplexHighPass, ComplexLowPass, CompleteFilter
from model.jriver import XOFilter, convert_filter_to_mc_dsp, get_channel_idx, OutputFormat, SingleFilter, Filter, Delay, \
    create_peq, Gain, MixType, Mix, user_channel_indexes, CompoundRoutingFilter, HighPass, LowPass, CustomPassFilter, \
    Polarity
from model.limits import dBRangeCalculator, PhaseRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import Preferences, XO_GRAPH_X_MIN, XO_GRAPH_X_MAX, get_filter_colour, XO_GEOMETRY
from model.signal import Signal
from model.xy import MagnitudeData
from ui.channel_matrix import Ui_channelMatrixDialog
from ui.delegates import CheckBoxDelegate
from ui.group_channels import Ui_groupChannelsDialog
from ui.jriver_channel_select import Ui_jriverChannelSelectDialog
from ui.xo import Ui_xoDialog

LFE_ADJUST_KEY = 'l'
ROUTING_KEY = 'r'
EDITORS_KEY = 'e'
EDITOR_NAME_KEY = 'n'
UNDERLYING_KEY = 'u'
WAYS_KEY = 'w'
SYM_KEY = 's'
SW_KEY = 'x'

logger = logging.getLogger('xo')


class Matrix:

    def __init__(self, inputs: Dict[str, int], outputs: List[str]):
        self.__inputs = inputs
        self.__outputs = outputs
        self.__row_keys = self.__make_row_keys()
        # input channel -> way -> output channel -> enabled
        self.__ways: Dict[str, Dict[int, Dict[str, bool]]] = self.__make_default_ways()

    def get_routes(self) -> List[Tuple[int, int, int, Optional[MixType]]]:
        '''
        :return: the routes defined by this matrix.
        '''
        routes: List[Tuple[int, int, int, Optional[MixType]]] = []
        for input_channel, v1 in self.__ways.items():
            for way, v2 in v1.items():
                for output_channel, routed in v2.items():
                    if routed:
                        mt = None if input_channel == output_channel else MixType.MOVE
                        routes.append((get_channel_idx(input_channel), way, get_channel_idx(output_channel), mt))
        return self.__reorder_routes(self.__sort_routes(self.__fixup_mix_types(routes)))

    @staticmethod
    def __fixup_mix_types(routes: List[Tuple[int, int, int, Optional[MixType]]]) -> List[Tuple[int, int, int, Optional[MixType]]]:
        '''
        :param routes: the raw routes (all using MOVE).
        :return: the routes with mixtype corrected to use add or copy as necessary.
        '''
        inputs_received = defaultdict(int)
        outputs_routed = defaultdict(int)
        for i, w, o, mt in routes:
            inputs_received[o] = inputs_received[o] + 1
            outputs_routed[(i, w)] = outputs_routed[(i, w)] + 1
        routes_with_mix = []
        for i, w, o, mt in routes:
            actual_mt = mt
            if inputs_received[o] > 1 and i != o:
                actual_mt = MixType.ADD
            elif outputs_routed[(i, w)] > 0 and mt:
                actual_mt = MixType.COPY
            if actual_mt:
                routes_with_mix.append((i, w, o, actual_mt))
        return routes_with_mix

    @staticmethod
    def __reorder_routes(routes: List[Tuple[int, int, int, Optional[MixType]]]) -> List[Tuple[int, int, int, Optional[MixType]]]:
        '''
        Reorders routing to ensure inputs are not overridden with outputs. Attempts to break circular dependencies using user channels if possible.
        :param routes: the routes.
        :return: the reordered routes.
        '''
        ordered_routes: List[Tuple[int, int, int, Optional[MixType], int]] = []
        u1_channel_idx = user_channel_indexes()[0]
        for r_input, r_way, r_output, r_mixtype in routes:
            def repack() -> Tuple[int, int, int, Optional[MixType], int]:
                return r_input, r_way, r_output, r_mixtype, -1
            if not ordered_routes or not r_mixtype or r_mixtype == MixType.ADD:
                ordered_routes.append(repack())
            else:
                insert_at = -1
                for idx, o_r in enumerate(ordered_routes):
                    # if a route wants to write to this input, make sure this route comes first
                    if o_r[2] == r_input:
                        insert_at = idx
                        break
                if insert_at == -1:
                    ordered_routes.append(repack())
                else:
                    broke_circular = False
                    for o_r in ordered_routes[insert_at:]:
                        if o_r[0] == r_output:
                            # break a circular dependency using user channel, i.e. if a later route has an input matching this output
                            broke_circular = True
                            inserted_route = ordered_routes[insert_at]
                            # only insert the copy to the user channel if we have not already done so
                            if inserted_route[0] != r_input or inserted_route[2] != u1_channel_idx:
                                ordered_routes.insert(insert_at, (r_input, r_way, u1_channel_idx, MixType.COPY, r_output))
                            # now append the copy from the user channel to the actual target channel
                            ordered_routes.insert(insert_at + 2, (u1_channel_idx, r_way, r_output, MixType.COPY, -1))
                            break
                    if not broke_circular:
                        if insert_at > 0:
                            inserted = ordered_routes[insert_at - 1]
                            if inserted[-1] > -1 and inserted[0] == r_input:
                                insert_at -= 1
                        ordered_routes.insert(insert_at, repack())
        u1_in_use_for = -1
        failed = False
        output: List[Tuple[int, int, int, Optional[MixType]]] = []
        for i, w, o, mt, target in ordered_routes:
            if target > -1:
                if u1_in_use_for == -1:
                    u1_in_use_for = target
                else:
                    if target != u1_in_use_for:
                        failed = True
            if i == u1_channel_idx:
                if o != u1_in_use_for:
                    failed = True
                else:
                    u1_in_use_for = -1
            output.append((i, w, o, mt))
        if failed:
            raise ValueError(f'Unresolvable circular dependencies found in {ordered_routes}')
        return output

    def __sort_routes(self, routes: List[Tuple[int, int, int, Optional[MixType]]]) -> List[Tuple[int, int, int, Optional[MixType]]]:
        '''
        applies a topological sort to the input-output routes, primarily resolves issues with bass management additions.
        :param routes: the channel routes.
        :return: the sorted routes.
        '''
        edges = defaultdict(set)
        for i, _, o, _ in routes:
            edges[f"O{o}"].add(f"I{i}")
        for i in self.__inputs.keys():
            if i in self.__outputs:
                edges[f"O{get_channel_idx(i)}"].add(f"I{get_channel_idx(i)}")
        sorted_channels = []
        for d in Matrix.do_sort(edges):
            sorted_channels.extend(sorted(d))
        sorted_channel_mapping = {int(v[1:]): i for i, v in enumerate(sorted_channels)}
        sorted_routes = sorted(routes, key=lambda x: (sorted_channel_mapping[x[0]], x[1]))
        return sorted_routes

    @staticmethod
    def do_sort(to_sort):
        edges = to_sort.copy()
        from functools import reduce
        extra_items_in_deps = reduce(set.union, edges.values()) - set(edges.keys())
        edges.update({item: set() for item in extra_items_in_deps})
        while True:
            ordered = set(item for item, dep in edges.items() if len(dep) == 0)
            if not ordered:
                break
            yield ordered
            edges = {item: (dep - ordered) for item, dep in edges.items() if item not in ordered}
        if len(edges) != 0:
            formatted = ', '.join([f'{k}:{v}' for k, v in sorted(edges.items())])
            raise ValueError(f'Circular dependencies found in {formatted}')

    def __make_default_ways(self) -> Dict[str, Dict[int, Dict[str, bool]]]:
        return {i: {w: {c: False for c in self.__outputs} for w in range(ways)} for i, ways in self.__inputs.items()}

    def __make_row_keys(self) -> List[Tuple[str, int]]:
        return [(c, w) for c, ways in self.__inputs.items() for w in range(ways)]

    @property
    def rows(self):
        return len(self.__row_keys)

    def row_name(self, idx: int):
        c, w = self.__row_keys[idx]
        suffix = '' if self.__inputs[c] < 2 else f" - {w+1}"
        return f"{c}{suffix}"

    @property
    def columns(self):
        return len(self.__outputs)

    def column_name(self, idx: int) -> str:
        return self.__outputs[idx]

    def toggle(self, row: int, column: int) -> str:
        c, w = self.__row_keys[row]
        output_channel = self.__outputs[column]
        now_enabled = not self.__ways[c][w][output_channel]
        self.__ways[c][w][output_channel] = now_enabled
        error_msg = None
        if now_enabled:
            try:
                self.get_routes()
            except ValueError as e:
                logger.exception(f"Unable to activate route from {c}{w} to {output_channel}: circular dependency")
                error_msg = 'Unable to route, circular dependency'
                self.__ways[c][w][output_channel] = False
        return error_msg

    def enable(self, channel: str, way: int, output: str):
        self.__ways[channel][way][output] = True

    def is_routed(self, row: int, column: int) -> bool:
        c, w = self.__row_keys[row]
        return self.__ways[c][w][self.__outputs[column]]

    def __repr__(self):
        return f"{self.__ways}"

    def clone(self):
        clone = Matrix(self.__inputs, self.__outputs)
        clone.__copy_matrix_values(self.__ways)
        return clone

    def __copy_matrix_values(self, source: Dict[str, Dict[int, Dict[str, bool]]]):
        for k1, v1 in source.items():
            for k2, v2 in v1.items():
                for k3, v3 in v2.items():
                    self.__ways[k1][k2][k3] = v3

    def resize(self, channel: str, ways: int):
        old_len = self.__inputs[channel]
        if ways < old_len:
            self.__inputs[channel] = ways
            self.__row_keys = self.__make_row_keys()
            for i in range(ways, old_len):
                del self.__ways[channel][i]
        elif ways > old_len:
            self.__inputs[channel] = ways
            self.__row_keys = self.__make_row_keys()
            old_ways = self.__ways
            self.__ways = self.__make_default_ways()
            self.__copy_matrix_values(old_ways)

    def get_mapping(self) -> Dict[str, Dict[int, str]]:
        '''
        :return: channel mapping as input channel -> way -> output channel
        '''
        mapping = defaultdict(dict)
        for input_channel, v1 in self.__ways.items():
            for way, v2 in v1.items():
                for output_channel, routed in v2.items():
                    if routed:
                        prefix = f"{mapping[input_channel][way]};" if way in mapping[input_channel] else ''
                        mapping[input_channel][way] = f"{prefix}{get_channel_idx(output_channel)}"
        return mapping

    def encode(self) -> List[str]:
        '''
        :return: currently stored routings in encoded form.
        '''
        routings = []
        for input_channel, v1 in self.__ways.items():
            for way, v2 in v1.items():
                for output_channel, routed in v2.items():
                    if routed:
                        routings.append(f"{input_channel}/{way}/{output_channel}")
        return routings

    def decode(self, routings: List[str]) -> None:
        '''
        Reloads the routing generated by encode.
        :param routings: the routings.
        '''
        for input_channel, v1 in self.__ways.items():
            for way, v2 in v1.items():
                for output_channel in v2.keys():
                    v2[output_channel] = False
        for r in routings:
            i, w, o = r.split('/')
            self.__ways[i][int(w)][o] = True


class XODialog(QDialog, Ui_xoDialog):

    def __init__(self, parent, prefs: Preferences, input_channels: List[str], output_channels: List[str],
                 output_format: OutputFormat, on_save: Callable[[CompoundRoutingFilter], None],
                 existing: CompoundRoutingFilter = None, **kwargs):
        super(XODialog, self).__init__(parent)
        self.__on_save = on_save
        self.__output_format = output_format
        self.__channel_groups = {c: [c] for c in input_channels}
        self.__output_channels = output_channels
        self.__sw_channels = self.__calculate_sw_channels()
        self.__mag_update_timer = QTimer(self)
        self.__mag_update_timer.setSingleShot(True)
        self.prefs = prefs
        self.setupUi(self)
        self.showMatrixButton.setIcon(qta.icon('fa5s.route'))
        self.setSWChannelsButton.clicked.connect(self.__show_sw_channel_selector_dialog)
        self.linkChannelsButton.clicked.connect(self.__show_group_channels_dialog)
        self.__matrix = None
        self.__editors = [
            ChannelEditor(self.channelsFrame, name, channels, output_format,
                          any(c in self.__sw_channels for c in channels), self.__trigger_redraw,
                          self.__on_output_channel_count_change)
            for name, channels in self.__channel_groups.items()
        ]
        last_widget = None
        for e in self.__editors:
            self.channelsLayout.addWidget(e.widget)
            self.setTabOrder(last_widget if last_widget else self.lfeAdjust, e.widget)
            last_widget = e.widget
        self.__matrix: Matrix = self.__create_matrix()
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, prefs,
                                                self.__get_data(), 'Filter', fill_primary=False,
                                                x_min_pref_key=XO_GRAPH_X_MIN, x_max_pref_key=XO_GRAPH_X_MAX,
                                                secondary_data_provider=self.__get_data('phase'),
                                                secondary_name='Phase', secondary_prefix='deg', fill_secondary=False,
                                                db_range_calc=dBRangeCalculator(60),
                                                y2_range_calc=PhaseRangeCalculator(), show_y2_in_legend=False, **kwargs)
        self.limitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.limitsButton.setToolTip('Set graph axis limits')
        self.showPhase.setIcon(qta.icon('mdi.cosine-wave'))
        self.showPhase.toggled.connect(self.__trigger_redraw)
        self.showPhase.setToolTip('Display phase response')
        if output_format.lfe_channels > 0:
            self.lfeAdjust.setEnabled(True)
            self.lfeAdjustLabel.setEnabled(True)
        self.__mag_update_timer.timeout.connect(self.__magnitude_model.redraw)
        self.__existing = existing
        self.linkChannelsButton.setFocus()
        self.__restore_geometry()

    def showEvent(self, event: QShowEvent):
        '''
        Loads any existing filter after the dialog is shown otherwise isVisible calls will always return false.
        '''
        event.accept()
        if self.__existing:
            self.__load_filter()
            self.__magnitude_model.redraw()
            self.__existing = None

    def __load_filter(self):
        metadata = json.loads(self.__existing.metadata())
        if LFE_ADJUST_KEY in metadata:
            self.lfeAdjust.setValue(metadata[LFE_ADJUST_KEY])
        if EDITORS_KEY in metadata:
            groups = {e[EDITOR_NAME_KEY]: e[UNDERLYING_KEY] for e in metadata[EDITORS_KEY]}
            self.__reconfigure_channel_groups(groups)
            for e in metadata[EDITORS_KEY]:
                editor: ChannelEditor = next((c for c in self.__editors if c.name == e[EDITOR_NAME_KEY]))
                editor.ways = e[WAYS_KEY]
                editor.symmetric = e[SYM_KEY]
            for f in self.__existing.filters:
                if isinstance(f, XOFilter):
                    match = next(e for e in self.__editors
                                 if e.name in groups.keys() and f.input_channel in e.underlying_channels)
                    match.load_filter(f)
        if SW_KEY in metadata:
            self.__sw_channels = metadata[SW_KEY]
        if ROUTING_KEY in metadata:
            self.__matrix.decode(metadata[ROUTING_KEY])

    def __calculate_sw_channels(self) -> List[str]:
        return [] if self.__output_format.lfe_channels == 0 else ['SW']

    def accepted(self):
        super().accepted()

    def __create_matrix(self):
        matrix = Matrix({c: len(e) for e in self.__editors for c in e.underlying_channels}, self.__output_channels)
        self.__initialise_routing(matrix)
        return matrix

    def __initialise_routing(self, matrix: Matrix):
        channel_ways = [(c, w) for e in self.__editors for c in e.underlying_channels for w in range(len(e))]
        if self.__output_format.lfe_channels == 0:
            for i, cw in enumerate(channel_ways):
                c, w = cw
                o = self.__output_channels[i % len(self.__output_channels)]
                matrix.enable(c, w, o)
        else:
            for c, w in channel_ways:
                if w == 0:
                    for sw in self.__sw_channels:
                        matrix.enable(c, w, sw)
                else:
                    matrix.enable(c, w, c)

    def __show_group_channels_dialog(self):
        GroupChannelsDialog(self, {e.name: e.underlying_channels for e in self.__editors if e.widget.isVisible()},
                            self.__reconfigure_channel_groups).exec()

    def __show_sw_channel_selector_dialog(self) -> None:
        '''
        Allows user to specify multiple sw output channels.
        '''
        def on_save(sw_channels):
            if sw_channels != self.__sw_channels:
                self.__sw_channels = sw_channels
                self.__matrix = self.__create_matrix()

        SWChannelSelectorDialog(self, self.__output_channels, self.__sw_channels, on_save).exec()

    def __reconfigure_channel_groups(self, grouped_channels: Dict[str, List[str]]):
        '''
        Reconfigures the UI to show the new channel groups.
        :param grouped_channels: the grouped channels.
        '''
        old_matrix = self.__matrix
        self.__matrix = None
        old_groups = self.__channel_groups
        self.__channel_groups = grouped_channels
        for name, channels in grouped_channels.items():
            matching_editor: ChannelEditor = next((e for e in self.__editors if e.name == name), None)
            if matching_editor:
                if matching_editor.underlying_channels != channels:
                    matching_editor.underlying_channels = channels
                    old_matrix = None
            else:
                new_editor = ChannelEditor(self.channelsFrame, name, channels, self.__output_format,
                                           any(c in self.__sw_channels for c in channels),
                                           self.__trigger_redraw, self.__on_output_channel_count_change)
                self.__editors.append(new_editor)
                self.channelsLayout.addWidget(new_editor.widget)
                old_matrix = None
        for name, channels in old_groups.items():
            if name not in grouped_channels.keys():
                matching_editor: ChannelEditor = next((e for e in self.__editors if e.name == name), None)
                if matching_editor:
                    matching_editor.hide()
                    old_matrix = None
        self.__matrix = self.__create_matrix() if not old_matrix else old_matrix

    def __on_output_channel_count_change(self, channel: str, ways: int):
        if self.__matrix:
            self.__matrix.resize(channel, ways)

    def show_matrix(self):
        MatrixDialog(self, self.__matrix, self.__set_matrix).show()

    def __set_matrix(self, matrix: Matrix):
        self.__matrix = matrix

    def accept(self):
        self.__on_save(self.__make_filters())
        self.prefs.set(XO_GEOMETRY, self.saveGeometry())
        super().accept()

    def reject(self):
        self.prefs.set(XO_GEOMETRY, self.saveGeometry())
        super().reject()

    def __make_filters(self) -> CompoundRoutingFilter:
        routes = self.__matrix.get_routes()
        routing_filters = [self.__convert_to_filter(i, mt, o) for i, w, o, mt in routes if mt]
        channel_mapping: Dict[str, Dict[int, str]] = self.__matrix.get_mapping()
        xo_filters = []
        for e in self.__editors:
            for c in e.underlying_channels:
                xo_filters.extend(e.get_xo_filters(c, channel_mapping[c]))
        meta = {
            EDITORS_KEY: [
                {EDITOR_NAME_KEY: e.name, UNDERLYING_KEY: e.underlying_channels, WAYS_KEY: len(e), SYM_KEY: e.symmetric}
                for e in self.__editors if e.widget.isVisible()
            ],
            ROUTING_KEY: self.__matrix.encode(),
        }
        if self.__sw_channels:
            meta[SW_KEY] = self.__sw_channels
        if self.lfeAdjust.isEnabled():
            meta[LFE_ADJUST_KEY] = self.lfeAdjust.value()
        # TODO add gain for LFE channel adjustments
        return CompoundRoutingFilter(json.dumps(meta), routing_filters, xo_filters)

    @staticmethod
    def __convert_to_filter(i: int, mt: MixType, o: int) -> Filter:
        vals = Mix.default_values()
        # TODO handle gain for bass management
        vals['Source'] = str(i)
        vals['Destination'] = str(o)
        vals['Mode'] = str(mt.value)
        return Mix(vals)

    def show_limits(self):
        ''' shows the limits dialog for the filter chart. '''
        self.__magnitude_model.show_limits()

    def __trigger_redraw(self):
        if not self.__mag_update_timer.isActive():
            self.__mag_update_timer.start(20)

    def __get_data(self, mode='mag'):
        return lambda *args, **kwargs: self.get_curve_data(mode, *args, **kwargs)

    def get_curve_data(self, mode, reference=None):
        ''' preview of the filter to display on the chart '''
        result = []
        extra = 0
        for editor in self.__editors:
            if editor.show_response:
                if mode == 'mag':
                    mag_data = editor.impulses
                    summed = None
                    for i in mag_data:
                        result.append(MagnitudeData(i.name, None, *i.avg, colour=get_filter_colour(len(result)),
                                                    linestyle='-'))
                        if summed is None:
                            summed = i
                        else:
                            summed = summed.add(i.samples)
                    if summed:
                        result.append(MagnitudeData(editor.name, None, *summed.avg,
                                                    colour=get_filter_colour(len(result)),
                                                    linestyle='-' if mode == 'mag' else '--'))
                elif mode == 'phase' and self.showPhase.isChecked():
                    for pr_mag in editor.phase_responses:
                        pr_mag.colour = get_filter_colour(len(result) + extra)
                        pr_mag.linestyle = '--'
                        result.append(pr_mag)
                extra += 1
        return result

    def __restore_geometry(self):
        ''' loads the saved window size '''
        geometry = self.prefs.get(XO_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)


class ChannelEditor:

    def __init__(self, channels_frame: QWidget, name: str, underlying_channels: List[str],
                 output_format: OutputFormat, is_sw_channel: bool,  on_filter_change: Callable[[], None],
                 on_way_count_change: Callable[[str, int], None]):
        self.__on_change = on_filter_change
        self.__is_sw_channel = is_sw_channel
        self.__output_format = output_format
        self.__name = name
        self.__underlying_channels = underlying_channels
        self.__editors: List[WayEditor] = []
        self.__frame = QFrame(channels_frame)
        self.__frame.setFrameShape(QFrame.StyledPanel)
        self.__frame.setFrameShadow(QFrame.Raised)
        self.__layout = QVBoxLayout(self.__frame)
        # header
        self.__header_layout = QHBoxLayout()
        header_font = QFont()
        header_font.setBold(True)
        header_font.setItalic(True)
        header_font.setWeight(75)
        self.__channel_name_label = QLabel(self.__frame)
        self.__channel_name_label.setFont(header_font)
        if len(underlying_channels) == 1:
            group = ''
            suffix = ''
        else:
            group = ' Group'
            suffix = f" [{','.join(underlying_channels)}]"
        self.__channel_name_label.setText(f"Channel{group}: {name}{suffix}")
        self.__ways = QSpinBox()
        self.__ways.setMinimum(1)
        self.__ways.setMaximum(8)
        self.__ways.setSingleStep(1)
        self.__ways.setSuffix(' way(s)')
        self.__show_response = QCheckBox(self.__frame)
        self.__show_response.setText('Preview?')
        self.__show_response.toggled.connect(self.__on_change)
        self.__symmetric = QCheckBox(self.__frame)
        self.__symmetric.setText('Symmetric XO?')
        self.__symmetric.setChecked(True)
        self.__is_symmetric = lambda: self.__symmetric.isChecked()
        self.__header_layout.addWidget(self.__channel_name_label)
        self.__header_layout.addWidget(self.__ways)
        self.__header_layout.addWidget(self.__show_response)
        self.__header_layout.addWidget(self.__symmetric)
        self.__header_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.__layout.addLayout(self.__header_layout)
        self.__way_frame = QFrame(self.__frame)
        self.__way_layout = QHBoxLayout(self.__way_frame)
        self.__way_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.__layout.addWidget(self.__way_frame)
        self.__ways.valueChanged.connect(self.__update_editors)
        self.__ways.valueChanged.connect(self.__on_change)
        self.__ways.valueChanged.connect(lambda v: self.__propagate_way_count_change(on_way_count_change, v))
        self.__frame.setTabOrder(self.__ways, self.__show_response)
        self.__frame.setTabOrder(self.__show_response, self.__symmetric)
        if self.__is_sw_channel:
            self.__ways.hide()
            self.__update_editors()
        else:
            self.__ways.setValue(self.__calculate_ways())

    def __propagate_way_count_change(self, func: Callable[[str, int], None], count: int):
        for c in self.__underlying_channels:
            func(c, count)

    def __calculate_ways(self) -> int:
        of = self.__output_format
        if of.lfe_channels == 0:
            if of.input_channels == of.output_channels:
                return 1
            else:
                return int(of.output_channels / of.input_channels)
        else:
            return int(of.output_channels / of.input_channels) + 1

    def __len__(self):
        return self.__ways.value()

    @property
    def name(self) -> str:
        return self.__name

    @property
    def ways(self) -> int:
        return self.__ways.value()

    @ways.setter
    def ways(self, ways: int):
        self.__ways.setValue(ways)

    @property
    def underlying_channels(self) -> List[str]:
        return self.__underlying_channels

    @underlying_channels.setter
    def underlying_channels(self, underlying_channels: List[str]):
        self.__underlying_channels = underlying_channels

    @property
    def symmetric(self) -> bool:
        return self.__symmetric.isChecked()

    @symmetric.setter
    def symmetric(self, symmetric: bool) -> None:
        self.__symmetric.setChecked(symmetric)

    @property
    def show_response(self) -> bool:
        return self.__show_response.isChecked()

    @property
    def impulses(self) -> List[Signal]:
        return [i for i in [way.impulse for way in self.__editors] if i]

    @property
    def phase_responses(self) -> List[MagnitudeData]:
        return [way.phase_response for way in self.__editors if way.phase_response]

    def __update_editors(self):
        for i in range(self.__ways.value()):
            if i >= len(self.__editors):
                self.__create_editor(i)
                if i > 1:
                    self.__frame.setTabOrder(self.__editors[i - 1].widget, self.__editors[i].widget)
            else:
                self.__editors[i].show()
        if self.__ways.value() < len(self.__editors):
            for i in range(self.__ways.value(), len(self.__editors)):
                self.__editors[i].hide()
        self.__symmetric.setVisible(self.__ways.value() > 1)

    def __create_editor(self, i: int):
        editor = WayEditor(self.__way_frame, self.__name, i, self.__on_change, self.__propagate_symmetric_filter,
                           self.__output_format.lfe_channels > 0, self.__is_sw_channel)
        self.__editors.append(editor)
        self.__way_layout.insertWidget(i, editor.widget)

    def __propagate_symmetric_filter(self, way: int, filter_type: str, freq: float, order: int):
        if self.__is_symmetric():
            if way+1 < len(self.__editors):
                self.__editors[way+1].set_high_pass(filter_type, freq, order)

    def show(self) -> None:
        self.__frame.show()

    def hide(self) -> None:
        self.__frame.hide()

    @property
    def widget(self) -> QWidget:
        return self.__frame

    def get_xo_filters(self, input_channel: str, output_channel_by_way: Dict[int, str]) -> List[XOFilter]:
        mc_filters: List[XOFilter] = []
        for way, e in enumerate(self.__editors):
            if way in output_channel_by_way:
                xo_filters = e.get_filters(output_channel_by_way[way])
                if xo_filters:
                    mc_filters.append(XOFilter(input_channel, way, xo_filters))
        return mc_filters

    def load_filter(self, f: XOFilter):
        '''
        Loads the filter into the specified way.
        :param f: the filter.
        '''
        self.__editors[f.way].load_filter(f)


class WayEditor:

    def __init__(self, parent: QWidget, channel: str, way: int, on_change: Callable[[], None],
                 propagate_low_pass_change: Callable[[int, str, float, int], None], allow_sw: bool, is_sw: bool):
        self.__way = way
        self.__propagate_low_pass = propagate_low_pass_change
        self.__channel = channel
        self.__pass_filters = None
        self.__notify_parent = on_change
        self.__frame = QFrame(parent)
        self.__frame.setFrameShape(QFrame.StyledPanel)
        self.__frame.setFrameShadow(QFrame.Raised)
        self.__layout = QGridLayout(self.__frame)
        self.__is_active = True
        # header
        checkable = False
        if way == 0 and allow_sw and not is_sw:
            self.__header = QPushButton(self.__frame)
            self.__header.setCheckable(True)
            self.__header.toggled.connect(self.__on_sw_change)
            checkable = True
        else:
            self.__header = QLabel(self.__frame)
            self.__header.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.__header.setFont(font)
        self.__header.setText('' if is_sw else f"Way {way + 1}")
        # controls
        self.__invert = QCheckBox(self.__frame)
        self.__invert.setText('Invert')
        self.__invert.toggled.connect(self.__on_value_change)
        self.__gain_label = QLabel(self.__frame)
        self.__gain_label.setText('Gain')
        self.__gain = QDoubleSpinBox(self.__frame)
        self.__gain.setMinimum(-60.0)
        self.__gain.setMaximum(60.0)
        self.__gain.setDecimals(2)
        self.__gain.setSuffix(' dB')
        self.__gain.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.__gain.valueChanged.connect(self.__on_value_change)
        self.__delay_label = QLabel(self.__frame)
        self.__delay_label.setText('Delay')
        self.__delay = QDoubleSpinBox(self.__frame)
        self.__delay.setMinimum(-2000)
        self.__delay.setMaximum(2000.0)
        self.__delay.setDecimals(2)
        self.__delay.setSuffix(' ms')
        self.__delay.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.__delay.valueChanged.connect(self.__on_value_change)
        # low pass
        self.__lp_label = QLabel(self.__frame)
        self.__lp_label.setText('Low Pass')
        self.__lp_freq = self.__make_freq_field(self.__frame, self.__on_lp_change)
        self.__lp_filter_type = self.__make_filter_type_selector(self.__frame, self.__on_lp_filter_type_change)
        self.__lp_order = self.__make_order_field(self.__frame, self.__on_lp_change)
        # high pass
        self.__hp_freq = self.__make_freq_field(self.__frame, self.__on_value_change)
        self.__hp_filter_type = self.__make_filter_type_selector(self.__frame, self.__on_hp_filter_type_change)
        self.__hp_order = self.__make_order_field(self.__frame, self.__on_value_change)
        self.__hp_label = QLabel(self.__frame)
        self.__hp_label.setText('High Pass')
        # layout
        row = 0
        self.__layout.addWidget(self.__invert, row, 0, 1, 1)
        self.__layout.addWidget(self.__header, row, 1, 1, -1)
        row += 1
        self.__layout.addWidget(self.__hp_label, row, 0, 1, 1)
        self.__layout.addWidget(self.__hp_filter_type, row, 1, 1, 1)
        self.__layout.addWidget(self.__hp_freq, row, 2, 1, 1)
        self.__layout.addWidget(self.__hp_order, row, 3, 1, 1)
        row += 1
        self.__layout.addWidget(self.__lp_label, row, 0, 1, 1)
        self.__layout.addWidget(self.__lp_filter_type, row, 1, 1, 1)
        self.__layout.addWidget(self.__lp_freq, row, 2, 1, 1)
        self.__layout.addWidget(self.__lp_order, row, 3, 1, 1)
        row += 1
        self.__layout.addItem(QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.__frame.setTabOrder(self.__invert, self.__hp_filter_type)
        self.__frame.setTabOrder(self.__hp_filter_type, self.__hp_freq)
        self.__frame.setTabOrder(self.__hp_freq, self.__hp_order)
        self.__frame.setTabOrder(self.__hp_order, self.__lp_filter_type)
        self.__frame.setTabOrder(self.__lp_filter_type, self.__lp_freq)
        self.__frame.setTabOrder(self.__lp_freq, self.__lp_order)
        self.__frame.setTabOrder(self.__lp_order, self.__gain)
        self.__frame.setTabOrder(self.__gain, self.__delay)
        self.__controls_layout = QHBoxLayout()
        self.__controls_layout.addWidget(self.__gain_label)
        self.__controls_layout.addWidget(self.__gain)
        self.__controls_layout.addWidget(self.__delay_label)
        self.__controls_layout.addWidget(self.__delay)
        self.__controls_layout.setStretch(1, 1)
        self.__controls_layout.setStretch(3, 1)
        self.__layout.addLayout(self.__controls_layout, row, 0, 1, -1)

        if self.__way == 0 and checkable:
            self.__header.setChecked(True)

    def __on_lp_change(self):
        self.__propagate_low_pass(self.__way, self.__lp_filter_type.currentText(), self.__lp_freq.value(),
                                  self.__lp_order.value())
        self.__on_value_change()

    def __on_sw_change(self, on: bool):
        if on:
            self.__header.setText(f"Way {self.__way + 1} -> Sub")
            self.__header.setToolTip('Select to send to a dedicated low frequency channel')
        else:
            self.__header.setText(f"Way {self.__way + 1}")
            self.__header.setToolTip('Select to send to a combined low frequency channel')
        self.__is_active = not on
        self.__invert.setVisible(self.__is_active)
        self.__hp_label.setVisible(self.__is_active)
        self.__hp_freq.setVisible(self.__is_active)
        self.__hp_order.setVisible(self.__is_active)
        self.__hp_filter_type.setVisible(self.__is_active)
        self.__lp_label.setVisible(self.__is_active)
        self.__lp_freq.setVisible(self.__is_active)
        self.__lp_order.setVisible(self.__is_active)
        self.__lp_filter_type.setVisible(self.__is_active)
        self.__gain.setVisible(self.__is_active)
        self.__gain_label.setVisible(self.__is_active)
        self.__delay.setVisible(self.__is_active)
        self.__delay_label.setVisible(self.__is_active)
        self.__on_value_change()

    @property
    def inverted(self) -> bool:
        return self.__invert.isChecked()

    def __on_value_change(self):
        self.__refresh_filters()
        self.__notify_parent()

    def __on_lp_filter_type_change(self):
        self.__change_pass_field_state(self.__lp_filter_type, self.__lp_order, self.__lp_freq)
        self.__on_lp_change()

    def __on_hp_filter_type_change(self):
        self.__change_pass_field_state(self.__hp_filter_type, self.__hp_order, self.__hp_freq)
        self.__on_value_change()

    @staticmethod
    def __change_pass_field_state(selector: QComboBox, order: QSpinBox, freq: QDoubleSpinBox) -> None:
        if selector.currentIndex() == 0:
            order.setEnabled(False)
            freq.setEnabled(False)
        else:
            order.setEnabled(True)
            freq.setEnabled(True)
            if selector.currentText() == FilterType.LINKWITZ_RILEY.display_name:
                if order.value() % 2 != 0:
                    order.setValue(max(2, order.value() - 1))
                order.setSingleStep(2)
                order.setMinimum(2)
            else:
                order.setSingleStep(1)
                order.setMinimum(1)

    @staticmethod
    def __make_filter_type_selector(parent: QWidget, on_change: Callable[[], None]):
        combo = QComboBox(parent)
        combo.addItem('')
        for ft in FilterType:
            combo.addItem(ft.display_name)
        combo.currentIndexChanged.connect(on_change)
        return combo

    @staticmethod
    def __make_order_field(parent: QWidget, on_change: Callable[[], None]) -> QSpinBox:
        widget = QSpinBox(parent)
        widget.setMinimum(1)
        widget.setMaximum(24)
        widget.setValue(2)
        widget.setEnabled(False)
        widget.valueChanged.connect(on_change)
        return widget

    @staticmethod
    def __make_freq_field(parent: QWidget, on_change: Callable[[], None]) -> QDoubleSpinBox:
        widget = QDoubleSpinBox(parent)
        widget.setMinimum(1)
        widget.setDecimals(1)
        widget.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        widget.setMaximum(24000)
        widget.setEnabled(False)
        widget.setSuffix(' Hz')
        widget.valueChanged.connect(on_change)
        return widget

    def show(self) -> None:
        self.__frame.show()

    def hide(self) -> None:
        self.__frame.hide()

    @property
    def widget(self) -> QWidget:
        return self.__frame

    @property
    def impulse(self) -> Optional[Signal]:
        if self.__is_active:
            fs = 48000
            signal = Signal(f"{self.__channel}{self.__way + 1}", unit_impulse(fs*4, 'mid') * 23453.66, fs=fs)
            f = self.__pass_filters
            if f:
                signal = signal.sosfilter(f.get_sos())
            if self.inverted:
                signal = signal.invert()
            if not math.isclose(self.__gain.value(), 0.0):
                signal = signal.adjust_gain(10 ** (self.__gain.value() / 20.0))
            if not math.isclose(self.__delay.value(), 0.0):
                signal = signal.shift(int((self.__delay.value() / 1000) / (1.0 / 48000)))
            return signal
        return None

    @property
    def phase_response(self) -> Optional[MagnitudeData]:
        if self.__pass_filters and self.__is_active:
            return self.__pass_filters.get_transfer_function().get_phase()
        return None

    def get_filters(self, channel_indexes: str) -> List[Filter]:
        if self.__invert.isVisible() and self.__pass_filters and self.__is_active:
            mc_filters = [convert_filter_to_mc_dsp(f, channel_indexes) for f in self.__pass_filters]
            if not math.isclose(self.__delay.value(), 0.0):
                mc_filters.append(self.__make_mc_filter(self.__delay, Delay, channel_indexes, 'Delay'))
            if not math.isclose(self.__gain.value(), 0.0):
                mc_filters.append(self.__make_mc_filter(self.__gain, Gain, channel_indexes, 'Gain'))
            if self.__invert.isChecked():
                mc_filters.append(Polarity({'Enabled': '1', 'Type': Polarity.TYPE, 'Channels': channel_indexes}))
            return mc_filters
        return []

    @staticmethod
    def __make_mc_filter(widget: QDoubleSpinBox, filt_type: Type[SingleFilter], channel_indexes: str,
                         key: str) -> Filter:
        return create_peq({
            **filt_type.default_values(),
            key: f"{widget.value():.7g}",
            'Channels': channel_indexes
        })

    def __refresh_filters(self) -> None:
        self.__pass_filters = self.__create_pass_filters()

    def __create_pass_filters(self) -> Optional[CompleteFilter]:
        f = []
        if self.__is_active:
            if self.__lp_filter_type.currentIndex() > 0:
                f.append(ComplexLowPass(FilterType.value_of(self.__lp_filter_type.currentText()),
                                        self.__lp_order.value(), 48000, self.__lp_freq.value()))
            if self.__hp_filter_type.currentIndex() > 0:
                f.append(ComplexHighPass(FilterType.value_of(self.__hp_filter_type.currentText()),
                                         self.__hp_order.value(), 48000, self.__hp_freq.value()))
        return CompleteFilter(fs=48000, filters=f) if f else None

    def set_high_pass(self, filter_type: str, freq: float, order: int):
        self.__hp_order.setValue(order)
        self.__hp_freq.setValue(freq)
        self.__hp_filter_type.setCurrentText(filter_type)

    def load_filter(self, xo: XOFilter):
        '''
        Loads the filter.
        :param xo: the filter.
        '''
        for f in xo.filters:
            if isinstance(f, Delay):
                self.__delay.setValue(f.delay)
            elif isinstance(f, Gain):
                self.__gain.setValue(f.gain)
            elif isinstance(f, Polarity):
                self.__invert.setChecked(True)
            elif isinstance(f, HighPass):
                self.__hp_filter_type.setCurrentText(FilterType.BUTTERWORTH.display_name)
                self.__hp_order.setValue(f.order)
                self.__hp_freq.setValue(f.freq)
            elif isinstance(f, LowPass):
                self.__lp_filter_type.setCurrentText(FilterType.BUTTERWORTH.display_name)
                self.__lp_order.setValue(f.order)
                self.__lp_freq.setValue(f.freq)
            elif isinstance(f, CustomPassFilter):
                ef = f.get_editable_filter()
                if isinstance(ef, ComplexHighPass):
                    self.__hp_freq.setValue(ef.freq)
                    self.__hp_filter_type.setCurrentText(ef.type.display_name)
                    self.__hp_order.setValue(ef.order)
                elif isinstance(ef, ComplexLowPass):
                    self.__lp_freq.setValue(ef.freq)
                    self.__lp_filter_type.setCurrentText(ef.type.display_name)
                    self.__lp_order.setValue(ef.order)
        self.__refresh_filters()


class MatrixTableModel(QAbstractTableModel):

    def __init__(self, matrix: Matrix, on_toggle: Callable[[str], None]):
        super().__init__()
        self.__matrix = matrix
        self.__on_toggle = on_toggle

    def rowCount(self, parent=None, *args, **kwargs):
        return self.__matrix.rows

    def columnCount(self, parent=None, *args, **kwargs):
        return self.__matrix.columns + 1

    def flags(self, idx):
        flags = super().flags(idx)
        if idx.column() > 0:
            flags |= Qt.ItemIsEditable
        return flags

    def data(self, index: QModelIndex, role: int = ...) -> Any:
        if not index.isValid() or role != Qt.DisplayRole:
            return QVariant()
        elif index.column() == 0:
             return QVariant(self.__matrix.row_name(index.row()))
        else:
            return QVariant(self.__matrix.is_routed(index.row(), index.column() - 1))

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> Any:
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                return QVariant()
            else:
                return QVariant(self.__matrix.column_name(section - 1))
        return QVariant()

    def delegate_to_checkbox(self, view):
        delegate = CheckBoxDelegate()
        for x in range(self.__matrix.columns):
            view.setItemDelegateForColumn(x + 1, delegate)

    def toggle(self, idx: QModelIndex):
        '''
        Toggles the model at the given index.
        :param idx: the index.
        '''
        error_msg = self.__matrix.toggle(idx.row(), idx.column() - 1)
        self.__on_toggle(error_msg)
        if not error_msg:
            self.dataChanged.emit(QModelIndex(), QModelIndex())


class MatrixDialog(QDialog, Ui_channelMatrixDialog):

    def __init__(self, parent: QWidget, matrix: Matrix, on_save: Callable[[Matrix], None]):
        super(MatrixDialog, self).__init__(parent)
        self.setupUi(self)
        self.errorMessage.setStyleSheet('color: red')
        self.__on_save = on_save
        self.__matrix = matrix.clone()
        self.__table_model = MatrixTableModel(self.__matrix, self.__set_error)
        self.matrix.setModel(self.__table_model)
        self.matrix.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.__table_model.delegate_to_checkbox(self.matrix)

    def __set_error(self, msg: str):
        if msg:
            self.errorMessage.setText(msg)
            self.errorMessageIcon.setPixmap(qta.icon('fa5s.exclamation-triangle', color='red').pixmap(QSize(32, 32)))
        else:
            self.errorMessage.clear()
            self.errorMessageIcon.clear()

    def accept(self):
        self.__on_save(self.__matrix)
        super().accept()


class GroupChannelsDialog(QDialog, Ui_groupChannelsDialog):

    GROUP_CHANNELS_ROLE = Qt.UserRole + 1

    def __init__(self, parent: QWidget, channels: Dict[str, List[str]],
                 on_save: Callable[[Dict[str, List[str]]], None]):
        super(GroupChannelsDialog, self).__init__(parent)
        self.setupUi(self)
        self.__on_save = on_save
        for c, u in channels.items():
            if len(u) == 1:
                self.channels.addItem(c)
            else:
                item = QListWidgetItem(c)
                item.setData(self.GROUP_CHANNELS_ROLE, u)
                self.channelGroups.addItem(item)
        self.groupName.setEnabled(False)
        self.channels.itemSelectionChanged.connect(self.__enable_add_button)
        self.groupName.textChanged.connect(self.__enable_add_button)
        output_model: QAbstractItemModel = self.channelGroups.model()
        output_model.rowsInserted.connect(self.__enable_save_button)
        output_model.rowsRemoved.connect(self.__enable_save_button)
        self.channelGroups.itemSelectionChanged.connect(self.__enable_remove_button)
        self.addGroupButton.clicked.connect(self.__add_group)
        self.addGroupButton.setIcon(qta.icon('fa5s.plus'))
        self.deleteGroupButton.setIcon(qta.icon('fa5s.minus'))
        self.deleteGroupButton.clicked.connect(self.__remove_group)
        self.__enable_save_button()

    def __enable_remove_button(self):
        self.deleteGroupButton.setEnabled(len(self.channelGroups.selectedItems()) > 0)

    def __enable_add_button(self):
        some_selected = len(self.channels.selectedIndexes()) > 0
        self.groupName.setEnabled(some_selected)
        self.addGroupButton.setEnabled(some_selected and len(self.groupName.text()) > 0)

    def __enable_save_button(self):
        self.buttonBox.button(QDialogButtonBox.Save).setEnabled(self.channelGroups.count() > 0)

    def __add_group(self):
        item: QListWidgetItem = QListWidgetItem(self.groupName.text())
        item.setData(self.GROUP_CHANNELS_ROLE, [i.text() for i in self.channels.selectedItems()])
        self.channelGroups.addItem(item)
        self.groupName.clear()
        self.__remove_selected(self.channels)

    def __remove_group(self):
        for i in self.channelGroups.selectedItems():
            for c in i.data(self.GROUP_CHANNELS_ROLE):
                self.channels.addItem(c)
        self.__remove_selected(self.channelGroups)

    @staticmethod
    def __remove_selected(widget: QListWidget):
        for i in widget.selectedItems():
            widget.takeItem(widget.indexFromItem(i).row())

    def accept(self):
        groups = {self.channelGroups.item(i).text(): self.channelGroups.item(i).data(self.GROUP_CHANNELS_ROLE) for i in
                  range(self.channelGroups.count())}
        individuals = {self.channels.item(i).text(): [self.channels.item(i).text()] for i in range(self.channels.count())}
        self.__on_save({**groups, **individuals})
        super().accept()


class SWChannelSelectorDialog(QDialog, Ui_jriverChannelSelectDialog):

    def __init__(self, parent: QDialog, channels: List[str], sw_channels: List[str],
                 on_save: Callable[[List[str]], None]):
        super(SWChannelSelectorDialog, self).__init__(parent)
        self.setupUi(self)
        self.channelList.setSelectionMode(QAbstractItemView.MultiSelection)
        self.__on_save = on_save
        self.setWindowTitle('Select SW Output Channels')
        self.channelListLabel.hide()
        for c in channels:
            self.channelList.addItem(c)
        for c in sw_channels:
            item: QListWidgetItem
            for item in self.channelList.findItems(c, Qt.MatchCaseSensitive):
                item.setSelected(True)

    def accept(self):
        self.__on_save([i.text() for i in self.channelList.selectedItems()])
        super().accept()

