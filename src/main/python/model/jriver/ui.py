from __future__ import annotations

import itertools
import json
import locale
import logging
import math
import os
import sys
import xml.etree.ElementTree as et
from builtins import isinstance
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Callable, Set, Any, Sequence

import qtawesome as qta
from qtpy.QtCore import QPoint, QModelIndex, Qt, QTimer, QAbstractTableModel, QVariant, QSize
from qtpy.QtGui import QColor, QPalette, QKeySequence, QCloseEvent, QShowEvent, QFont, QIcon
from qtpy.QtWidgets import QDialog, QFileDialog, QMenu, QAction, QListWidgetItem, QMessageBox, QInputDialog, \
    QDialogButtonBox, QAbstractItemView, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox, \
    QSpacerItem, QSizePolicy, QGridLayout, QPushButton, QDoubleSpinBox, QAbstractSpinBox, QComboBox, QHeaderView, \
    QListWidget, QTableWidgetItem

from model.filter import FilterModel, FilterDialog
from model.iir import SOS, CompleteFilter, FilterType, ComplexLowPass, ComplexHighPass, DEFAULT_Q
from model.jriver import JRIVER_FS, flatten, ImpossibleRoutingError
from model.jriver.codec import get_element, xpath_to_key_data_value, write_dsp_file, get_peq_key_name
from model.jriver.common import get_channel_name, get_channel_idx, OutputFormat, SHORT_USER_CHANNELS, make_dirac_pulse, \
    OUTPUT_FORMATS
from model.jriver.dsp import JRiverDSP
from model.jriver.filter import Divider, GEQFilter, CompoundRoutingFilter, CustomPassFilter, GainQFilter, Gain, Pass, \
    LinkwitzTransform, Polarity, Mix, Delay, Filter, create_single_filter, MixType, ChannelFilter, \
    convert_filter_to_mc_dsp, \
    SingleFilter, XOFilter, HighPass, LowPass, SimulationFailed, MSOFilter, MDSXO, WayValues, XO, \
    StandardXO, MultiwayCrossover, MultiwayFilter
from model.jriver.mcws import MediaServer, MCWSError, DSPMismatchError
from model.jriver.parser import from_mso
from model.jriver.render import render_dot
from model.jriver.routing import Matrix, LFE_ADJUST_KEY, EDITORS_KEY, EDITOR_NAME_KEY, \
    UNDERLYING_KEY, WAYS_KEY, SYM_KEY, LFE_IN_KEY, ROUTING_KEY, calculate_compound_routing_filter, normalise_delays, \
    group_routes_by_output_channel
from model.limits import DecibelRangeCalculator, PhaseRangeCalculator
from model.magnitude import MagnitudeModel
from model.preferences import JRIVER_GEOMETRY, JRIVER_GRAPH_X_MIN, JRIVER_GRAPH_X_MAX, JRIVER_DSP_DIR, \
    get_filter_colour, Preferences, XO_GEOMETRY, JRIVER_MCWS_CONNECTIONS
from model.signal import Signal
from model.xy import MagnitudeData
from ui.channel_matrix import Ui_channelMatrixDialog
from ui.channel_select import Ui_channelSelectDialog
from ui.delegates import CheckBoxDelegate, FreqRangeEditor
from ui.group_channels import Ui_groupChannelsDialog
from ui.jriver import Ui_jriverDspDialog
from ui.jriver_delay_filter import Ui_jriverDelayDialog
from ui.jriver_gain_filter import Ui_jriverGainDialog
from ui.jriver_mix_filter import Ui_jriverMixDialog
from ui.load_zone import Ui_loadDspFromZoneDialog
from ui.mds import Ui_mdsDialog
from ui.mso import Ui_msoDialog
from ui.pipeline import Ui_jriverGraphDialog
from ui.xo import Ui_xoDialog
from ui.xofilters import Ui_xoFiltersDialog

FILTER_ID_ROLE = Qt.UserRole + 1

logger = logging.getLogger('jriver.ui')


class JRiverDSPDialog(QDialog, Ui_jriverDspDialog):

    def __init__(self, parent, prefs: Preferences):
        super(JRiverDSPDialog, self).__init__(parent)
        self.__ignore_gz_not_installed = False
        self.__selected_node_names: Set[str] = set()
        self.__current_dot_txt = None
        self.prefs = prefs
        self.setupUi(self)
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)
        self.__decorate_buttons()
        self.loadZoneButton.clicked.connect(self.__show_zone_dialog)
        self.addFilterButton.setMenu(self.__populate_add_filter_menu(QMenu(self)))
        self.pipelineView.signal.on_click.connect(self.__on_node_click)
        self.pipelineView.signal.on_double_click.connect(self.__show_edit_filter_dialog)
        self.pipelineView.signal.on_context.connect(self.__show_edit_menu)
        self.showDotButton.clicked.connect(self.__show_dot_dialog)
        self.uploadButton.clicked.connect(self.__upload_dsp)
        self.direction.toggled.connect(self.__regen)
        self.viewSplitter.setSizes([100000, 100000])
        self.filterList.model().rowsMoved.connect(self.__reorder_filters)
        self.__dsp: Optional[JRiverDSP] = None
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, prefs,
                                                self.__get_data(), 'Filter', fill_primary=False,
                                                x_min_pref_key=JRIVER_GRAPH_X_MIN, x_max_pref_key=JRIVER_GRAPH_X_MAX,
                                                x_scale_pref_key=None,
                                                secondary_data_provider=self.__get_data('phase'),
                                                secondary_name='Phase', secondary_prefix='deg', fill_secondary=False,
                                                y_range_calc=DecibelRangeCalculator(60),
                                                y2_range_calc=PhaseRangeCalculator(), show_y2_in_legend=False)
        self.__restore_geometry()

    def __show_zone_dialog(self):
        def on_select(zone_name: str, dsp: str, convert_q: bool):
            logger.info(f"Received dsp config from {zone_name} len {len(dsp)}")
            self.__load_dsp(zone_name, txt=dsp, convert_q=convert_q, allow_padding=not convert_q)

        MCWSDialog(self, self.prefs, on_select=on_select).exec()

    def __upload_dsp(self):
        if self.dsp:
            logger.info(f"Uploading dsp config")
            MCWSDialog(self, self.prefs, txt_provider=self.dsp.config_txt, download=False).exec()

    def __enable_history_buttons(self, back: bool, fwd: bool) -> None:
        self.backButton.setEnabled(back)
        self.forwardButton.setEnabled(fwd)

    def __undo(self):
        if self.dsp.active_graph.undo():
            self.show_filters()

    def __redo(self):
        if self.dsp.active_graph.redo():
            self.show_filters()

    @property
    def dsp(self) -> Optional[JRiverDSP]:
        return self.__dsp

    def __decorate_buttons(self):
        self.newConfigButton.setToolTip('Create New Configuration')
        self.newConfigButton.setIcon(qta.icon('fa5s.file'))
        self.newConfigButton.setShortcut(QKeySequence.New)
        self.addFilterButton.setToolTip('Add New Filter')
        self.editFilterButton.setToolTip('Edit the selected filter')
        self.deleteFilterButton.setToolTip('Delete the selected filter(s)')
        self.deleteFilterButton.setShortcut(QKeySequence.Delete)
        self.clearFiltersButton.setToolTip('Delete all filters')
        self.splitFilterButton.setToolTip('Split multi channel filter into separate filters per channel')
        self.mergeFilterButton.setToolTip('Merge individual filters into a single multi channel filter')
        self.limitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.fullRangeButton.setIcon(qta.icon('fa5s.expand'))
        self.subOnlyButton.setIcon(qta.icon('fa5s.compress'))
        self.findFilenameButton.setIcon(qta.icon('fa5s.folder-open'))
        self.showDotButton.setIcon(qta.icon('fa5s.info-circle'))
        self.showPhase.setIcon(qta.icon('mdi.cosine-wave'))
        self.saveButton.setIcon(qta.icon('fa5s.save'))
        self.saveAsButton.setIcon(qta.icon('fa5s.file-export'))
        self.addFilterButton.setIcon(qta.icon('fa5s.plus'))
        self.deleteFilterButton.setIcon(qta.icon('fa5s.times'))
        self.clearFiltersButton.setIcon(qta.icon('fa5s.trash'))
        self.editFilterButton.setIcon(qta.icon('fa5s.edit'))
        self.splitFilterButton.setIcon(qta.icon('fa5s.object-ungroup'))
        self.mergeFilterButton.setIcon(qta.icon('fa5s.object-group'))
        self.moveTopButton.setIcon(qta.icon('fa5s.angle-double-up'))
        self.moveUpButton.setIcon(qta.icon('fa5s.angle-up'))
        self.moveDownButton.setIcon(qta.icon('fa5s.angle-down'))
        self.moveBottomButton.setIcon(qta.icon('fa5s.angle-double-down'))
        self.findFilenameButton.setShortcut(QKeySequence.Open)
        self.saveButton.setShortcut(QKeySequence.Save)
        self.saveButton.setToolTip('Save DSP Config to currently loaded file')
        self.saveAsButton.setShortcut(QKeySequence.SaveAs)
        self.saveAsButton.setToolTip('Save DSP Config to a selected file')
        self.findFilenameButton.setToolTip('Load DSP Config')
        self.moveTopButton.setToolTip('Move selected filters to the top')
        self.moveUpButton.setToolTip('Move selected filters up')
        self.moveDownButton.setToolTip('Move selected filters down')
        self.moveBottomButton.setToolTip('Move selected filters to the bottom')
        self.limitsButton.setToolTip('Set graph axis range')
        self.subOnlyButton.setToolTip('Restrict graph to subwoofer frequency range')
        self.fullRangeButton.setToolTip('Expand graph to full frequency range')
        self.showPhase.setToolTip('Show Phase Response')
        self.forwardButton.setIcon(qta.icon('fa5s.arrow-right'))
        self.forwardButton.setToolTip('Redo')
        self.forwardButton.setShortcut(QKeySequence.Redo)
        self.backButton.setIcon(qta.icon('fa5s.arrow-left'))
        self.backButton.setToolTip('Undo')
        self.backButton.setShortcut(QKeySequence.Undo)
        self.backButton.clicked.connect(self.__undo)
        self.forwardButton.clicked.connect(self.__redo)
        self.loadZoneButton.setIcon(qta.icon('fa5s.download'))
        self.loadZoneButton.setToolTip('Connect to JRiver Media Center to load DSP config for a zone')
        self.loadZoneButton.setShortcut("Ctrl+Shift+O")
        self.uploadButton.setIcon(qta.icon('fa5s.upload'))
        self.uploadButton.setToolTip('Load the current DSP to JRiver Media Center')
        self.uploadButton.setShortcut("Ctrl+Shift+S")

    def create_new_config(self):
        '''
        Creates a new configuration with a selected output format.
        '''
        mc_version = self.__pick_mc_version()
        of: OutputFormat
        all_formats: List[OutputFormat] = sorted([of for of in OUTPUT_FORMATS.values() if of.is_compatible(mc_version)])
        output_formats = [of.display_name for of in all_formats]
        item, ok = QInputDialog.getItem(self, "Create New DSP Config", "Output Format:", output_formats, 0, False)
        if ok and item:
            selected: OutputFormat = next((of for of in all_formats if of.display_name == item))
            selected_padding: int = 0
            if selected.paddings and mc_version >= 29:
                item, ok = QInputDialog.getItem(self, "Extra Channels?", "Count:",
                                                ["None"] + [f"{p} channels" for p in selected.paddings], 0, False)
                if ok and item and item != 'None':
                    selected_padding = int(item.split()[0].rstrip())
            logger.info(
                f"Creating new MC{mc_version} configuration for {selected} with {selected_padding} extra channels")
            if getattr(sys, 'frozen', False):
                file_path = os.path.join(sys._MEIPASS, f"default_jriver_config_{mc_version}.xml")
            else:
                file_path = os.path.abspath(os.path.join(os.path.dirname('__file__'),
                                                         f'../xml/default_jriver_config_{mc_version}.xml'))
            config_txt = Path(file_path).read_text()
            root = et.fromstring(config_txt)

            def get(key) -> et.Element:
                return get_element(root, xpath_to_key_data_value('Audio Settings', key))

            output_channels = get('Output Channels')
            padding = get('Output Padding Channels')
            layout = get('Output Channel Layout')
            output_channels.text = f"{selected.xml_vals[0]}"
            default_file_name: str = selected.display_name
            if selected_padding > 0:
                padding.text = str(selected_padding)
                default_file_name = f"{default_file_name}+{selected_padding}C"
            elif len(selected.xml_vals) > 1:
                padding.text = f"{selected.xml_vals[1]}"
            if len(selected.xml_vals) > 2:
                layout.text = f"{selected.xml_vals[2]}"
            file_name, ok = QInputDialog.getText(self, "Create New DSP Config", "Config Name:", text=default_file_name)
            output_file = os.path.join(self.prefs.get(JRIVER_DSP_DIR), f"{file_name}.dsp")
            write_dsp_file(root, output_file)
            self.__load_dsp(output_file, allow_padding=selected_padding > 0)

    def __pick_mc_version(self) -> int:
        mc_version = 30
        item, ok = QInputDialog.getItem(self, 'Choose MC Version', 'Version: ', ['30', '29', '28'], 0, False)
        if ok and item:
            mc_version = int(item)
        return mc_version

    def find_dsp_file(self):
        '''
        Allows user to select a DSP file and loads it as a set of graphs.
        '''
        mc_version = self.__pick_mc_version()
        dsp_dir = self.prefs.get(JRIVER_DSP_DIR)
        kwargs = {
            'caption': 'Select JRiver Media Centre DSP File',
            'filter': 'DSP (*.dsp)'
        }
        if dsp_dir is not None and len(dsp_dir) > 0 and os.path.exists(dsp_dir):
            kwargs['directory'] = dsp_dir
        selected = QFileDialog.getOpenFileName(parent=self, **kwargs)
        if selected is not None and len(selected[0]) > 0:
            legacy = mc_version < 29
            self.__load_dsp(selected[0], convert_q=legacy, allow_padding=not legacy)

    def __load_dsp(self, name: str, txt: str = None, convert_q: bool = False, allow_padding: bool = False) -> None:
        '''
        Loads the selected file.
        :param name: the name of the dsp.
        :param txt: the dsp config txt.
        '''
        try:
            main_colour = QColor(QPalette().color(QPalette.Active, QPalette.Text)).name()
            highlight_colour = QColor(QPalette().color(QPalette.Active, QPalette.Highlight)).name()
            self.__dsp = JRiverDSP(name, lambda: txt if txt else Path(name).read_text(),
                                   convert_q=convert_q, allow_padding=allow_padding,
                                   colours=(main_colour, highlight_colour),
                                   on_delta=self.__enable_history_buttons)
            self.__refresh_channel_list()
            self.filename.setText(name if txt else os.path.basename(name)[:-4])
            self.outputFormat.setText(self.dsp.output_format.display_name)
            self.filterList.clear()
            self.show_filters()
            self.saveButton.setEnabled(True)
            self.saveAsButton.setEnabled(True)
            self.addFilterButton.setEnabled(True)
            self.showDotButton.setEnabled(True)
            self.direction.setEnabled(True)
            self.uploadButton.setEnabled(True)
            if not txt:
                self.prefs.set(JRIVER_DSP_DIR, os.path.dirname(name))
        except Exception as e:
            logger.exception(f"Unable to parse {name}")
            from model.catalogue import show_alert
            show_alert('Unable to load DSP file', f"Invalid file\n\n{e}")

    def show_filters(self):
        '''
        Displays the complete filter list for the selected PEQ block.
        '''
        if self.dsp is not None:
            try:
                self.dsp.activate(self.blockSelector.currentIndex())
            except SimulationFailed as e:
                msg_box = QMessageBox()
                msg_box.setText(f"{e}")
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle('Failed to Simulate Filter Pipeline!')
                msg_box.exec()

            from model.report import block_signals
            with block_signals(self.filterList):
                selected_ids = [i.data(FILTER_ID_ROLE) for i in self.filterList.selectedItems()]
                i = 0
                for f in self.dsp.active_graph.filters:
                    if not isinstance(f, Divider):
                        if self.filterList.count() > i:
                            item: QListWidgetItem = self.filterList.item(i)
                            if item.data(FILTER_ID_ROLE) != f.id:
                                item.setText(str(f))
                                item.setData(FILTER_ID_ROLE, f.id)
                            else:
                                f_t = str(f)
                                if item.text() != f_t:
                                    item.setText(f_t)
                        else:
                            item = QListWidgetItem(str(f))
                            item.setData(FILTER_ID_ROLE, f.id)
                            self.filterList.addItem(item)
                        i += 1
                        item.setSelected(f.id in selected_ids)
                for j in range(self.filterList.count(), i, -1):
                    self.filterList.takeItem(j - 1)
            self.__regen()
        self.redraw()

    def edit_selected_filter(self):
        if self.filterList.selectedItems():
            self.edit_filter(self.filterList.selectedItems()[0])

    def __is_editable(self, item: QListWidgetItem) -> bool:
        '''
        :param item: the item.
        :return: true if this item is editable directly, typically means applies to a single channel.
        '''
        f_id = item.data(FILTER_ID_ROLE)
        selected_filter: Optional[Filter] = next((f for f in self.dsp.active_graph.filters if f.id == f_id), None)
        if selected_filter:
            if isinstance(selected_filter, (GEQFilter, CompoundRoutingFilter, MSOFilter)):
                return True
            elif isinstance(selected_filter, CustomPassFilter):
                return selected_filter.channels and len(selected_filter.channels) == 1
            else:
                if isinstance(selected_filter, (GainQFilter, Gain, Pass, LinkwitzTransform)):
                    return len(selected_filter.channels) == 1
                else:
                    vals = selected_filter.get_all_vals()
                    return len(vals) == 1 and isinstance(selected_filter, (Delay, Mix, Polarity))
        else:
            logger.warning(f"Selected item has no filter {item.text()}")

    def edit_filter(self, item: QListWidgetItem) -> None:
        '''
        Shows a jriver style filter dialog for the selected filter.
        :param item: the item describing the filter.
        '''
        f_id = item.data(FILTER_ID_ROLE)
        selected_filter: Optional[Filter] = next((f for f in self.dsp.active_graph.filters if f.id == f_id), None)
        logger.debug(f"Showing edit dialog for {selected_filter}")
        if isinstance(selected_filter, GEQFilter):
            self.__start_geq_edit_session(selected_filter, selected_filter.channel_names)
        elif isinstance(selected_filter, CompoundRoutingFilter):
            self.__update_xo(selected_filter)
        elif isinstance(selected_filter, MSOFilter):
            self.__update_mso(selected_filter)
        else:
            vals = selected_filter.get_all_vals()
            if not self.__show_basic_edit_filter_dialog(selected_filter, vals):
                if len(selected_filter.nodes) == 1:
                    self.__show_edit_filter_dialog(selected_filter.nodes[0])
                elif selected_filter.get_editable_filter() is not None \
                        and isinstance(selected_filter, ChannelFilter) \
                        and len(selected_filter.channels) > 1:
                    result = QMessageBox.question(self,
                                                  'Split to Separate Filters?',
                                                  f"The selected filter is applied to {len(selected_filter.channels)} channels."
                                                  f"\n\nDo you want to split into a filter per channel in order to edit?",
                                                  QMessageBox.Yes | QMessageBox.No,
                                                  QMessageBox.No)
                    if result == QMessageBox.Yes:
                        self.split_filter()
                else:
                    logger.debug(f"Filter {selected_filter} at node {item.text()} is not editable")
            else:
                logger.warning(f"Unexpected filter type {selected_filter} at {item.text()}, unable to edit")

    def delete_filter(self) -> None:
        '''
        Deletes the selected filter(s).
        '''
        selected_items = [i for i in self.filterList.selectedItems()]
        selected_filter_ids = [i.data(FILTER_ID_ROLE) for i in selected_items]
        to_delete = [f for f in self.dsp.active_graph.filters if f.id in selected_filter_ids]
        logger.debug(f"Deleting filter ids {selected_filter_ids} -> {to_delete}")
        self.dsp.active_graph.delete(to_delete)
        self.__on_graph_change()
        last_row = 0
        i: QModelIndex
        from model.report import block_signals
        with block_signals(self.filterList):
            for i in selected_items:
                last_row = self.filterList.indexFromItem(i).row()
                removed = self.filterList.takeItem(last_row)
                logger.debug(f"Removing filter at row {last_row} - {removed.text()}")
        if self.filterList.count() > 0:
            self.filterList.item(max(min(last_row, self.filterList.count()) - 1, 0)).setSelected(True)
        self.__enable_edit_buttons_if_filters_selected()

    def split_filter(self):
        ''' Splits a selected multichannel filter into separate filters. '''
        selected_items: List[QListWidgetItem] = self.filterList.selectedItems()
        splittable = self.dsp.active_graph.get_filter_by_id(selected_items[0].data(FILTER_ID_ROLE))
        item_idx: QModelIndex = self.filterList.indexFromItem(selected_items[0])
        base_idx = item_idx.row()
        if splittable.can_split():
            split = splittable.split()
            self.__insert_multiple_filters(base_idx, split)
            self.dsp.active_graph.delete([splittable])
            self.filterList.takeItem(self.filterList.indexFromItem(selected_items[0]).row())
            self.__regen()

    def __insert_multiple_filters(self, base_idx: int, to_insert: List[Filter]):
        for i, f in enumerate(to_insert):
            insert_at = base_idx + i
            self.dsp.active_graph.insert(f, insert_at, regen=(i + 1 == len(to_insert)))
            new_item = QListWidgetItem(str(f))
            new_item.setData(FILTER_ID_ROLE, f.id)
            self.filterList.insertItem(insert_at, new_item)

    def merge_filters(self):
        ''' Merges multiple identical filters into a single multichannel filter. '''
        selected_filters = [self.dsp.active_graph.get_filter_by_id(i.data(FILTER_ID_ROLE))
                            for i in self.filterList.selectedItems()]
        item_idx: QModelIndex = self.filterList.indexFromItem(self.filterList.selectedItems()[0])
        # can only merge a ChannelFilter
        channels = ';'.join([str(i) for i in sorted(set(c for f in selected_filters for c in f.channels))])
        merged_filter = create_single_filter({**selected_filters[0].get_all_vals()[0], 'Channels': channels})
        insert_at = item_idx.row()
        # insert the new one in both the graph and the filter list
        self.dsp.active_graph.insert(merged_filter, insert_at)
        new_item = QListWidgetItem(str(merged_filter))
        new_item.setData(FILTER_ID_ROLE, merged_filter.id)
        self.filterList.insertItem(insert_at, new_item)
        # delete the old ones
        self.dsp.active_graph.delete(selected_filters)
        from model.report import block_signals
        with block_signals(self.filterList):
            for i in self.filterList.selectedItems():
                self.filterList.takeItem(self.filterList.indexFromItem(i).row())
            self.filterList.clearSelection()
        new_item.setSelected(True)

    def move_filter_to_top(self):
        '''
        Moves the selected filter(s) to the top.
        '''
        selected = [i.row() for i in self.filterList.selectedIndexes()]
        self.__move_to(selected[0], selected[-1], 0)

    def __move_to(self, start: int, end: int, to: int):
        self.dsp.active_graph.reorder(start, end, to)
        self.show_filters()

    def move_filter_up(self):
        '''
        moves the selected filter(s) up one slot.
        '''
        selected = [i.row() for i in self.filterList.selectedIndexes()]
        self.__move_to(selected[0], selected[-1], max(0, selected[0] - 1))

    def move_filter_down(self):
        '''
        moves the selected filter(s) down one slot.
        '''
        selected = [i.row() for i in self.filterList.selectedIndexes()]
        self.__move_to(selected[0], selected[-1], min(self.filterList.count() - len(selected), selected[0] + 1) + 1)

    def move_filter_to_bottom(self):
        '''
        Moves the selected filter(s) to the end of the list.
        '''
        selected = [i.row() for i in self.filterList.selectedIndexes()]
        self.__move_to(selected[0], selected[-1], self.filterList.count() - len(selected) + 1)

    def on_filter_select(self) -> None:
        '''
        Ensures the selected nodes match the selected filters & regenerates the svg to display this.
        '''
        self.__selected_node_names.clear()
        if self.__enable_edit_buttons_if_filters_selected():
            selected_indexes = [i.row() for i in self.filterList.selectedIndexes()]
            for i in selected_indexes:
                for f in flatten(self.dsp.active_graph.filters[i]):
                    for n in f.nodes:
                        self.__selected_node_names.add(n)
        self.__regen()

    def clear_filters(self):
        '''
        Deletes all filters in the active graph.
        '''
        self.dsp.active_graph.clear_filters()
        self.show_filters()

    def save_dsp(self):
        '''
        Writes the graphs to the loaded file.
        '''
        if self.dsp:
            self.dsp.write_to_file()

    def save_as_dsp(self):
        '''
        Writes the graphs to a user specified file.
        '''
        if self.dsp:
            file_name = QFileDialog(self).getSaveFileName(self, caption='Save DSP Config',
                                                          directory=os.path.dirname(self.dsp.filename),
                                                          filter="JRiver DSP (*.dsp)")
            file_name = str(file_name[0]).strip()
            if len(file_name) > 0:
                self.dsp.write_to_file(file=file_name)

    def show_impulse(self):
        '''
        Shows the impulse charts.
        '''
        pass

    def __refresh_channel_list(self, retain_selected=False):
        ''' Refreshes the output channels with the current channel list. '''
        from model.report import block_signals
        with block_signals(self.channelList):
            selected = [i.text() for i in self.channelList.selectedItems()]
            self.channelList.clear()
            for i, n in enumerate(self.dsp.channel_names(output=True)):
                self.channelList.addItem(n)
                item: QListWidgetItem = self.channelList.item(i)
                item.setSelected(n in selected if retain_selected else n not in SHORT_USER_CHANNELS)
        with block_signals(self.blockSelector):
            self.blockSelector.clear()
            for i in range(self.dsp.graph_count):
                self.blockSelector.addItem(get_peq_key_name(self.dsp.graph(i).stage))

    def __show_edit_menu(self, node_name: str, pos: QPoint) -> None:
        '''
        Displays a context menu to allow edits to the graph to be driven from a selected node.
        :param node_name: the selected node.
        :param pos: the location to place the menu.
        '''
        filt = self.dsp.active_graph.get_filter_at_node(node_name)
        if filt:
            menu = QMenu(self)
            self.__populate_edit_node_add_menu(menu.addMenu('&Add'), node_name)
            edit = QAction(f"&Edit", self)
            edit.triggered.connect(lambda: self.__show_edit_filter_dialog(node_name))
            menu.addAction(edit)
            if not isinstance(filt, CompoundRoutingFilter):
                delete = QAction(f"&Delete", self)
                delete.triggered.connect(lambda: self.__delete_node(node_name))
                menu.addAction(delete)
            menu.exec(pos)

    def __populate_edit_node_add_menu(self, add_menu: QMenu, node_name: str):
        '''
        Adds all filter actions to the context menu shown against a node.
        :param add_menu: the add menu.
        :param node_name: the selected node name.
        '''
        add, copy, delay, move, peq, polarity, gain, subtract, swap, geq, xo, mso = self.__add_actions_to_filter_menu(
            add_menu)
        idx = self.__get_idx_to_insert_filter_at_from_node_name(node_name)
        peq.triggered.connect(lambda: self.__insert_peq_after_node(node_name))
        polarity.triggered.connect(lambda: self.__insert_polarity(idx))
        gain.triggered.connect(lambda: self.__insert_gain(idx))
        delay.triggered.connect(lambda: self.__insert_delay(idx))
        add.triggered.connect(lambda: self.__insert_mix(MixType.ADD, idx))
        copy.triggered.connect(lambda: self.__insert_mix(MixType.COPY, idx))
        move.triggered.connect(lambda: self.__insert_mix(MixType.MOVE, idx))
        swap.triggered.connect(lambda: self.__insert_mix(MixType.SWAP, idx))
        subtract.triggered.connect(lambda: self.__insert_mix(MixType.SUBTRACT, idx))
        geq.triggered.connect(lambda: self.__insert_geq(idx))
        xo.triggered.connect(lambda: self.__insert_xo(idx))
        mso.triggered.connect(lambda: self.__insert_mso(idx))

    def __get_idx_to_insert_filter_at_from_node_name(self, node_name: str) -> int:
        '''
        Locates the position in the filter list at which a filter should be added in order to be placed after the
        specified node in the pipeline.
        :param node_name: the node name.
        :return: an index to insert at.
        '''
        filt = self.dsp.active_graph.get_filter_at_node(node_name)
        if filt:
            match = self.__find_item_by_filter_id(filt.id)
            if match:
                return match[0] + 1
        return 0

    def __get_idx_to_insert_filter_at_from_selection(self) -> int:
        '''
        Locates the position in the filter list at which a filter should be added based on the selected filters.
        :return: an index to insert at.
        '''
        selected = [i.row() for i in self.filterList.selectedIndexes()]
        return max(selected) + 1 if selected else self.filterList.count()

    def __find_item_by_filter_id(self, f_id: int) -> Optional[Tuple[int, QListWidgetItem]]:
        '''
        Locates the item in the filter list that has the specified filter id.
        :param f_id: the filter id.
        :return: index, item or none if the id is not found.
        '''
        for i in range(self.filterList.count()):
            item: QListWidgetItem = self.filterList.item(i)
            if item.data(FILTER_ID_ROLE) == f_id:
                return i, item
        return None

    def __insert_peq_after_node(self, node_name: str) -> None:
        '''
        Triggers the peq filter dialog to allow a user to insert a new chain of PEQs after the specified node in the
        pipeline.
        :param node_name: the node to insert after.
        '''
        filt = self.dsp.active_graph.get_filter_at_node(node_name)
        if filt:
            match = self.__find_item_by_filter_id(filt.id)
            if match:
                if filt.get_editable_filter():
                    node_idx, node_chain = self.dsp.active_graph.get_editable_node_chain(node_name)
                    filters: List[Filter] = [self.dsp.active_graph.get_filter_at_node(n) for n in node_chain]
                    insert_at, _ = self.__find_item_by_filter_id(filters[0].id)
                    # TODO insert a filter into the chain immediately after this one
                    self.__start_peq_edit_session([f.get_editable_filter() for f in filters], node_name.split('_')[0],
                                                  node_chain, insert_at + 1)
                else:
                    self.__start_peq_edit_session(None, node_name.split('_')[0], [], match[0] + 1)

    def __delete_node(self, node_name: str) -> None:
        '''
        Deletes the node from the filter list.
        :param node_name: the node to remove.
        '''
        f = self.dsp.active_graph.get_filter_at_node(node_name)
        node_channel = node_name.split('_')[0]
        if f:
            if isinstance(f, ChannelFilter):
                item: QListWidgetItem
                if self.dsp.active_graph.delete_channel(f, node_channel):
                    match = self.__find_item_by_filter_id(f.id)
                    if match:
                        self.filterList.takeItem(match[0])
                else:
                    match = self.__find_item_by_filter_id(f.id)
                    if match:
                        match[1].setText(str(f))
            else:
                self.dsp.active_graph.delete([f])
            self.__regen()

    def __insert_xo(self, idx: int) -> None:
        '''
        Shows the XO dialog and inserts the resulting filters at the specified index.
        :param idx: the index.
        '''

        def on_save(xo_filters: CompoundRoutingFilter):
            self.__insert_filter(idx, xo_filters)
            self.__on_graph_change()

        self.__show_xo_dialog(on_save)

    def __update_xo(self, existing: CompoundRoutingFilter) -> None:
        '''
        Shows the XO dialog and replaces the existing filter at the specified index.
        :param existing: the current xo filter.
        '''

        def on_save(xo_filters: CompoundRoutingFilter):
            if self.dsp.active_graph.replace(existing, xo_filters):
                self.__on_graph_change()

        self.__show_xo_dialog(on_save, existing=existing)

    def __show_xo_dialog(self, on_save: Callable[[CompoundRoutingFilter], None],
                         existing: CompoundRoutingFilter = None):
        # TODO the input channels will have changed if any mix operations before here, detect this and update matrix
        XODialog(self, self.prefs, self.dsp.channel_names(), self.dsp.channel_names(output=True, exclude_user=True),
                 self.dsp.output_format, on_save, existing=existing).exec()

    def __insert_mso(self, idx: int) -> None:
        '''
        Shows the MSO dialog and inserts the resulting filters at the specified index.
        :param idx: the index.
        '''

        def on_save(mso_filter: MSOFilter):
            self.__insert_filter(idx, mso_filter)
            self.__on_graph_change()

        self.__show_mso_dialog(on_save)

    def __update_mso(self, existing: MSOFilter) -> None:
        '''
        Shows the mso dialog and replaces the existing filter at the specified index.
        :param existing: the current mso filter.
        '''

        def on_save(mso_filters: MSOFilter):
            if self.dsp.active_graph.replace(existing, mso_filters):
                self.__on_graph_change()

        self.__show_mso_dialog(on_save, existing=existing)

    def __show_mso_dialog(self, on_save: Callable[[MSOFilter], None], existing: MSOFilter = None):
        MSODialog(existing, on_save, self.dsp.output_format, self).exec()

    def __populate_add_filter_menu(self, menu: QMenu) -> QMenu:
        '''
        Adds filter editing actions to the add button next to the filter list.
        :param menu: the menu to add to.
        :return: the menu.
        '''
        add, copy, delay, move, peq, polarity, gain, subtract, swap, geq, xo, mso = self.__add_actions_to_filter_menu(
            menu)
        peq.triggered.connect(lambda: self.__insert_peq(self.__get_idx_to_insert_filter_at_from_selection()))
        delay.triggered.connect(lambda: self.__insert_delay(self.__get_idx_to_insert_filter_at_from_selection()))
        polarity.triggered.connect(lambda: self.__insert_polarity(self.__get_idx_to_insert_filter_at_from_selection()))
        gain.triggered.connect(lambda: self.__insert_gain(self.__get_idx_to_insert_filter_at_from_selection()))
        add.triggered.connect(lambda: self.__insert_mix(MixType.ADD,
                                                        self.__get_idx_to_insert_filter_at_from_selection()))
        copy.triggered.connect(lambda: self.__insert_mix(MixType.COPY,
                                                         self.__get_idx_to_insert_filter_at_from_selection()))
        move.triggered.connect(lambda: self.__insert_mix(MixType.MOVE,
                                                         self.__get_idx_to_insert_filter_at_from_selection()))
        swap.triggered.connect(lambda: self.__insert_mix(MixType.SWAP,
                                                         self.__get_idx_to_insert_filter_at_from_selection()))
        subtract.triggered.connect(lambda: self.__insert_mix(MixType.SUBTRACT,
                                                             self.__get_idx_to_insert_filter_at_from_selection()))
        geq.triggered.connect(lambda: self.__insert_geq(self.__get_idx_to_insert_filter_at_from_selection()))
        xo.triggered.connect(lambda: self.__insert_xo(self.__get_idx_to_insert_filter_at_from_selection()))
        mso.triggered.connect(lambda: self.__insert_mso(self.__get_idx_to_insert_filter_at_from_selection()))
        return menu

    def __add_actions_to_filter_menu(self, menu) -> Tuple[QAction, ...]:
        '''
        Adds all filter edit actions to the menu. No triggers are linked at this point.
        :param menu: the menu to add to.
        :return: the actions.
        '''
        geq = self.__create_geq_action(1, menu)
        peq = self.__create_peq_action(2, menu)
        xo = self.__create_xo_action(3, menu)
        mix_menu = menu.addMenu('&4: Mix')
        delay = self.__create_delay_action(5, menu)
        polarity = self.__create_polarity_action(6, menu)
        gain = self.__create_gain_action(7, menu)
        add = self.__create_add_action(1, mix_menu)
        copy = self.__create_copy_action(2, mix_menu)
        move = self.__create_move_action(3, mix_menu)
        swap = self.__create_swap_action(4, mix_menu)
        subtract = self.__create_subtract_action(5, mix_menu)
        mso = self.__create_mso_action(8, menu)
        return add, copy, delay, move, peq, polarity, gain, subtract, swap, geq, xo, mso

    def __create_geq_action(self, prefix: int, menu: QMenu) -> QAction:
        geq = QAction(f"&{prefix}: GEQ", self)
        menu.addAction(geq)
        return geq

    def __create_subtract_action(self, prefix: int, menu: QMenu) -> QAction:
        subtract = QAction(f"&{prefix}: Subtract", self)
        menu.addAction(subtract)
        return subtract

    def __create_swap_action(self, prefix: int, menu: QMenu) -> QAction:
        swap = QAction(f"&{prefix}: Swap", self)
        menu.addAction(swap)
        return swap

    def __create_move_action(self, prefix: int, menu: QMenu) -> QAction:
        move = QAction(f"&{prefix}: Move", self)
        menu.addAction(move)
        return move

    def __create_copy_action(self, prefix: int, menu: QMenu) -> QAction:
        copy = QAction(f"&{prefix}: Copy", self)
        menu.addAction(copy)
        return copy

    def __create_add_action(self, prefix: int, menu: QMenu) -> QAction:
        add = QAction(f"&{prefix}: Add", self)
        menu.addAction(add)
        return add

    def __create_polarity_action(self, prefix: int, menu: QMenu) -> QAction:
        polarity = QAction(f"&{prefix}: Polarity", self)
        menu.addAction(polarity)
        return polarity

    def __create_gain_action(self, prefix: int, menu: QMenu) -> QAction:
        gain = QAction(f"&{prefix}: Gain", self)
        menu.addAction(gain)
        return gain

    def __create_delay_action(self, prefix: int, menu: QMenu) -> QAction:
        delay = QAction(f"&{prefix}: Delay", self)
        menu.addAction(delay)
        return delay

    def __create_peq_action(self, prefix: int, menu: QMenu) -> QAction:
        peq = QAction(f"&{prefix}: PEQ", self)
        menu.addAction(peq)
        return peq

    def __create_xo_action(self, prefix: int, menu: QMenu) -> QAction:
        xo = QAction(f"&{prefix}: Crossover and Bass Management", self)
        menu.addAction(xo)
        return xo

    def __create_mso_action(self, prefix: int, menu: QMenu) -> QAction:
        mso = QAction(f"&{prefix}: Multi-Sub Optimiser", self)
        menu.addAction(mso)
        return mso

    def __insert_peq(self, idx: int) -> None:
        '''
        Allows user to insert a new set of PEQ filters at the specified index in the filter list.
        :param idx: the index to insert at.
        '''
        channel: Optional[str] = None

        def __on_save(vals: Dict[str, str]):
            nonlocal channel
            channel = get_channel_name(int(vals['Channels']))

        val = JRiverChannelOnlyFilterDialog(self, self.dsp.channel_names(output=True), __on_save, {}, title='PEQ',
                                            multi=False).exec()
        if val == QDialog.Accepted and channel is not None:
            self.__start_peq_edit_session(None, channel, [], idx)

    def __insert_geq(self, idx: int) -> None:
        '''
        Allows user to insert a new set of GEQ filters at the specified index in the filter list.
        :param idx: the index to insert at.
        '''
        channels: Optional[List[str]] = None

        def __on_save(vals: Dict[str, str]):
            nonlocal channels
            channels = [get_channel_name(int(i)) for i in vals['Channels'].split(';')]

        val = JRiverChannelOnlyFilterDialog(self, self.dsp.channel_names(output=True), __on_save, {},
                                            title='GEQ').exec()
        if val == QDialog.Accepted and channels:
            self.__start_geq_edit_session(None, channels, idx)

    def __start_geq_edit_session(self, geq: Optional[GEQFilter], channels: List[str],
                                 insert_at: Optional[int] = None) -> None:
        '''
        Creates a GEQ to insert.
        :param geq: the existing geq, if set any value for insert_at is ignored.
        :param channels: the channels to start with.
        :param insert_at: the idx to insert at, geq must be None to use this.
        '''
        all_channels = self.dsp.channel_names(output=True)

        def __on_save(channel_names: List[str], filters: List[SOS]):
            formatted_channels = ';'.join([str(get_channel_idx(c)) for c in channel_names])
            mc_filters = [convert_filter_to_mc_dsp(f, formatted_channels) for f in filters]
            new_geq = GEQFilter(mc_filters)
            if geq is not None:
                if self.dsp.active_graph.replace(geq, new_geq):
                    item = self.__find_item_by_filter_id(new_geq.id)
                    if item:
                        item[1].setText(str(new_geq))
                    self.__on_graph_change()
                else:
                    logger.warning(f"Unable to replace {geq} with {new_geq}")
            else:
                self.__insert_filter(insert_at, new_geq)

        existing_filters = [f.get_editable_filter() for f in geq.filters if f.get_editable_filter()] if geq else []
        from model.geq import GeqDialog
        GeqDialog(self, self.prefs, {c: c in channels for c in all_channels}, existing_filters, __on_save).exec()

    def __insert_delay(self, idx: int):
        '''
        Allows user to insert a delay filter at the specified index in the filter list.
        :param idx: the index to insert at.
        '''
        JRiverDelayDialog(self, self.dsp.channel_names(output=True), lambda vals: self.__add_filter(vals, idx),
                          Delay.default_values()).exec()

    def __insert_polarity(self, idx: int):
        '''
        Allows user to insert an invert polarity filter at the specified index in the filter list.
        :param idx: the index to insert at.
        '''
        JRiverChannelOnlyFilterDialog(self, self.dsp.channel_names(output=True),
                                      lambda vals: self.__add_filter(vals, idx), Polarity.default_values()).exec()

    def __insert_gain(self, idx: int):
        '''
        Allows user to insert a gain filter at the specified index in the filter list.
        :param idx: the index to insert at.
        '''
        JRiverGainDialog(self, self.dsp.channel_names(output=True), lambda vals: self.__add_filter(vals, idx),
                         Gain.default_values()).exec()

    def __insert_mix(self, mix_type: MixType, idx: int):
        '''
        Allows user to insert a mix filter at the specified index in the filter list.
        :param mix_type: the type of mix.
        :param idx: the index to insert at.
        '''
        JRiverMixFilterDialog(self, self.dsp.channel_names(output=True), lambda vals: self.__add_filter(vals, idx),
                              mix_type, Mix.default_values()).exec()

    def __add_filter(self, vals: Dict[str, str], idx: int) -> None:
        '''
        Creates a filter from the values supplied and inserts it into the filter list at the given index.
        :param vals: the filter values.
        :param idx: the index to insert at.
        '''
        to_add = create_single_filter(vals)
        if to_add:
            logger.info(f"Storing {vals} as {to_add}")
            self.__insert_filter(idx, to_add)
        else:
            logger.error(f"No PEQ created for {vals}")

    def __insert_filter(self, idx: int, to_add: Filter) -> None:
        '''
        inserts the filter to the active graph and updates the UI.
        :param idx: the idx to insert at.
        :param to_add: the filter.
        '''
        self.dsp.active_graph.insert(to_add, idx)
        self.__on_graph_change()
        item = QListWidgetItem(str(to_add))
        item.setData(FILTER_ID_ROLE, to_add.id)
        self.filterList.insertItem(idx, item)
        item.setSelected(self.filterList.item(self.filterList.count() - 1) == item)

    def __on_graph_change(self) -> None:
        '''
        regenerates the svg and redraws the chart.
        '''
        self.__regen()
        self.dsp.simulate()
        self.redraw()

    def __enable_edit_buttons_if_filters_selected(self) -> bool:
        '''
        :return: true if the delete button was enabled.
        '''
        count = len(self.filterList.selectedItems())
        self.deleteFilterButton.setEnabled(count > 0)
        self.moveTopButton.setEnabled(count > 0)
        self.moveUpButton.setEnabled(count > 0)
        self.moveDownButton.setEnabled(count > 0)
        self.moveBottomButton.setEnabled(count > 0)
        enable_split = False
        enable_merge = False
        enable_edit = False
        if count == 1:
            selected_item = self.filterList.selectedItems()[0]
            enable_edit = self.__is_editable(selected_item)
            f_id = selected_item.data(FILTER_ID_ROLE)
            enable_split = self.dsp.active_graph.get_filter_by_id(f_id).can_split()
        elif count > 1:
            selected_filters = [self.dsp.active_graph.get_filter_by_id(i.data(FILTER_ID_ROLE))
                                for i in self.filterList.selectedItems()]
            enable_merge = all(c[0].can_merge(c[1]) for c in itertools.combinations(selected_filters, 2))
        self.splitFilterButton.setEnabled(enable_split)
        self.mergeFilterButton.setEnabled(enable_merge)
        self.editFilterButton.setEnabled(enable_edit)
        return count > 0

    def __reorder_filters(self, parent: QModelIndex, start: int, end: int, dest: QModelIndex, row: int) -> None:
        '''
        Moves the filters at start to end to row and regenerates the views.
        :param parent: the parent of the selected items.
        :param start: start index of row(s) to move.
        :param end: end index of row(s) to move.
        :param dest: the parent of the target items.
        :param row: the row to move to.
        '''
        self.dsp.active_graph.reorder(start, end, row)
        self.__on_graph_change()

    def __show_edit_filter_dialog(self, node_name: str) -> None:
        '''
        Shows the edit dialog for the selected node.
        :param node_name: the node.
        '''
        filt = self.dsp.active_graph.get_filter_at_node(node_name)
        if filt:
            if isinstance(filt, GEQFilter):
                self.__start_geq_edit_session(filt, filt.channel_names)
            elif isinstance(filt, CompoundRoutingFilter):
                self.__update_xo(filt)
            elif isinstance(filt, MSOFilter):
                self.__update_mso(filt)
            elif filt.get_editable_filter():
                node_idx, node_chain = self.dsp.active_graph.get_editable_node_chain(node_name)
                filters: List[Filter] = [self.dsp.active_graph.get_filter_at_node(n) for n in node_chain]
                insert_at, _ = self.__find_item_by_filter_id(filters[0].id)
                self.__start_peq_edit_session([f.get_editable_filter() for f in filters], node_name.split('_')[0],
                                              node_chain, insert_at + 1, selected_filter_idx=node_idx)
            else:
                vals = filt.get_all_vals()
                if isinstance(filt, SingleFilter):
                    if not self.__show_basic_edit_filter_dialog(filt, vals):
                        logger.debug(f"Filter {filt} at node {node_name} is not editable")
                else:
                    logger.warning(f"Unexpected filter type {filt} at {node_name}, unable to edit")
        else:
            logger.debug(f"No filter at node {node_name}")

    def __show_basic_edit_filter_dialog(self, to_edit: Filter, vals: List[Dict[str, str]]) -> bool:
        if len(vals) == 1 and hasattr(to_edit, 'default_values'):

            def __on_save(vals_to_save: Dict[str, str]):
                to_add = create_single_filter(vals_to_save)
                if to_add:
                    logger.info(f"Storing {vals_to_save} as {to_add}")
                    if self.dsp.active_graph.replace(to_edit, to_add):
                        item = self.__find_item_by_filter_id(to_edit.id)
                        if item:
                            item[1].setText(str(to_add))
                        self.__on_graph_change()
                    else:
                        logger.error(f"Failed to replace {to_edit}")

            if isinstance(to_edit, Delay):
                JRiverDelayDialog(self, self.dsp.channel_names(output=True), __on_save, vals[0]).exec()
            elif isinstance(to_edit, Polarity):
                JRiverChannelOnlyFilterDialog(self, self.dsp.channel_names(output=True), __on_save, vals[0]).exec()
            elif isinstance(to_edit, Mix):
                JRiverMixFilterDialog(self, self.dsp.channel_names(output=True), __on_save, to_edit.mix_type,
                                      vals[0]).exec()
            elif isinstance(to_edit, Gain):
                JRiverGainDialog(self, self.dsp.channel_names(output=True), __on_save, vals[0]).exec()
            else:
                return False
            return True
        else:
            return False

    def __start_peq_edit_session(self, filters: Optional[List[SOS]], channel: str, node_chain: List[str],
                                 insert_at: int, selected_filter_idx: int = -1) -> None:
        '''
        Starts a PEQ editing session.
        :param filters: the filters being edited.
        :param channel: the channel the filters are on.
        :param node_chain: the nodes which provide these filters.
        :param insert_at: the index to insert at.
        :param selected_filter_idx: the filter, if any, which the user selected to start this edit session.
        '''
        filter_model = FilterModel(None, self.prefs)
        sorted_filters = [self.__enforce_filter_order(i, f) for i, f in enumerate(filters)] if filters else None
        filter_model.filter = CompleteFilter(fs=JRIVER_FS, filters=sorted_filters, sort_by_id=True, description=channel)
        self.dsp.active_graph.start_edit(channel, node_chain, insert_at)

        def __on_save():
            if self.dsp.active_graph.end_edit(filter_model.filter):
                self.show_filters()

        x_lim = (self.__magnitude_model.limits.x_min, self.__magnitude_model.limits.x_max)
        # TODO fix var_q filters + check behaviour in jriver
        FilterDialog(self.prefs, make_dirac_pulse(channel), filter_model, __on_save, parent=self,
                     selected_filter=sorted_filters[selected_filter_idx] if selected_filter_idx > -1 else None,
                     x_lim=x_lim, allow_var_q_pass=True, window_title=f"Edit Filter - {channel}").show()

    @staticmethod
    def __enforce_filter_order(index: int, f: SOS) -> SOS:
        f.id = index
        return f

    def __show_dot_dialog(self):
        '''
        Shows the underlying graphviz spec in a dialog for debugging.
        '''
        if self.__current_dot_txt:
            def on_change(txt):
                self.__current_dot_txt = txt
                self.__gen_svg()

            JRiverFilterPipelineDialog(self.__current_dot_txt, on_change, self).show()

    def __regen(self):
        '''
        Regenerates the graphviz display.
        '''
        self.clearFiltersButton.setEnabled(len(self.dsp.active_graph.filters) > 0)
        self.__current_dot_txt = self.dsp.as_dot(self.blockSelector.currentIndex(),
                                                 vertical=self.direction.isChecked(),
                                                 selected_nodes=self.__selected_node_names)
        from graphviz import ExecutableNotFound
        try:
            self.__gen_svg()
        except ExecutableNotFound as enf:
            if not self.__ignore_gz_not_installed:
                self.__ignore_gz_not_installed = True
                msg_box = QMessageBox()
                msg_box.setText(f"Please install graphviz using your system package manager.")
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle('Graphviz is not installed')
                msg_box.exec()
        except Exception as e:
            logger.exception(f"Failed to render {self.__current_dot_txt}")
            msg_box = QMessageBox()
            msg_box.setText(f"Invalid rendering ")
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle('Unable to render graph')
            msg_box.exec()

    def __gen_svg(self):
        '''
        Updates the svg viewer.
        '''
        self.pipelineView.render_bytes(render_dot(self.__current_dot_txt))

    def __on_node_click(self, node_name: str) -> None:
        '''
        Selects or deselects the node.
        :param node_name: the node which received the click.
        '''
        if node_name in self.__selected_node_names:
            self.__on_node_deselect(node_name)
        else:
            self.__on_node_select(node_name)
        self.__regen()

    def __on_node_select(self, node_name: str) -> None:
        '''
        Adds the node to the selected list and ensures all other nodes on the same filter are also selected if they
        have not already been deselected.
        :param node_name: the name of the node selected.
        '''
        self.__selected_node_names.add(node_name)
        self.__highlight_nodes_for_selected_filters(node_name)

    def __on_node_deselect(self, node_name: str) -> None:
        '''
        Removes the node from the selected list and deselects the filter in the filter list if required.
        :param node_name: the name of the deselected node.
        '''
        self.__selected_node_names.remove(node_name)
        node_filter = self.dsp.active_graph.get_filter_at_node(node_name)
        if node_filter:
            if isinstance(node_filter, Sequence):
                nodes_in_filter = [n for f in flatten(node_filter) for n in f.nodes]
            else:
                nodes_in_filter = [n for n in node_filter.nodes]
            if not any(s in self.__selected_node_names for s in nodes_in_filter):
                from model.report import block_signals
                with block_signals(self.filterList):
                    for item in self.filterList.selectedItems():
                        if item.data(FILTER_ID_ROLE) == node_filter.id:
                            item.setSelected(False)
                self.__enable_edit_buttons_if_filters_selected()

    def __highlight_nodes_for_selected_filters(self, selected_node: str) -> None:
        '''
        Ensures that all nodes for selected filters are shown as selected if they have not already been deselected.
        :param selected_node: the node that was just selected.
        '''
        graph = self.dsp.active_graph
        filts = [graph.get_filter_at_node(n) for n in self.__selected_node_names if
                 graph.get_filter_at_node(n) is not None]
        if filts:
            selected_filter = graph.get_filter_at_node(selected_node)
            if selected_filter:
                i = 0
                for f in graph.filters:
                    if not isinstance(f, Divider):
                        from model.report import block_signals
                        with block_signals(self.filterList):
                            item: QListWidgetItem = self.filterList.item(i)
                            if filts and f in filts:
                                if f.id == selected_filter.id:
                                    if not isinstance(f, Mix) or f.mix_type in [MixType.ADD, MixType.SUBTRACT]:
                                        for n in f.nodes:
                                            self.__selected_node_names.add(n)
                                    item.setSelected(True)
                            else:
                                item.setSelected(False)
                        i += 1
                self.__enable_edit_buttons_if_filters_selected()
            else:
                logger.warning(f"No filter at node {selected_node}")
        else:
            self.filterList.clearSelection()

    def redraw(self):
        self.__magnitude_model.redraw()

    def __restore_geometry(self):
        ''' loads the saved window size '''
        geometry = self.prefs.get(JRIVER_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)

    def __get_data(self, mode='mag'):
        return lambda *args, **kwargs: self.get_curve_data(mode, *args, **kwargs)

    def get_curve_data(self, mode, reference=None):
        ''' preview of the filter to display on the chart '''
        result = []
        if mode == 'mag' or self.showPhase.isChecked():
            if self.dsp:
                names = [n.text() for n in self.channelList.selectedItems()]
                for signal in self.dsp.signals:
                    if signal.name in names:
                        result.append(MagnitudeData(signal.name, None, *signal.avg,
                                                    colour=get_filter_colour(len(result)),
                                                    linestyle='-' if mode == 'mag' else '--'))
        return result

    def show_limits(self):
        ''' shows the limits dialog for the filter chart. '''
        self.__magnitude_model.show_limits(parent=self)

    def show_full_range(self):
        ''' sets the limits to full range. '''
        self.__magnitude_model.show_full_range()

    def show_sub_only(self):
        ''' sets the limits to sub only. '''
        self.__magnitude_model.show_sub_only()

    def show_phase_response(self):
        self.redraw()

    def closeEvent(self, event: QCloseEvent):
        ''' Stores the window size on close. '''
        self.prefs.set(JRIVER_GEOMETRY, self.saveGeometry())
        super().closeEvent(event)


class MSODialog(QDialog, Ui_msoDialog):

    def __init__(self, mso_filter: Optional[MSOFilter], on_change: Callable[[MSOFilter], None],
                 output_format: OutputFormat, parent):
        super(MSODialog, self).__init__(parent)
        self.setupUi(self)
        self.fileSelect.setIcon(qta.icon('fa5s.folder-open'))
        self.file.setReadOnly(True)
        self.status.setReadOnly(True)
        self.__output_format = output_format
        self.__mso_filter = mso_filter
        self.__on_change = on_change
        self.fileSelect.clicked.connect(self.__select_filter)
        self.__show_filters()

    def __select_filter(self):
        selected = QFileDialog.getOpenFileName(parent=self, caption='Import MSO Filters', filter='Filter (*.json)')
        if selected is not None and len(selected[0]) > 0:
            txt = Path(selected[0]).read_text()
            self.file.setText(selected[0])
            self.__mso_filter = from_mso(txt)
            self.__show_filters()

    def __show_filters(self):
        self.filterList.clear()
        self.filterStatus.setIcon(QIcon())
        reason: str = 'No Filter'
        if self.__mso_filter:
            for f in self.__mso_filter:
                self.filterList.addItem(str(f))
            if self.__mso_filter.filters:
                bad_filters = [f for f in self.__mso_filter if not self.__output_format.has_channels(f.channels)]
                if bad_filters:
                    bad_channels = ', '.join(sorted(list(set(c for f in bad_filters for c in f.channel_names))))
                    reason = f"Unsupported channels for {self.__output_format.display_name} - {bad_channels}"
                else:
                    reason = ''
        if reason:
            self.status.setText(reason)
            self.filterStatus.setIcon(qta.icon('fa5s.times', color='red'))
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)
        else:
            self.status.clear()
            self.filterStatus.setIcon(qta.icon('fa5s.check', color='green'))
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def accept(self):
        self.__on_change(self.__mso_filter)
        QDialog.accept(self)


class JRiverFilterPipelineDialog(QDialog, Ui_jriverGraphDialog):

    def __init__(self, dot: str, on_change, parent):
        super(JRiverFilterPipelineDialog, self).__init__(parent)
        self.setupUi(self)
        self.source.setPlainText(dot)
        if on_change:
            self.source.textChanged.connect(lambda: on_change(self.source.toPlainText()))


class JRiverGainDialog(QDialog, Ui_jriverGainDialog):

    def __init__(self, parent: QDialog, channels: List[str], on_save: Callable[[Dict[str, str]], None],
                 vals: Dict[str, str] = None):
        super(JRiverGainDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('Gain')
        self.__validators: List[Callable[[], bool]] = []
        self.__on_save = on_save
        self.__vals = vals if vals else {}
        self.__configure_channel_list(channels)
        self.gain.valueChanged.connect(lambda v: self.__set_val('Gain', f"{v:.7g}"))
        self.gain.valueChanged.connect(self.__enable_accept)
        self.__validators.append(lambda: not math.isclose(self.gain.value(), 0.0))
        if 'Gain' in self.__vals:
            self.gain.setValue(locale.atof(self.__vals['Gain']))
        self.gain.setFocus()
        self.__enable_accept()

    def accept(self):
        self.__on_save(self.__vals)
        super().accept()

    def __set_val(self, key, val):
        self.__vals[key] = val

    def __configure_channel_list(self, channels):
        for c in channels:
            self.channelList.addItem(c)
        self.channelList.itemSelectionChanged.connect(self.__enable_accept)
        if 'Channels' in self.__vals and self.__vals['Channels']:
            selected = [get_channel_name(int(i)) for i in self.__vals['Channels'].split(';')]
            for c in selected:
                item: QListWidgetItem
                for item in self.channelList.findItems(c, Qt.MatchFlag.MatchCaseSensitive):
                    item.setSelected(True)
        self.__validators.append(lambda: len(self.channelList.selectedItems()) > 0)

        def __as_channel_indexes():
            return ';'.join([str(get_channel_idx(c.text())) for c in self.channelList.selectedItems()])

        self.channelList.itemSelectionChanged.connect(lambda: self.__set_val('Channels', __as_channel_indexes()))

    def __enable_accept(self):
        self.buttonBox.button(QDialogButtonBox.Save).setEnabled(all(v() for v in self.__validators))


class JRiverDelayDialog(QDialog, Ui_jriverDelayDialog):

    def __init__(self, parent: QDialog, channels: List[str], on_save: Callable[[Dict[str, str]], None],
                 vals: Dict[str, str] = None):
        super(JRiverDelayDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('Delay')
        self.changeUnitButton.setIcon(qta.icon('fa5s.exchange-alt'))
        self.changeUnitButton.clicked.connect(self.__toggle_active_unit)
        self.__validators: List[Callable[[], bool]] = []
        self.__on_save = on_save
        self.__vals = vals if vals else {}
        self.__configure_channel_list(channels)
        self.millis.valueChanged.connect(lambda v: self.__set_val('Delay', f"{v:.7g}"))
        self.distance.valueChanged.connect(lambda v: self.__set_val('Delay', f"{self.__to_millis(v):.7g}"))
        self.millis.valueChanged.connect(self.__enable_accept)
        self.distance.valueChanged.connect(self.__enable_accept)
        self.millis.valueChanged.connect(lambda v: self.distance.setValue(self.__to_metres(v)))
        self.distance.valueChanged.connect(lambda v: self.millis.setValue(self.__to_millis(v)))
        self.__validators.append(lambda: not math.isclose(self.millis.value(), 0.0))
        self.distance.blockSignals(True)
        if 'Delay' in self.__vals:
            self.millis.setValue(locale.atof(self.__vals['Delay']))
        self.millis.setFocus()
        self.__enable_accept()

    @staticmethod
    def __to_millis(d: float):
        return d / 343.0 * 1000.0

    @staticmethod
    def __to_metres(d: float):
        return d / 1000.0 * 343.0

    def __toggle_active_unit(self):
        enable = self.millis.isEnabled()
        self.millis.setEnabled(not enable)
        self.millis.blockSignals(enable)
        self.distance.setEnabled(enable)
        self.distance.blockSignals(not enable)

    def accept(self):
        self.__on_save(self.__vals)
        super().accept()

    def __set_val(self, key, val):
        self.__vals[key] = val

    def __configure_channel_list(self, channels):
        for c in channels:
            self.channelList.addItem(c)
        self.channelList.itemSelectionChanged.connect(self.__enable_accept)
        if 'Channels' in self.__vals and self.__vals['Channels']:
            selected = [get_channel_name(int(i)) for i in self.__vals['Channels'].split(';')]
            for c in selected:
                item: QListWidgetItem
                for item in self.channelList.findItems(c, Qt.MatchFlag.MatchCaseSensitive):
                    item.setSelected(True)
        self.__validators.append(lambda: len(self.channelList.selectedItems()) > 0)

        def __as_channel_indexes():
            return ';'.join([str(get_channel_idx(c.text())) for c in self.channelList.selectedItems()])

        self.channelList.itemSelectionChanged.connect(lambda: self.__set_val('Channels', __as_channel_indexes()))

    def __enable_accept(self):
        self.buttonBox.button(QDialogButtonBox.Save).setEnabled(all(v() for v in self.__validators))


class JRiverMixFilterDialog(QDialog, Ui_jriverMixDialog):

    def __init__(self, parent: QDialog, channels: List[str], on_save: Callable[[Dict[str, str]], None],
                 mix_type: MixType, vals: Dict[str, str] = None):
        super(JRiverMixFilterDialog, self).__init__(parent)
        self.setupUi(self)
        self.__validators: List[Callable[[], bool]] = []
        self.__on_save = on_save
        self.__vals = vals if vals else {
            'Source': str(get_channel_idx(channels[0])),
            'Destination': str(get_channel_idx(channels[0])),
            'Gain': '0'
        }
        self.__vals['Mode'] = str(mix_type.value)
        self.setWindowTitle(f"{mix_type.name.capitalize()}")
        self.__configure_channel_list(channels)
        self.gain.valueChanged.connect(lambda v: self.__set_val('Gain', f"{v:.7g}"))
        self.source.currentTextChanged.connect(lambda v: self.__set_val('Source', str(get_channel_idx(v))))
        self.destination.currentTextChanged.connect(lambda v: self.__set_val('Destination', str(get_channel_idx(v))))
        self.__enable_accept()

    def accept(self):
        self.__on_save(self.__vals)
        super().accept()

    def __set_val(self, key, val):
        self.__vals[key] = val

    def __configure_channel_list(self, channels):
        for c in channels:
            self.source.addItem(c)
            self.destination.addItem(c)
        self.source.currentTextChanged.connect(self.__enable_accept)
        self.destination.currentTextChanged.connect(self.__enable_accept)
        if 'Source' in self.__vals:
            self.source.setCurrentText(get_channel_name(int(self.__vals['Source'])))
        if 'Destination' in self.__vals:
            self.destination.setCurrentText(get_channel_name(int(self.__vals['Destination'])))
        if 'Gain' in self.__vals:
            self.gain.setValue(locale.atof(self.__vals['Gain']))
        self.__validators.append(lambda: self.source.currentText() != self.destination.currentText())

    def __enable_accept(self):
        self.buttonBox.button(QDialogButtonBox.Save).setEnabled(all(v() for v in self.__validators))


class JRiverChannelOnlyFilterDialog(QDialog, Ui_channelSelectDialog):

    def __init__(self, parent: QDialog, channels: List[str], on_save: Callable[[Dict[str, str]], None],
                 vals: Dict[str, str] = None, title: str = 'Polarity', multi: bool = True):
        super(JRiverChannelOnlyFilterDialog, self).__init__(parent)
        self.setupUi(self)
        self.lfeChannel.setVisible(False)
        self.lfeChannelLabel.setVisible(False)
        if multi:
            self.channelList.setSelectionMode(QAbstractItemView.MultiSelection)
        else:
            self.channelList.setSelectionMode(QAbstractItemView.SingleSelection)
        self.__validators: List[Callable[[], bool]] = []
        self.__on_save = on_save
        self.__vals = vals if vals else {}
        self.setWindowTitle(title)
        self.__configure_channel_list(channels)
        self.__enable_accept()

    def accept(self):
        self.__on_save(self.__vals)
        super().accept()

    def __set_val(self, key, val):
        self.__vals[key] = val

    def __configure_channel_list(self, channels):
        for c in channels:
            self.channelList.addItem(c)
        self.channelList.itemSelectionChanged.connect(self.__enable_accept)
        if 'Channels' in self.__vals and self.__vals['Channels']:
            selected = [get_channel_name(int(i)) for i in self.__vals['Channels'].split(';')]
            for c in selected:
                item: QListWidgetItem
                for item in self.channelList.findItems(c, Qt.MatchCaseSensitive):
                    item.setSelected(True)
        self.__validators.append(lambda: len(self.channelList.selectedItems()) > 0)

        def __as_channel_indexes():
            return ';'.join([str(get_channel_idx(c.text())) for c in self.channelList.selectedItems()])

        self.channelList.itemSelectionChanged.connect(lambda: self.__set_val('Channels', __as_channel_indexes()))

    def __enable_accept(self):
        self.buttonBox.button(QDialogButtonBox.Save).setEnabled(all(v() for v in self.__validators))


class ShowFiltersDialog(QDialog, Ui_xoFiltersDialog):

    def __init__(self, parent, filt: CompoundRoutingFilter):
        super(ShowFiltersDialog, self).__init__(parent)
        self.setupUi(self)
        self.f = filt
        from model.report import block_signals
        with block_signals(self.selector):
            self.selector.addItem('Show All')
            if filt.routing:
                self.selector.addItem('Gain/Routing')
            if filt.delays:
                self.selector.addItem('Delay Normalisation')
            for f in filt.xo:
                self.selector.addItem(f'Channel {f.input_channel}')
        self.update_list('Show All')

    def update_list(self, txt: str):
        self.filters.clear()
        for f in flatten(self.f.routing):
            if txt == 'Show All' or txt == 'Gain/Routing':
                self.filters.addItem(str(f))
        for f in flatten(self.f.delays):
            if txt == 'Show All' or txt == 'Delay Normalisation':
                self.filters.addItem(str(f))
        for xo in self.f.xo:
            added_spacer = False
            if txt == 'Show All':
                if self.filters.count() > 0:
                    self.filters.addItem('')
                    added_spacer = True
                self.filters.addItem(f'*** Channel {xo.input_channel} ***')
            if txt == 'Show All' or txt[8:] == xo.input_channel:
                if self.filters.count() > 0 and not added_spacer:
                    self.filters.addItem('')
                for f in flatten(xo.filters):
                    self.filters.addItem(str(f))


class XODialog(QDialog, Ui_xoDialog):

    def __init__(self, parent, prefs: Preferences, input_channels: List[str], output_channels: List[str],
                 output_format: OutputFormat, on_save: Callable[[CompoundRoutingFilter], None],
                 existing: CompoundRoutingFilter = None, **kwargs):
        super(XODialog, self).__init__(parent)
        self.__matrix: Matrix = None
        self.__on_save = on_save
        self.__output_format = output_format
        channels_sw_first = sorted(input_channels, key=lambda c: get_channel_idx(c) if c != 'SW' else -1)
        self.__channel_groups = {c: [c] for c in channels_sw_first}
        self.__output_channels = output_channels
        self.__mag_update_timer = QTimer(self)
        self.__mag_update_timer.setSingleShot(True)
        self.prefs = prefs
        self.setupUi(self)
        self.showMatrixButton.setIcon(qta.icon('fa5s.route'))
        if self.__output_format.lfe_channels > 0:
            for c in input_channels:
                self.lfeChannelSelector.addItem(c)
            self.__lfe_channel: Optional[str] = 'SW'
            self.lfeChannelSelector.setCurrentText('SW')
            self.lfeChannelSelector.currentTextChanged.connect(self.__set_lfe_channel)
        else:
            self.lfeChannelSelector.setVisible(False)
            self.lfeChannelSelectorLabel.setVisible(False)
            self.lfeAdjust.setVisible(False)
            self.lfeAdjustLabel.setVisible(False)
            self.__lfe_channel: Optional[str] = None
        self.linkChannelsButton.clicked.connect(self.__show_group_channels_dialog)
        self.__editors = []
        for name, channels in self.__channel_groups.items():
            self.__editors.append(ChannelEditor(self.channelsFrame, name, channels, output_format,
                                                any(c == self.__lfe_channel for c in channels), self.__trigger_redraw,
                                                self.__on_output_channel_count_change))
        last_widget = None
        for e in self.__editors:
            self.channelsLayout.addWidget(e.widget)
            self.setTabOrder(last_widget if last_widget else self.lfeAdjust, e.widget)
            last_widget = e.widget
        self.__set_matrix(self.__create_matrix())
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, prefs,
                                                self.__get_data(), 'Filter', fill_primary=False,
                                                x_min_pref_key=JRIVER_GRAPH_X_MIN, x_max_pref_key=JRIVER_GRAPH_X_MAX,
                                                x_scale_pref_key=None, secondary_data_provider=self.__get_data('phase'),
                                                secondary_name='Phase', secondary_prefix='deg', fill_secondary=False,
                                                y_range_calc=DecibelRangeCalculator(60),
                                                y2_range_calc=PhaseRangeCalculator(), show_y2_in_legend=False, **kwargs)
        self.limitsButton.setIcon(qta.icon('fa5s.arrows-alt'))
        self.limitsButton.setToolTip('Set graph axis limits')
        self.limitsButton.clicked.connect(self.show_limits)
        self.showPhase.setIcon(qta.icon('mdi.cosine-wave'))
        self.showPhase.toggled.connect(self.__trigger_redraw)
        self.showPhase.setToolTip('Display phase response')
        self.showFiltersButton.setIcon(qta.icon('fa5s.info-circle'))
        self.showFiltersButton.setToolTip('Show All Filters')
        self.showFiltersButton.clicked.connect(self.__show_filters_dialog)
        if output_format.lfe_channels == 0:
            self.lfeAdjust.setVisible(False)
            self.lfeAdjustLabel.setVisible(False)
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

    def __show_filters_dialog(self):
        ShowFiltersDialog(self, self.__make_filters()).show()

    def __load_filter(self):
        metadata = json.loads(self.__existing.metadata())
        if LFE_ADJUST_KEY in metadata:
            self.lfeAdjust.setValue(-metadata[LFE_ADJUST_KEY])
        if EDITORS_KEY in metadata:
            groups = {e[EDITOR_NAME_KEY]: e[UNDERLYING_KEY] for e in metadata[EDITORS_KEY]}
            # TODO warn if the groups are not in the inputs
            self.__reconfigure_channel_groups(groups)
            for e in metadata[EDITORS_KEY]:
                editor: ChannelEditor = next((c for c in self.__editors if c.name == e[EDITOR_NAME_KEY]))
                editor.ways = e[WAYS_KEY]
                editor.symmetric = e[SYM_KEY]
        if LFE_IN_KEY in metadata:
            self.__lfe_channel = get_channel_name(metadata[LFE_IN_KEY])
        if ROUTING_KEY in metadata:
            self.__matrix.decode(metadata[ROUTING_KEY])
            self.__on_matrix_update()
        if EDITORS_KEY in metadata:
            groups = {e[EDITOR_NAME_KEY]: e[UNDERLYING_KEY] for e in metadata[EDITORS_KEY]}
            for f in self.__existing.filters:
                if isinstance(f, XOFilter):
                    match = next(e for e in self.__editors
                                 if e.name in groups.keys() and f.input_channel in e.underlying_channels and e.visible)
                    match.load_legacy_filter(f)
                elif isinstance(f, MultiwayFilter):
                    match = next(e for e in self.__editors
                                 if e.name in groups.keys() and f.input_channel in e.underlying_channels and e.visible)
                    match.load_filter(f)
        for e in self.__editors:
            e.resize_mds()

    def __calculate_sw_channel(self) -> str:
        return None if self.__output_format.lfe_channels == 0 else 'SW'

    def accepted(self):
        super().accepted()

    def __create_matrix(self):
        matrix = Matrix({c: len(e) for e in self.__editors for c in e.underlying_channels},
                        [c for c in self.__output_channels])
        self.__initialise_routing(matrix)
        return matrix

    def __initialise_routing(self, matrix: Matrix):
        output_channels = [matrix.column_name(i - 1) for i in range(matrix.columns, 0, -1)]
        input_channels = {c for e in self.__editors for c in e.underlying_channels}
        output_channels = [c for c in output_channels if c not in input_channels]
        for e in self.__editors:
            if e.visible:
                for i, c in enumerate(e.underlying_channels):
                    way_count = len(e)
                    for w in range(way_count):
                        if w == 0:
                            if self.__output_format.lfe_channels:
                                matrix.enable(c, w, self.__lfe_channel)
                            else:
                                matrix.enable(c, w, c)
                        elif w == 1 and self.__output_format.lfe_channels:
                            matrix.enable(c, w, c)
                        else:
                            try:
                                matrix.enable(c, w, output_channels.pop())
                            except IndexError:
                                raise ImpossibleRoutingError(
                                    f"Channel {c} Way {w + 1} has no output channel to route to")

    def __show_group_channels_dialog(self):
        GroupChannelsDialog(self, {e.name: e.underlying_channels for e in self.__editors
                                   if e.visible and e.name != self.__lfe_channel},
                            self.__reconfigure_channel_groups).exec()

    def __set_lfe_channel(self, lfe_channel: str) -> None:
        '''
        Allows user to specify the current LFE input channel
        '''
        if lfe_channel != self.__lfe_channel:
            self.__lfe_channel = lfe_channel
            self.__set_matrix(self.__create_matrix())

    def __reconfigure_channel_groups(self, grouped_channels: Dict[str, List[str]]):
        '''
        Reconfigures the UI to show the new channel groups.
        :param grouped_channels: the grouped channels.
        '''
        new_widget_added = False
        old_matrix = self.__matrix
        self.__matrix = None
        old_groups = self.__channel_groups
        if self.__lfe_channel:
            grouped_channels[self.__lfe_channel] = [self.__lfe_channel]
        self.__channel_groups = grouped_channels
        for name, channels in grouped_channels.items():
            matching_editor: ChannelEditor = next((e for e in self.__editors if e.name == name), None)
            if matching_editor:
                if matching_editor.underlying_channels != channels:
                    matching_editor.underlying_channels = channels
                    old_matrix = None
                matching_editor.show()
            else:
                new_editor = ChannelEditor(self.channelsFrame, name, channels, self.__output_format,
                                           any(c == self.__lfe_channel for c in channels),
                                           self.__trigger_redraw, self.__on_output_channel_count_change)
                self.__editors.append(new_editor)
                new_widget_added = True
                old_matrix = None
        for name, channels in old_groups.items():
            if name not in grouped_channels.keys():
                matching_editor: ChannelEditor = next((e for e in self.__editors if e.name == name), None)
                if matching_editor:
                    matching_editor.hide()
                    old_matrix = None
        if new_widget_added:
            for e in self.__editors:
                self.channelsLayout.removeWidget(e.widget)
            sw_first_editors = sorted(self.__editors, key=lambda e: -1 if 'SW' in e.underlying_channels else min(
                get_channel_idx(c) for c in e.underlying_channels))
            for e in sw_first_editors:
                self.channelsLayout.addWidget(e.widget)
                if e.visible:
                    e.show()
                else:
                    e.hide()
        # TODO don't completely reinit it, just update anything that has changed
        self.__set_matrix(self.__create_matrix() if not old_matrix else old_matrix)

    def __on_output_channel_count_change(self, channel: str, ways: int):
        if self.__matrix:
            self.__matrix.resize(channel, ways)

    def show_matrix(self):
        MatrixDialog(self, self.__matrix, self.__set_matrix, self.__create_matrix).show()

    def __set_matrix(self, matrix: Matrix):
        self.__matrix = matrix
        self.__on_matrix_update()

    def __on_matrix_update(self):
        for e in self.__editors:
            e.update_output_channels(self.__matrix.get_output_channels_for(e.underlying_channels))

    def accept(self):
        self.__on_save(self.__make_filters())
        self.prefs.set(XO_GEOMETRY, self.saveGeometry())
        super().accept()

    def reject(self):
        self.prefs.set(XO_GEOMETRY, self.saveGeometry())
        super().reject()

    def __get_lfe_metadata(self) -> Tuple[int, int, int]:
        '''
        Determines the gain adjustments required for bass management and the channels involved.
        :return: input lfe channel index, lfe gain adjustment, main channel gain adjustment.
        '''
        lfe_adjust = 0
        if self.__output_format.lfe_channels > 0:
            lfe_adjust = -(self.lfeAdjust.value())
        if lfe_adjust == 0:
            main_adjust = 0
        else:
            main_adjust = lfe_adjust - 10
        if self.__lfe_channel:
            lfe_channel_idx = get_channel_idx(self.__lfe_channel)
        else:
            lfe_channel_idx = None
        return lfe_channel_idx, lfe_adjust, main_adjust

    def __make_filters(self) -> CompoundRoutingFilter:
        '''
        Creates the routing filter.
        :return: the filter.
        '''
        lfe_channel_idx, lfe_adjust, main_adjust = self.__get_lfe_metadata()
        mds_channels = [e for e in self.__editors if e.visible if not math.isclose(e.xo_induced_delay, 0.0)]
        _, summed_routes = group_routes_by_output_channel(self.__matrix.active_routes)
        summed_routes_use_mds = False
        if mds_channels and summed_routes:
            # see if mds output is found
            pass

        xo_filters: List[MultiwayFilter] = []
        xo_induced_delay: Dict[float, List[str]] = defaultdict(list)
        for e in self.__editors:
            if e.visible:
                xo_induced_delay[e.xo_induced_delay].extend(list({c for e in e.output_channels.values() for c in e}))
                for c in e.underlying_channels:
                    # TODO if MDS and summing to an output channel, change the output to free channel & then add
                    xo_filters.append(MultiwayFilter(c, e.output_channels[c], e.filters[c], e.meta))
        editor_meta = [
            {EDITOR_NAME_KEY: e.name, UNDERLYING_KEY: e.underlying_channels, WAYS_KEY: len(e), SYM_KEY: e.symmetric}
            for e in self.__editors if e.visible
        ]
        return calculate_compound_routing_filter(self.__matrix, editor_meta, xo_induced_delay, xo_filters, main_adjust,
                                                 lfe_adjust, lfe_channel_idx)

    def show_limits(self):
        ''' shows the limits dialog for the xo filter chart. '''
        self.__magnitude_model.show_limits(parent=self)

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
                signals = editor.impulses
                if mode == 'mag':
                    summed = None
                    for i in signals:
                        if i:
                            result.append(MagnitudeData(i.name, None, *i.avg, colour=get_filter_colour(len(result)),
                                                        linestyle='-'))
                            if summed is None:
                                summed = i
                            else:
                                summed = summed.add(i)
                    if summed and len(signals) > 1:
                        result.append(MagnitudeData(f"sum {editor.name}", None, *summed.avg,
                                                    colour=get_filter_colour(len(result)),
                                                    linestyle='-' if mode == 'mag' else '--'))
                elif mode == 'phase' and self.showPhase.isChecked():
                    pass
                    # TODO remove?
                    # for pr_mag in editor.phase_responses:
                    #     pr_mag.colour = get_filter_colour(len(result) + extra)
                    #     pr_mag.linestyle = '--'
                    #     result.append(pr_mag)
                extra += 1
        return result

    def __restore_geometry(self):
        ''' loads the saved window size '''
        geometry = self.prefs.get(XO_GEOMETRY)
        if geometry is not None:
            self.restoreGeometry(geometry)


class ChannelEditor:

    def __init__(self, channels_frame: QWidget, name: str, underlying_channels: List[str],
                 output_format: OutputFormat, is_sw_channel: bool, on_filter_change: Callable[[], None],
                 on_way_count_change: Callable[[str, int], None], mds_points: List[Tuple[int, int, float]] = None):
        self.__xos: Dict[str, MultiwayCrossover] = {}
        self.__meta: dict = {}
        self.__notify_parent = on_filter_change
        self.__is_sw_channel = is_sw_channel
        self.__output_format = output_format
        self.__output_channels_by_underlying: Dict[str, List[Tuple[int, List[str]]]] = {}
        self.__mds_points: List[Tuple[int, int, float]] = [] if mds_points is None else mds_points
        self.__mds_xos: List[Optional[MDSXO]] = []
        self.__name = name
        self.__underlying_channels = underlying_channels
        self.__editors: List[WayEditor] = []
        self.__visible = True
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
        self.__show_response.toggled.connect(self.__notify_parent)
        self.__symmetric = QCheckBox(self.__frame)
        self.__symmetric.setText('Symmetric XO?')
        self.__symmetric.setChecked(True)
        self.__is_symmetric = lambda: self.__symmetric.isChecked()
        self.__mds = QPushButton(self.__frame)
        self.__mds.setText('MDS')
        self.__mds.setToolTip('Configure Matched Delay Subtractive Crossover')
        self.__mds.clicked.connect(self.__design_mds)
        self.__mds.setVisible(self.__ways.value() > 1)
        self.__header_layout.addWidget(self.__channel_name_label)
        self.__header_layout.addWidget(self.__ways)
        self.__header_layout.addWidget(self.__show_response)
        self.__header_layout.addWidget(self.__symmetric)
        self.__header_layout.addWidget(self.__mds)
        self.__header_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.__layout.addLayout(self.__header_layout)
        self.__way_frame = QFrame(self.__frame)
        self.__way_layout = QHBoxLayout(self.__way_frame)
        self.__way_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.__layout.addWidget(self.__way_frame)
        self.__propagate_way_count_change = on_way_count_change
        self.__ways.valueChanged.connect(self.__on_way_count_change)
        self.__frame.setTabOrder(self.__ways, self.__show_response)
        self.__frame.setTabOrder(self.__show_response, self.__symmetric)
        if self.__is_sw_channel:
            self.__ways.hide()
            self.__update_editors()
        else:
            self.__ways.setValue(self.__calculate_ways())

    @property
    def uses_mds(self) -> bool:
        return any(x for x in self.__mds_xos if x)

    def is_sw_channel(self) -> bool:
        return self.__is_sw_channel

    @property
    def bass_managed(self) -> bool:
        return self.__editors[0].bass_managed

    def __on_way_count_change(self, v: int):
        self.__mds.setVisible(v > 1)
        self.__on_change()
        self.__update_editors()
        for c in self.__underlying_channels:
            self.__propagate_way_count_change(c, v)

    def __on_change(self):
        self.__reset()
        self.__notify_parent()

    def update_output_channels(self, channel_names: Dict[str, List[Tuple[int, List[str]]]]):
        self.__output_channels_by_underlying = channel_names
        self.__reset()

    def __reset(self):
        self.__xos = {}
        self.__meta = {}

    def __repr__(self):
        return f"ChannelEditor {self.__underlying_channels} {self.__ways.value()}"

    def __design_mds(self):
        MDSDialog(self.__frame, self.__mds_points, self.__on_mds_change, self.__ways.value() - 1).show()

    def __on_mds_change(self, ways: List[Tuple[int, int, float]]):
        self.__mds_points = ways
        self.__mds_xos = [MDSXO(w[1], w[2], calc=True, lp_channel=[], hp_channel=[]) if w[2] else None for w in ways]
        has_mds = any((x is not None for x in self.__mds_xos))
        self.__mds.setIcon(qta.icon('fa5s.check') if has_mds else QIcon())
        for i, xo in enumerate(self.__mds_xos):
            if xo:
                self.__editors[i].set_mds_low(xo.order, xo.lp_fc)
                self.__editors[i + 1].set_mds_high(xo.order, xo.lp_fc)
            else:
                self.__editors[i].clear_mds_low()
                self.__editors[i + 1].clear_mds_high()

    def __calculate_ways(self) -> int:
        of = self.__output_format
        if of.lfe_channels == 0:
            if of.input_channels == of.output_channels:
                return 1
            else:
                return int(of.output_channels / of.input_channels)
        else:
            return int((of.output_channels - of.lfe_channels) / (of.input_channels - of.lfe_channels)) + 1

    def __len__(self):
        len = self.__ways.value()
        return len

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
        return self.visible and self.__show_response.isChecked()

    @property
    def impulses(self) -> List[Signal]:
        self.__recalc()
        if self.__xos:
            return next(iter(self.__xos.values())).output
        else:
            return []

    @property
    def filters(self) -> Dict[str, List[Filter]]:
        self.__recalc()
        return {c: x.graph.filters for c, x in self.__xos.items()}

    @property
    def meta(self) -> dict:
        self.__recalc()
        return self.__meta

    @property
    def xo_induced_delay(self) -> float:
        self.__recalc()
        return next(iter(self.__xos.values())).xo_induced_delay if self.__xos else 0.0

    @property
    def output_channels(self) -> Dict[str, List[str]]:
        return {k: [c for a, b in v1 for c in b] for k, v1 in self.__output_channels_by_underlying.items()}

    def __recalc(self):
        if not self.__xos:
            self.__xos = {}
            meta_wv = None
            for in_ch, out_chs in self.__output_channels_by_underlying.items():
                ss_filter: Optional[ComplexHighPass] = None
                xos: List[XO] = []
                way_values: List[WayValues] = []
                lp_filter: Optional[ComplexLowPass] = None
                for w in range(min(self.ways, len(out_chs))):
                    e = self.__editors[w]
                    values = e.get_way_values()
                    output_channels_this_way = out_chs[values.way][1]
                    if w == 0:
                        if e.hp_filter_type:
                            ss_filter = e.hp_filter
                        if self.ways == 1:
                            xos.append(
                                StandardXO(output_channels_this_way, output_channels_this_way, low_pass=e.lp_filter))
                        else:
                            lp_filter = e.lp_filter
                    else:
                        try:
                            mds_xo = self.__mds_xos[len(xos)]
                        except IndexError as e:
                            raise e
                        output_channels_previous_way = out_chs[values.way - 1][1]
                        if mds_xo:
                            mds_xo.out_channel_lp = output_channels_previous_way
                            mds_xo.out_channel_hp = output_channels_this_way
                            xos.append(mds_xo)
                        else:
                            xos.append(StandardXO(output_channels_previous_way, output_channels_this_way,
                                                  low_pass=lp_filter, high_pass=e.hp_filter))
                        lp_filter = e.lp_filter
                    way_values.append(values)
                self.__xos[in_ch] = MultiwayCrossover(in_ch, xos, way_values, subsonic_filter=ss_filter)
                if not meta_wv:
                    meta_wv = [w.to_json() for w in way_values]
            self.__meta = {
                'w': meta_wv,
                'm': self.__mds_points,
                'h': self.uses_mds
            }

    def __update_editors(self):
        num_ways = self.__ways.value()
        for i in range(num_ways):
            if i >= len(self.__editors):
                self.__create_editor(i)
                if i > 1:
                    self.__frame.setTabOrder(self.__editors[i - 1].widget, self.__editors[i].widget)
            else:
                self.__editors[i].show()
        if num_ways < len(self.__editors):
            for i in range(num_ways, len(self.__editors)):
                self.__editors[i].hide()
        self.__symmetric.setVisible(num_ways > 1)
        for i in range(num_ways):
            self.__editors[i].can_low_pass(num_ways == 1 or i < num_ways - 1)
        self.resize_mds()

    def __create_editor(self, i: int):
        editor = WayEditor(self.__way_frame, self.__name, i, self.__on_change, self.__propagate_symmetric_filter,
                           self.__output_format.lfe_channels > 0, self.__is_sw_channel)
        self.__editors.append(editor)
        self.__way_layout.insertWidget(i, editor.widget)

    def __propagate_symmetric_filter(self, way: int, filter_type: str, freq: float, order: int):
        if self.__is_symmetric():
            if way + 1 < len(self.__editors):
                self.__editors[way + 1].set_high_pass(filter_type, freq, order)

    @property
    def visible(self) -> bool:
        return self.__visible

    def show(self) -> None:
        self.__visible = True
        self.__frame.show()

    def hide(self) -> None:
        self.__visible = False
        self.__frame.hide()

    @property
    def widget(self) -> QWidget:
        return self.__frame

    def load_legacy_filter(self, f: XOFilter):
        self.__editors[f.way].load_legacy_filter(f)

    def load_filter(self, f: MultiwayFilter):
        '''
        Loads the filter into the specified way.
        :param f: the filter.
        '''
        way_values = {w.way: w for w in [WayValues.from_json(json.loads(w)) for w in f.ui_meta['w']]}
        for i, e in enumerate(self.__editors):
            if e.visible:
                e.load_way_values(way_values[i])
        self.__on_mds_change(f.ui_meta['m'])
        self.resize_mds()

    def resize_mds(self):
        delta = len(self.__mds_xos) - self.ways
        if delta > 0:
            self.__mds_xos = self.__mds_xos[0: self.ways]
        elif delta < -1:
            self.__mds_xos.extend([None] * (abs(delta) - 1))
        delta = len(self.__mds_points) - self.ways
        if delta > 0:
            self.__mds_points = self.__mds_points[0: self.ways]
        elif delta < -1:
            pts = len(self.__mds_points)
            self.__mds_points.extend([(pts + i, 4, 0.0) for i in range(abs(delta) - 1)])


class WayEditor:

    def __init__(self, parent: QWidget, channel: str, way: int, on_change: Callable[[], None],
                 propagate_low_pass_change: Callable[[int, str, float, int], None], allow_sw: bool, is_sw: bool):
        self.__way = way
        self.__visible = True
        self.__propagate_low_pass = propagate_low_pass_change
        self.__channel = channel
        self.__notify_parent = on_change
        self.__frame = QFrame(parent)
        self.__frame.setFrameShape(QFrame.StyledPanel)
        self.__frame.setFrameShadow(QFrame.Raised)
        self.__layout = QGridLayout(self.__frame)
        self.__bass_managed = False
        # header
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
        self.__hp_label.setText('High Pass' if way > 0 else 'Subsonic Filter')
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

    def can_low_pass(self, enabled: bool):
        self.__lp_label.setVisible(enabled)
        self.__lp_freq.setVisible(enabled)
        self.__lp_order.setVisible(enabled)
        self.__lp_filter_type.setVisible(enabled)

    @property
    def lp_order(self) -> int:
        return self.__lp_order.value()

    @property
    def lp_fc(self) -> float:
        return self.__lp_freq.value()

    @property
    def lp_filter_type(self) -> Optional[FilterType]:
        value = self.__lp_filter_type.currentText()
        if value:
            return FilterType.value_of(value)
        return None

    @property
    def hp_order(self) -> int:
        return self.__hp_order.value()

    @property
    def hp_fc(self) -> float:
        return self.__hp_freq.value()

    @property
    def hp_filter_type(self) -> Optional[FilterType]:
        value = self.__hp_filter_type.currentText()
        if value:
            return FilterType.value_of(value)
        return None

    @property
    def hp_filter(self) -> Optional[ComplexHighPass]:
        if self.hp_filter_type:
            return ComplexHighPass(self.hp_filter_type, self.hp_order, JRIVER_FS, self.hp_fc, q=DEFAULT_Q)
        return None

    @property
    def lp_filter(self) -> Optional[ComplexLowPass]:
        if self.lp_filter_type:
            return ComplexLowPass(self.lp_filter_type, self.lp_order, JRIVER_FS, self.lp_fc, q=DEFAULT_Q)
        return None

    def get_way_values(self) -> WayValues:
        return WayValues(self.__way, self.__delay.value(), self.__gain.value(), self.inverted,
                         lp=[self.__lp_filter_type.currentText(), self.__lp_order.value(), self.__lp_freq.value()],
                         hp=[self.__hp_filter_type.currentText(), self.__hp_order.value(), self.__hp_freq.value()])

    def set_mds_low(self, order: int, freq: float) -> None:
        from model.report import block_signals
        with block_signals(self.__lp_filter_type):
            self.__lp_filter_type.setCurrentText(FilterType.BESSEL_MAG6.display_name)
            self.__lp_filter_type.setEnabled(False)
        with block_signals(self.__lp_order):
            self.__lp_order.setValue(order)
            self.__lp_order.setEnabled(False)
        with block_signals(self.__lp_freq):
            self.__lp_freq.setValue(freq)
            self.__lp_freq.setEnabled(False)
        self.__on_value_change()

    def clear_mds_low(self) -> None:
        self.__lp_filter_type.setEnabled(True)
        self.__lp_order.setEnabled(True)
        self.__lp_freq.setEnabled(True)

    def set_mds_high(self, order: int, freq: float) -> None:
        from model.report import block_signals
        with block_signals(self.__hp_filter_type):
            self.__hp_filter_type.setCurrentText(FilterType.BESSEL_MAG6.display_name)
            self.__hp_filter_type.setEnabled(False)
        with block_signals(self.__hp_order):
            self.__hp_order.setValue(order)
            self.__hp_order.setEnabled(False)
        with block_signals(self.__hp_freq):
            self.__hp_freq.setValue(freq)
            self.__hp_freq.setEnabled(False)
        self.__on_value_change()

    def clear_mds_high(self) -> None:
        self.__hp_filter_type.setEnabled(True)
        self.__hp_order.setEnabled(True)
        self.__hp_freq.setEnabled(True)

    def __on_lp_change(self):
        self.__propagate_low_pass(self.__way, self.__lp_filter_type.currentText(), self.__lp_freq.value(),
                                  self.__lp_order.value())
        self.__on_value_change()

    @property
    def bass_managed(self) -> bool:
        return self.__bass_managed

    @bass_managed.setter
    def bass_managed(self, bass_managed: bool):
        self.__bass_managed = bass_managed

    @property
    def inverted(self) -> bool:
        return self.__invert.isChecked()

    def __on_value_change(self):
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
        self.__visible = True
        self.__frame.show()

    def hide(self) -> None:
        self.__visible = False
        self.__frame.hide()

    @property
    def visible(self) -> bool:
        return self.__visible

    @visible.setter
    def visible(self, visible: bool):
        if visible:
            self.show()
        else:
            self.hide()

    @property
    def widget(self) -> QWidget:
        return self.__frame

    def set_high_pass(self, filter_type: str, freq: float, order: int):
        self.__hp_order.setValue(order)
        self.__hp_freq.setValue(freq)
        self.__hp_filter_type.setCurrentText(filter_type)

    def load_way_values(self, values: WayValues):
        self.__delay.setValue(values.delay_millis)
        self.__gain.setValue(values.gain)
        self.__invert.setChecked(values.inverted)
        if values.lp:
            self.__lp_filter_type.setCurrentText(values.lp[0])
            self.__lp_order.setValue(values.lp[1])
            self.__lp_freq.setValue(values.lp[2])
        if values.hp:
            self.__hp_filter_type.setCurrentText(values.hp[0])
            self.__hp_order.setValue(values.hp[1])
            self.__hp_freq.setValue(values.hp[2])

    def load_legacy_filter(self, xo: XOFilter):
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

    def __init__(self, parent: QWidget, matrix: Matrix, on_save: Callable[[Matrix], None],
                 re_init: Callable[[], Matrix]):
        super(MatrixDialog, self).__init__(parent)
        self.setupUi(self)
        self.errorMessage.setStyleSheet('color: red')
        self.__re_init = re_init
        self.buttonBox.button(QDialogButtonBox.Reset).clicked.connect(self.__reinit_propagate)
        self.__on_save = on_save
        self.__matrix = matrix.clone()
        self.__table_model = MatrixTableModel(self.__matrix, self.__on_toggle)
        self.matrix.setModel(self.__table_model)
        self.matrix.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.__table_model.delegate_to_checkbox(self.matrix)
        self.adjustSize()

    def __reinit_propagate(self):
        self.__matrix = self.__re_init()
        self.__table_model = MatrixTableModel(self.__matrix, self.__on_toggle)
        self.matrix.setModel(self.__table_model)

    def __on_toggle(self, msg: str):
        enable = True
        if not msg:
            not_routed = []
            for r in range(self.__matrix.rows):
                routed = [c for c in range(self.__matrix.columns) if self.__matrix.is_routed(r, c)]
                if not routed:
                    not_routed.append(self.__matrix.row_name(r))
            if not_routed:
                msg = f"Input{'s' if len(not_routed) > 1 else ''} not routed to any output:{','.join(not_routed)}"
                enable = False
        self.buttonBox.button(QDialogButtonBox.Save).setEnabled(enable)
        self.__set_err_msg(msg)

    def __set_err_msg(self, msg):
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
        self.channelGroups.itemSelectionChanged.connect(self.__enable_remove_button)
        self.addGroupButton.clicked.connect(self.__add_group)
        self.addGroupButton.setIcon(qta.icon('fa5s.plus'))
        self.addGroupButton.setToolTip('Create new channel group')
        self.addGroupButton.setEnabled(True)
        self.deleteGroupButton.setIcon(qta.icon('fa5s.minus'))
        self.deleteGroupButton.setToolTip('Remove selected group')
        self.linkAllButton.setIcon(qta.icon('fa5s.object-group'))
        self.linkAllButton.clicked.connect(self.__group_all)
        self.linkAllButton.setToolTip('Group all channels into a single group')
        self.deleteGroupButton.clicked.connect(self.__remove_group)

    def __group_all(self):
        self.__add_named('All', [self.channels.item(i).text() for i in range(self.channels.count())])

    def __enable_remove_button(self):
        self.deleteGroupButton.setEnabled(len(self.channelGroups.selectedItems()) > 0)

    def __enable_add_button(self):
        some_selected = len(self.channels.selectedIndexes()) > 0
        self.groupName.setEnabled(some_selected)

    def __add_group(self):
        group_name = self.groupName.text()
        selected_channels = [i.text() for i in self.channels.selectedItems()]
        if not group_name:
            group_name = ''.join(selected_channels)
        item: QListWidgetItem = QListWidgetItem(group_name)
        item.setData(self.GROUP_CHANNELS_ROLE, selected_channels)
        self.channelGroups.addItem(item)
        self.groupName.clear()
        self.__remove_selected(self.channels)

    def __add_named(self, name: str, selected: List[str]):
        if selected:
            item: QListWidgetItem = QListWidgetItem(name)
            item.setData(self.GROUP_CHANNELS_ROLE, selected)
            self.channelGroups.addItem(item)
            self.groupName.clear()
            self.__remove_named(selected, self.channels)

    def __remove_group(self):
        for i in self.channelGroups.selectedItems():
            for c in i.data(self.GROUP_CHANNELS_ROLE):
                self.channels.addItem(c)
        self.__remove_selected(self.channelGroups)

    @staticmethod
    def __remove_named(named: List[str], widget: QListWidget):
        for i in reversed(range(widget.count())):
            if widget.item(i).text() in named:
                widget.takeItem(i)

    @staticmethod
    def __remove_selected(widget: QListWidget):
        for i in widget.selectedItems():
            widget.takeItem(widget.indexFromItem(i).row())

    def accept(self):
        groups = {self.channelGroups.item(i).text(): self.channelGroups.item(i).data(self.GROUP_CHANNELS_ROLE) for i in
                  range(self.channelGroups.count())}
        individuals = {self.channels.item(i).text(): [self.channels.item(i).text()] for i in
                       range(self.channels.count())}
        self.__on_save({**groups, **individuals})
        super().accept()


class SWChannelSelectorDialog(QDialog, Ui_channelSelectDialog):

    def __init__(self, parent: QDialog, channels: List[str], lfe_channel: str, sw_channels: List[str],
                 on_save: Callable[[str, List[str]], None]):
        super(SWChannelSelectorDialog, self).__init__(parent)
        self.setupUi(self)
        self.channelList.setSelectionMode(QAbstractItemView.MultiSelection)
        self.__on_save = on_save
        self.setWindowTitle('Set LFE/SW Channels')
        for c in channels:
            self.channelList.addItem(c)
            self.lfeChannel.addItem(c)
        self.lfeChannel.setCurrentText(lfe_channel)
        for c in sw_channels:
            item: QListWidgetItem
            for item in self.channelList.findItems(c, Qt.MatchCaseSensitive):
                item.setSelected(True)

    def accept(self):
        self.__on_save(self.lfeChannel.currentText(), [i.text() for i in self.channelList.selectedItems()])
        super().accept()


class MDSDialog(QDialog, Ui_mdsDialog):
    ORDER_ROLE = Qt.UserRole + 1
    FREQ_ROLE = Qt.UserRole + 2

    def __init__(self, parent: QWidget, current: List[Tuple[int, int, float]],
                 on_update: Callable[[List[Tuple[int, int, float]]], None], max_ways: int):
        super(MDSDialog, self).__init__(parent)
        self.setupUi(self)
        self.waysTable.setRowCount(max_ways)
        self.waysTable.setHorizontalHeaderLabels(['Order', 'Freq (Hz)'])
        self.waysTable.setVerticalHeaderLabels([str(i + 1) for i in range(0, max_ways)])
        self.__current_ways = current
        for i in range(0, max_ways):
            self.waysTable.setCellWidget(i, 0, self.__make_combo(i, 8))
        self.waysTable.setItemDelegateForColumn(1, FreqRangeEditor())
        for way, order, freq in self.__current_ways:
            if freq:
                self.waysTable.cellWidget(way, 0).setCurrentText(str(order))
                item = self.waysTable.item(way, 1)
                if not item:
                    item = QTableWidgetItem()
                    self.waysTable.setItem(way, 1, item)
                item.setText(str(freq))
        self.__on_update = on_update

    def __make_combo(self, row: int, orders: int) -> QComboBox:
        cb = QComboBox()
        for i in range(1, orders):
            cb.addItem(str(i + 1))
        try:
            data = self.__current_ways[row]
            cb.setCurrentText(str(data[1]) if data[2] else '4')
        except IndexError:
            cb.setCurrentText('4')
        return cb

    def update_mds(self):
        vals = []
        for row in range(0, self.waysTable.rowCount()):
            freq_item = self.waysTable.item(row, 1)
            freq = float(freq_item.text()) if freq_item else None
            vals.append((row, int(self.waysTable.cellWidget(row, 0).currentText()), freq))
        freqs = [v[2] for v in vals if v[2]]
        fail = False
        if len(freqs) > 1:
            last = freqs[0]
            for f in freqs[1:]:
                if f <= last or math.isclose(f, last):
                    fail = True
                    break
        if fail:
            msg_box = QMessageBox()
            msg_box.setText(f"Crossing points must be in ascending order [supplied: {freqs}]")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Invalid MDS Configuration!')
            msg_box.exec()
        else:
            self.__on_update(vals)


class MCWSDialog(QDialog, Ui_loadDspFromZoneDialog):
    MCWS_ROLE = Qt.UserRole + 1

    def __init__(self, parent: QDialog, prefs: Preferences, download: bool = True,
                 txt_provider: Callable[[bool], str] = None, on_select: Callable[[str, str, bool], None] = None):
        super(MCWSDialog, self).__init__(parent)
        self.setupUi(self)
        self.__last_test = None
        self.prefs = prefs
        self.__download = download
        self.__txt_provider = txt_provider
        self.addNewButton.setIcon(qta.icon('fa5s.plus'))
        self.addNewButton.setToolTip('Add New MC Connection')
        self.deleteSaved.setIcon(qta.icon('fa5s.trash-alt'))
        self.deleteSaved.setToolTip('Delete selected connection')
        self.testConnectionButton.setIcon(qta.icon('fa5s.sync'))
        self.testConnectionButton.setToolTip('Check connection to MC')
        if download:
            self.upload.setIcon(qta.icon('fa5s.download'))
            self.upload.setToolTip('Download DSP Configuration to selected zone')
        else:
            self.upload.setIcon(qta.icon('fa5s.upload'))
            self.upload.setToolTip('Upload DSP Configuration to selected zone')
        self.__on_select = on_select
        import re
        self.__ip_pattern = re.compile(r'[0-9]+(?:\.[0-9]+){3}:[0-9]+')
        mcws_connections = self.prefs.get(JRIVER_MCWS_CONNECTIONS)
        for ip, auth in mcws_connections.items():
            self.__add_media_server(ip, auth)
        self.__media_server: Optional[MediaServer] = None
        self.mcIP.textChanged.connect(self.__enable_buttons)
        self.username.textChanged.connect(self.__enable_buttons)
        self.password.textChanged.connect(self.__enable_buttons)
        self.testConnectionButton.clicked.connect(self.__attempt_connect)
        self.addNewButton.clicked.connect(self.__add_pending)
        self.auth.clicked.connect(self.__check_auth)
        self.deleteSaved.clicked.connect(self.__delete_selected)
        self.savedConnections.selectionModel().selectionChanged.connect(self.__enable_buttons)
        self.savedConnections.selectionModel().selectionChanged.connect(self.__load_zones)
        self.upload.clicked.connect(self.__handle_config)
        self.__enable_buttons()
        self.__load_zones()

    def __check_auth(self, required: bool):
        self.username.setEnabled(required)
        self.password.setEnabled(required)
        self.__enable_buttons()

    def __load_zones(self):
        selection = self.savedConnections.selectionModel()
        self.zones.clear()
        if selection.hasSelection():
            self.__media_server = self.savedConnections.selectedItems()[0].data(self.MCWS_ROLE)
            try:
                zones = self.__media_server.get_zones()
                for zone_name, zone_id in zones.items():
                    item = QListWidgetItem(zone_name)
                    item.setData(self.MCWS_ROLE, zone_id)
                    self.zones.addItem(item)
                if zones:
                    self.upload.setEnabled(True)
                else:
                    self.upload.setEnabled(False)
            except MCWSError as e:
                self.resultText.setPlainText(f"{e.url} - {e.status_code}\n\n{e.msg}\n\n{e.resp}")
                self.zones.clear()

    def __enable_buttons(self):
        if self.__last_test is not None:
            self.__last_test = None
            self.testConnectionButton.setIcon(qta.icon('fa5s.sync'))
        self.__media_server = None
        can_test = len(self.username.text()) > 0 and len(self.password.text()) > 0 if self.auth.isChecked() else True
        can_test = can_test and self.__ip_pattern.match(self.mcIP.text()) is not None
        self.testConnectionButton.setEnabled(can_test)
        self.addNewButton.setEnabled(False)
        self.deleteSaved.setEnabled(len(self.savedConnections.selectedItems()) > 0)

    def __attempt_connect(self):
        auth = (self.username.text(), self.password.text()) if self.auth.isChecked() else None
        self.__media_server = MediaServer(self.mcIP.text(), auth=auth, secure=self.https.isChecked())
        try:
            self.__media_server.authenticate()
            self.addNewButton.setEnabled(True)
            self.testConnectionButton.setIcon(qta.icon('fa5s.check', color='green'))
            self.__last_test = True
            self.resultText.clear()
        except MCWSError as e:
            self.__media_server = None
            self.resultText.setPlainText(f"{e.url} - {e.status_code}\n\n{e.msg}\n\n{e.resp}")
            self.testConnectionButton.setIcon(qta.icon('fa5s.times', color='red'))
            self.__last_test = False

    def __add_pending(self):
        if self.__media_server:
            item = QListWidgetItem(f"{self.__media_server}")
            item.setData(self.MCWS_ROLE, self.__media_server)
            self.savedConnections.addItem(item)
            self.prefs.set(JRIVER_MCWS_CONNECTIONS, {
                **self.prefs.get(JRIVER_MCWS_CONNECTIONS),
                **self.__media_server.as_dict()
            })
            self.mcIP.clear()
            self.username.clear()
            self.password.clear()
            self.__media_server = None

    def __add_media_server(self, ip, auth):
        if len(auth) == 3:
            auth = ((auth[0], auth[1]), auth[2])
        item = QListWidgetItem(f"{ip} [{auth[0][0]}]" if auth[0] else f"{ip} [Unauthenticated]")
        item.setData(self.MCWS_ROLE, MediaServer(ip, *auth))
        self.savedConnections.addItem(item)

    def __delete_selected(self):
        to_delete = self.savedConnections.selectedItems()
        if to_delete:
            for d in to_delete:
                self.savedConnections.takeItem(self.savedConnections.indexFromItem(d).row())
            to_save = [self.savedConnections.item(i).data(self.MCWS_ROLE).as_dict()
                       for i in range(self.savedConnections.count())]
            output = {}
            for t in to_save:
                output = {**output, **t}
            self.prefs.set(JRIVER_MCWS_CONNECTIONS, output)

    def __handle_config(self):
        if self.__media_server:
            if self.__download:
                if self.savedConnections.selectionModel().hasSelection() and self.zones.selectionModel().hasSelection():
                    media_server: MediaServer = self.savedConnections.selectedItems()[0].data(self.MCWS_ROLE)
                    zone = self.zones.selectedItems()[0]
                    try:
                        dsp = media_server.get_dsp(zone.data(self.MCWS_ROLE))
                        self.resultText.setPlainText(f"Downloaded DSP from {zone.text()}\n\n{dsp}")
                        self.__on_select(zone.text(), dsp, media_server.convert_q)
                        self.upload.setIcon(qta.icon('fa5s.download', color='green'))
                    except MCWSError as e:
                        self.resultText.setPlainText(f"{e.url} - {e.status_code}\n\n{e.msg}\n\n{e.resp}")
                        self.upload.setIcon(qta.icon('fa5s.download', color='red'))
            else:
                if self.zones.selectionModel().hasSelection():
                    zone = self.zones.selectedItems()[0]
                    try:
                        result = self.__media_server.set_dsp(zone.data(self.MCWS_ROLE), self.__txt_provider)
                        if result:
                            self.resultText.setPlainText(f"Uploaded dsp to {zone.text()}")
                            self.upload.setIcon(qta.icon('fa5s.upload', color='green'))
                        else:
                            self.resultText.setPlainText(f"Failed to upload dsp to {zone.text()}")
                            self.upload.setIcon(qta.icon('fa5s.upload', color='red'))
                    except DSPMismatchError as e:
                        self.resultText.setPlainText(f"{e}\n\n{e.expected}\n\n{e.actual}")
                        self.upload.setIcon(qta.icon('fa5s.upload', color='red'))
                    except MCWSError as e:
                        self.resultText.setPlainText(f"{e.url} - {e.status_code}\n\n{e.msg}\n\n{e.resp}")
                        self.upload.setIcon(qta.icon('fa5s.upload', color='red'))
