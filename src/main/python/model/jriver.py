from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as et
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Callable, Union, Set, Iterable

import itertools
import math
import qtawesome as qta
import time
from builtins import isinstance
from qtpy.QtCore import QPoint, QModelIndex, Qt
from qtpy.QtGui import QColor, QPalette, QKeySequence, QCloseEvent
from qtpy.QtWidgets import QDialog, QFileDialog, QMenu, QAction, QListWidgetItem, QAbstractItemView, \
    QDialogButtonBox
from scipy.signal import unit_impulse

from model import iir
from model.filter import FilterModel, FilterDialog
from model.iir import s_to_q, SOS, CompleteFilter, SecondOrder_HighPass, PeakingEQ, LowShelf as LS, Gain as G, \
    LinkwitzTransform as LT, CompoundPassFilter, ComplexHighPass, BiquadWithQGain, q_to_s, SecondOrder_LowPass, \
    ComplexLowPass, FilterType, FirstOrder_LowPass, ComplexFilter, PassFilter, FirstOrder_HighPass
from model.limits import dBRangeCalculator, PhaseRangeCalculator
from model.log import to_millis
from model.magnitude import MagnitudeModel
from model.preferences import JRIVER_GEOMETRY, JRIVER_GRAPH_X_MIN, JRIVER_GRAPH_X_MAX, JRIVER_DSP_DIR, Preferences, \
    get_filter_colour
from model.signal import Signal
from model.xy import MagnitudeData
from ui.jriver import Ui_jriverDspDialog
from ui.jriver_channel_select import Ui_jriverChannelSelectDialog
from ui.jriver_delay_filter import Ui_jriverDelayDialog
from ui.jriver_mix_filter import Ui_jriverMixDialog
from ui.pipeline import Ui_jriverGraphDialog

USER_CHANNELS = ['User 1', 'User 2']
SHORT_USER_CHANNELS = ['U1', 'U2']
JRIVER_NAMED_CHANNELS = [None, None, 'Left', 'Right', 'Centre', 'Subwoofer', 'Surround Left', 'Surround Right',
                         'Rear Left', 'Rear Right', None] + USER_CHANNELS
JRIVER_SHORT_NAMED_CHANNELS = [None, None, 'L', 'R', 'C', 'SW', 'SL', 'SR', 'RL', 'RR', None] + SHORT_USER_CHANNELS
JRIVER_CHANNELS = JRIVER_NAMED_CHANNELS + [f"Channel {i + 9}" for i in range(24)]
JRIVER_SHORT_CHANNELS = JRIVER_SHORT_NAMED_CHANNELS + [f"#{i + 9}" for i in range(24)]

FILTER_ID_ROLE = Qt.UserRole + 1

logger = logging.getLogger('jriver')


def short_to_long(short: str) -> str:
    return JRIVER_CHANNELS[JRIVER_SHORT_CHANNELS.index(short)]


def get_channel_name(idx: int, short: bool = True) -> str:
    '''
    Converts a channel index to a named channel.
    :param idx: the index.
    :param short: get the short name if true.
    :return: the name.
    '''
    channels = JRIVER_SHORT_CHANNELS if short else JRIVER_CHANNELS
    return channels[idx]


def get_channel_idx(name: str, short: bool = True) -> int:
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
        self.__selected_node_names: Set[str] = set()
        self.__current_dot_txt = None
        self.prefs = prefs
        self.setupUi(self)
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
        self.findFilenameButton.setShortcut(QKeySequence.Open)
        self.saveButton.setShortcut(QKeySequence.Save)
        self.saveAsButton.setShortcut(QKeySequence.SaveAs)
        self.addFilterButton.setMenu(self.__populate_add_filter_menu(QMenu(self)))
        self.pipelineView.signal.on_click.connect(self.__on_node_click)
        self.pipelineView.signal.on_double_click.connect(self.__show_edit_filter_dialog)
        self.pipelineView.signal.on_context.connect(self.__show_edit_menu)
        self.showDotButton.clicked.connect(self.__show_dot_dialog)
        self.direction.toggled.connect(self.__regen)
        self.viewSplitter.setSizes([100000, 100000])
        self.filterList.model().rowsMoved.connect(self.__reorder_filters)
        self.__dsp: Optional[JRiverDSP] = None
        self.__magnitude_model = MagnitudeModel('preview', self.previewChart, prefs,
                                                self.__get_data(), 'Filter', fill_primary=False,
                                                x_min_pref_key=JRIVER_GRAPH_X_MIN, x_max_pref_key=JRIVER_GRAPH_X_MAX,
                                                secondary_data_provider=self.__get_data('phase'),
                                                secondary_name='Phase', secondary_prefix='deg', fill_secondary=False,
                                                db_range_calc=dBRangeCalculator(60),
                                                y2_range_calc=PhaseRangeCalculator(), show_y2_in_legend=False)
        self.__restore_geometry()

    def find_dsp_file(self):
        '''
        Allows user to select a DSP file and loads it as a set of graphs.
        '''
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
                main_colour = QColor(QPalette().color(QPalette.Active, QPalette.Text)).name()
                highlight_colour = QColor(QPalette().color(QPalette.Active, QPalette.Highlight)).name()
                self.__dsp = JRiverDSP(selected[0], colours=(main_colour, highlight_colour))
                self.__refresh_channel_list()
                self.filename.setText(os.path.basename(selected[0])[:-4])
                self.filterList.clear()
                self.show_filters()
                self.saveButton.setEnabled(True)
                self.saveAsButton.setEnabled(True)
                self.addFilterButton.setEnabled(True)
                self.showDotButton.setEnabled(True)
                self.direction.setEnabled(True)
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
            self.__dsp.activate(self.blockSelector.currentIndex())
            from model.report import block_signals
            with block_signals(self.filterList):
                selected_ids = [i.data(FILTER_ID_ROLE) for i in self.filterList.selectedItems()]
                i = 0
                for f in self.__dsp.active_graph.filters:
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
                        if f.id in selected_ids:
                            item.setSelected(True)
                for j in range(i, self.filterList.count()):
                    self.filterList.takeItem(j)
            self.__regen()
        self.redraw()

    def edit_selected_filter(self):
        if self.filterList.selectedItems():
            self.edit_filter(self.filterList.selectedItems()[0])

    def edit_filter(self, item: QListWidgetItem) -> None:
        '''
        Shows a jriver style filter dialog for the selected filter.
        :param item: the item describing the filter.
        '''
        f_id = item.data(FILTER_ID_ROLE)
        selected_filter = next((f for f in self.__dsp.active_graph.filters if f.id == f_id), None)
        logger.debug(f"Showing edit dialog for {selected_filter}")
        if isinstance(selected_filter, CustomFilter):
            pass
        else:
            vals = selected_filter.get_all_vals()
            if isinstance(selected_filter, SingleFilter):
                if not self.__show_basic_edit_filter_dialog(selected_filter, vals):
                    logger.debug(f"Filter {selected_filter} at node {item.text()} is not editable")
            else:
                logger.warning(f"Unexpected filter type {selected_filter} at {item.text()}, unable to edit")

    def delete_filter(self) -> None:
        '''
        Deletes the selected filter(s).
        '''
        selected_filter_ids = [i.data(FILTER_ID_ROLE) for i in self.filterList.selectedItems()]
        to_delete = [f for f in self.__dsp.active_graph.filters if f.id in selected_filter_ids]
        logger.debug(f"Deleting filter ids {selected_filter_ids} -> {to_delete}")
        self.__dsp.active_graph.delete(to_delete)
        self.__on_graph_change()
        last_row = 0
        i: QModelIndex
        from model.report import block_signals
        with block_signals(self.filterList):
            for i in self.filterList.selectedIndexes():
                last_row = i.row()
                removed = self.filterList.takeItem(last_row)
                logger.debug(f"Removing filter at row {last_row} - {removed.text()}")
            if self.filterList.count() > 0:
                self.filterList.item(max(min(last_row, self.filterList.count()) - 1, 0)).setSelected(True)
        self.__enable_edit_buttons_if_filters_selected()

    def split_filter(self):
        ''' Splits a selected multichannel filter into separate filters. '''
        selected_items: List[QListWidgetItem] = self.filterList.selectedItems()
        splittable = self.__dsp.active_graph.get_filter_by_id(selected_items[0].data(FILTER_ID_ROLE))
        item_idx: QModelIndex = self.filterList.indexFromItem(selected_items[0])
        if splittable.can_split():
            split = splittable.split()
            for i, f in enumerate(split):
                insert_at = item_idx.row() + i
                self.__dsp.active_graph.insert(f, insert_at, regen=(i + 1 == len(split)))
                new_item = QListWidgetItem(str(f))
                new_item.setData(FILTER_ID_ROLE, f.id)
                self.filterList.insertItem(insert_at, new_item)
            self.__dsp.active_graph.delete([splittable])
            self.filterList.takeItem(self.filterList.indexFromItem(selected_items[0]).row())
            self.__regen()

    def merge_filters(self):
        ''' Merges multiple identical filters into a single multichannel filter. '''
        selected_filters = [self.__dsp.active_graph.get_filter_by_id(i.data(FILTER_ID_ROLE))
                            for i in self.filterList.selectedItems()]
        item_idx: QModelIndex = self.filterList.indexFromItem(self.filterList.selectedItems()[0])
        # can only merge a ChannelFilter
        channels = ';'.join([str(f.channels[0]) for f in selected_filters])
        merged_filter = create_peq({**selected_filters[0].get_all_vals()[0], 'Channels': channels})
        insert_at = item_idx.row() + 1
        # insert the new one in both the graph and the filter list
        self.__dsp.active_graph.insert(merged_filter, insert_at)
        new_item = QListWidgetItem(str(merged_filter))
        new_item.setData(FILTER_ID_ROLE, merged_filter.id)
        self.filterList.insertItem(insert_at, new_item)
        # delete the old ones
        self.__dsp.active_graph.delete(selected_filters)
        for i in self.filterList.selectedItems():
            self.filterList.takeItem(self.filterList.indexFromItem(i).row())

    def on_filter_select(self) -> None:
        '''
        Ensures the selected nodes match the selected filters & regenerates the svg to display this.
        '''
        self.__selected_node_names.clear()
        if self.__enable_edit_buttons_if_filters_selected():
            selected_indexes = [i.row() for i in self.filterList.selectedIndexes()]
            for i in selected_indexes:
                for n in self.__dsp.active_graph.filters[i].nodes:
                    self.__selected_node_names.add(n.name)
        self.__regen()

    def clear_filters(self):
        '''
        Deletes all filters in the active graph.
        '''
        self.__dsp.active_graph.clear_filters()
        self.show_filters()

    def save_dsp(self):
        '''
        Writes the graphs to the loaded file.
        '''
        if self.__dsp:
            self.__dsp.write_to_file()

    def save_as_dsp(self):
        '''
        Writes the graphs to a user specified file.
        '''
        if self.__dsp:
            file_name = QFileDialog(self).getSaveFileName(self, caption='Save DSP Config',
                                                          directory=os.path.dirname(self.__dsp.filename),
                                                          filter="JRiver DSP (*.dsp)")
            file_name = str(file_name[0]).strip()
            if len(file_name) > 0:
                self.__dsp.write_to_file(file=file_name)

    def __refresh_channel_list(self, retain_selected=False):
        ''' Refreshes the output channels with the current channel list. '''
        from model.report import block_signals
        with block_signals(self.channelList):
            selected = [i.text() for i in self.channelList.selectedItems()]
            self.channelList.clear()
            for i, n in enumerate(self.__dsp.channel_names(output=True)):
                self.channelList.addItem(n)
                item: QListWidgetItem = self.channelList.item(i)
                item.setSelected(n in selected if retain_selected else n not in SHORT_USER_CHANNELS)
        with block_signals(self.blockSelector):
            self.blockSelector.clear()
            for i in range(self.__dsp.graph_count):
                self.blockSelector.addItem(get_peq_key_name(self.__dsp.graph(i).stage))

    def __show_edit_menu(self, node_name: str, pos: QPoint) -> None:
        '''
        Displays a context menu to allow edits to the graph to be driven from a selected node.
        :param node_name: the selected node.
        :param pos: the location to place the menu.
        '''
        menu = QMenu(self)
        self.__populate_edit_node_add_menu(menu.addMenu('&Add'), node_name)
        edit = QAction(f"&Edit", self)
        edit.triggered.connect(lambda: self.__show_edit_filter_dialog(node_name))
        menu.addAction(edit)
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
        add, copy, delay, move, peq, polarity, subtract, swap = self.__add_actions_to_filter_menu(add_menu)
        idx = self.__get_idx_to_insert_filter_at_from_node_name(node_name)
        peq.triggered.connect(lambda: self.__insert_peq_after_node(node_name))
        polarity.triggered.connect(lambda: self.__insert_polarity(idx))
        delay.triggered.connect(lambda: self.__insert_delay(idx))
        add.triggered.connect(lambda: self.__insert_mix(MixType.ADD, idx))
        copy.triggered.connect(lambda: self.__insert_mix(MixType.COPY, idx))
        move.triggered.connect(lambda: self.__insert_mix(MixType.MOVE, idx))
        swap.triggered.connect(lambda: self.__insert_mix(MixType.SWAP, idx))
        subtract.triggered.connect(lambda: self.__insert_mix(MixType.SUBTRACT, idx))

    def __get_idx_to_insert_filter_at_from_node_name(self, node_name: str) -> int:
        '''
        Locates the position in the filter list at which a filter should be added in order to be placed after the
        specified node in the pipeline.
        :param node_name: the node name.
        :return: an index to insert at.
        '''
        node = self.__dsp.active_graph.get_node(node_name)
        if node.filt:
            match = self.__find_item_by_filter_id(node.filt.id)
            if match:
                return match[0] + 1
        return 0

    def __get_idx_to_insert_filter_at_from_selection(self) -> int:
        '''
        Locates the position in the filter list at which a filter should be added based on the selected filters.
        :return: an index to insert at.
        '''
        selected = [i.row() for i in self.filterList.selectedIndexes()]
        return max(selected) + 1 if selected else 0

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
        node = self.__dsp.active_graph.get_node(node_name)
        if node.filt:
            match = self.__find_item_by_filter_id(node.filt.id)
            if match:
                if node.has_editable_filter():
                    node_idx, node_chain = node.editable_node_chain
                    filters: List[SOS] = [f.editable_filter for f in node_chain]
                    insert_at, _ = self.__find_item_by_filter_id(node_chain[0].filt.id)
                    # TODO insert a filter into the chain immediately after this one
                    self.__start_peq_edit_session(filters, node.channel, node_chain, insert_at + 1)
                else:
                    self.__start_peq_edit_session(None, node.channel, [], match[0] + 1)

    def __delete_node(self, node_name: str) -> None:
        '''
        Deletes the node from the filter list.
        :param node_name: the node to remove.
        '''
        node = self.__dsp.active_graph.get_node(node_name)
        if node and node.filt:
            f = node.filt
            if isinstance(f, ChannelFilter):
                item: QListWidgetItem
                if self.__dsp.active_graph.delete_channel(f, node.channel):
                    match = self.__find_item_by_filter_id(f.id)
                    if match:
                        self.filterList.removeItemWidget(match[1])
                else:
                    match = self.__find_item_by_filter_id(f.id)
                    if match:
                        match[1].setText(str(f))
            else:
                self.__dsp.active_graph.delete([f])
            self.__regen()

    def __populate_add_filter_menu(self, menu: QMenu) -> QMenu:
        '''
        Adds filter editing actions to the add button next to the filter list.
        :param menu: the menu to add to.
        :return: the menu.
        '''
        add, copy, delay, move, peq, polarity, subtract, swap = self.__add_actions_to_filter_menu(menu)
        peq.triggered.connect(lambda: self.__insert_peq(self.__get_idx_to_insert_filter_at_from_selection()))
        delay.triggered.connect(lambda: self.__insert_delay(self.__get_idx_to_insert_filter_at_from_selection()))
        polarity.triggered.connect(lambda: self.__insert_polarity(self.__get_idx_to_insert_filter_at_from_selection()))
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
        return menu

    def __add_actions_to_filter_menu(self, menu) -> Tuple[QAction, ...]:
        '''
        Adds all filter edit actions to the menu. No triggers are linked at this point.
        :param menu: the menu to add to.
        :return: the actions.
        '''
        peq = self.__create_peq_action(menu)
        delay = self.__create_delay_action(menu)
        polarity = self.__create_polarity_action(menu)
        add = self.__create_add_action(menu)
        copy = self.__create_copy_action(menu)
        move = self.__create_move_action(menu)
        swap = self.__create_swap_action(menu)
        subtract = self.__create_subtract_action(menu)
        return add, copy, delay, move, peq, polarity, subtract, swap

    def __create_subtract_action(self, menu) -> QAction:
        subtract = QAction(f"&8: Subtract", self)
        menu.addAction(subtract)
        return subtract

    def __create_swap_action(self, menu) -> QAction:
        swap = QAction(f"&7: Swap", self)
        menu.addAction(swap)
        return swap

    def __create_move_action(self, menu) -> QAction:
        move = QAction(f"&6: Move", self)
        menu.addAction(move)
        return move

    def __create_copy_action(self, menu) -> QAction:
        copy = QAction(f"&5: Copy", self)
        menu.addAction(copy)
        return copy

    def __create_add_action(self, menu) -> QAction:
        add = QAction(f"&4: Add", self)
        menu.addAction(add)
        return add

    def __create_polarity_action(self, menu) -> QAction:
        polarity = QAction(f"&3: Polarity", self)
        menu.addAction(polarity)
        return polarity

    def __create_delay_action(self, menu) -> QAction:
        delay = QAction(f"&2: Delay", self)
        menu.addAction(delay)
        return delay

    def __create_peq_action(self, menu) -> QAction:
        peq = QAction(f"&1: PEQ", self)
        menu.addAction(peq)
        return peq

    def __insert_peq(self, idx: int) -> None:
        '''
        Allows user to insert a new set of PEQ filters at the specified index in the filter list.
        :param idx: the index to insert at.
        '''
        channel: Optional[str] = None

        def __on_save(vals: Dict[str, str]):
            nonlocal channel
            channel = get_channel_name(int(vals['Channels']))

        val = JRiverChannelOnlyFilterDialog(self, self.__dsp.channel_names(output=True), __on_save, {}, title='PEQ',
                                            multi=False).exec()
        if val == QDialog.Accepted and channel is not None:
            self.__start_peq_edit_session(None, channel, [], idx)

    def __insert_delay(self, idx: int):
        '''
        Allows user to insert a delay filter at the specified index in the filter list.
        :param idx: the index to insert at.
        '''
        JRiverDelayDialog(self, self.__dsp.channel_names(output=True), lambda vals: self.__add_filter(vals, idx),
                          Delay.default_values()).exec()

    def __insert_polarity(self, idx: int):
        '''
        Allows user to insert an invert polarity filter at the specified index in the filter list.
        :param idx: the index to insert at.
        '''
        JRiverChannelOnlyFilterDialog(self, self.__dsp.channel_names(output=True),
                                      lambda vals: self.__add_filter(vals, idx), Polarity.default_values()).exec()

    def __insert_mix(self, mix_type: MixType, idx: int):
        '''
        Allows user to insert a mix filter at the specified index in the filter list.
        :param mix_type: the type of mix.
        :param idx: the index to insert at.
        '''
        JRiverMixFilterDialog(self, self.__dsp.channel_names(output=True), lambda vals: self.__add_filter(vals, idx),
                              mix_type, Mix.default_values()).exec()

    def __add_filter(self, vals: Dict[str, str], idx: int) -> None:
        '''
        Creates a filter from the valures supplied and inserts it into the filter list at the given index.
        :param vals: the filter values.
        :param idx: the index to insert at.
        '''
        to_add = create_peq(vals)
        if to_add:
            logger.info(f"Storing {vals} as {to_add}")
            self.__dsp.active_graph.insert(to_add, idx)
            self.__on_graph_change()
            item = QListWidgetItem(str(to_add))
            item.setData(FILTER_ID_ROLE, to_add.id)
            self.filterList.insertItem(idx, item)
        else:
            logger.error(f"No PEQ created for {vals}")

    def __on_graph_change(self) -> None:
        '''
        regenerates the svg and redraws the chart.
        '''
        self.__regen()
        self.redraw()

    def __enable_edit_buttons_if_filters_selected(self) -> bool:
        '''
        :return: true if the delete button was enabled.
        '''
        count = len(self.filterList.selectedItems())
        self.editFilterButton.setEnabled(count == 1)
        self.deleteFilterButton.setEnabled(count > 0)
        enable_split = False
        enable_merge = False
        if count == 1:
            f_id = self.filterList.selectedItems()[0].data(FILTER_ID_ROLE)
            enable_split = self.__dsp.active_graph.get_filter_by_id(f_id).can_split()
        elif count > 1:
            selected_filters = [self.__dsp.active_graph.get_filter_by_id(i.data(FILTER_ID_ROLE))
                                for i in self.filterList.selectedItems()]
            enable_merge = all(c[0].can_merge(c[1]) for c in itertools.combinations(selected_filters, 2))
        self.splitFilterButton.setEnabled(enable_split)
        self.mergeFilterButton.setEnabled(enable_merge)
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
        self.__dsp.active_graph.reorder(start, end, row)
        self.__on_graph_change()

    def __show_edit_filter_dialog(self, node_name: str) -> None:
        '''
        Shows the edit dialog for the selected node.
        :param node_name: the node.
        '''
        node = self.__dsp.graph(self.blockSelector.currentIndex()).get_node(node_name)
        if node:
            filt = node.filt
            if filt:
                if node.has_editable_filter():
                    node_idx, node_chain = node.editable_node_chain
                    filters: List[SOS] = [f.editable_filter for f in node_chain]
                    insert_at, _ = self.__find_item_by_filter_id(node_chain[0].filt.id)
                    self.__start_peq_edit_session(filters, node.channel, node_chain, insert_at + 1,
                                                  selected_filter_idx=node_idx)
                else:
                    vals = filt.get_all_vals()
                    if isinstance(filt, SingleFilter):
                        if not self.__show_basic_edit_filter_dialog(filt, vals):
                            logger.debug(f"Filter {filt} at node {node_name} is not editable")
                    else:
                        logger.warning(f"Unexpected filter type {filt} at {node_name}, unable to edit")
            else:
                logger.debug(f"No filter at node {node_name}")
        else:
            logger.debug(f"No such node {node_name}")

    def __show_basic_edit_filter_dialog(self, to_edit: SingleFilter, vals: List[Dict[str, str]]) -> bool:
        if len(vals) == 1 and hasattr(to_edit, 'default_values'):

            def __on_save(vals_to_save: Dict[str, str]):
                to_add = create_peq(vals_to_save)
                if to_add:
                    to_add.id = to_edit.id
                    logger.info(f"Storing {vals_to_save} as {to_add}")
                    if self.__dsp.active_graph.replace(to_edit, to_add):
                        item = self.__find_item_by_filter_id(to_edit.id)
                        if item:
                            item[1].setText(str(to_add))
                        self.__on_graph_change()
                    else:
                        logger.error(f"Failed to replace {to_edit}")

            if isinstance(to_edit, Delay):
                JRiverDelayDialog(self, self.__dsp.channel_names(output=True), __on_save, vals[0]).exec()
            elif isinstance(to_edit, Polarity):
                JRiverChannelOnlyFilterDialog(self, self.__dsp.channel_names(output=True), __on_save, vals[0]).exec()
            elif isinstance(to_edit, Mix):
                JRiverMixFilterDialog(self, self.__dsp.channel_names(output=True), __on_save, to_edit.mix_type,
                                      vals[0]).exec()
            elif to_edit.get_editable_filter() is not None:
                logger.debug("Unable to edit PEQ from the filter list")
                return False
            else:
                return False
            return True
        else:
            return False

    def __start_peq_edit_session(self, filters: Optional[List[SOS]], channel: str, node_chain: List[Node],
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
        filter_model.filter = CompleteFilter(fs=48000, filters=sorted_filters, sort_by_id=True, description=channel)
        self.__dsp.active_graph.start_edit(channel, node_chain, insert_at)

        def __on_save():
            if self.__dsp.active_graph.end_edit(filter_model.filter):
                self.show_filters()

        x_lim = (self.__magnitude_model.limits.x_min, self.__magnitude_model.limits.x_max)
        FilterDialog(self.prefs, make_signal(channel), filter_model, __on_save, parent=self,
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
        self.clearFiltersButton.setEnabled(len(self.__dsp.active_graph.filters) > 0)
        self.__current_dot_txt = self.__dsp.as_dot(self.blockSelector.currentIndex(),
                                                   vertical=self.direction.isChecked(),
                                                   selected_nodes=self.__selected_node_names)
        self.__gen_svg()

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
        node_filter = self.__dsp.active_graph.get_filter_at_node(node_name)
        if node_filter:
            nodes_in_filter = [n.name for n in node_filter.nodes]
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
        graph = self.__dsp.graph(self.blockSelector.currentIndex())
        filts = [graph.get_filter_at_node(n) for n in self.__selected_node_names if graph.get_filter_at_node(n) is not None]
        if filts:
            selected_filter = graph.get_filter_at_node(selected_node)
            i = 0
            for f in self.__dsp.graph(self.blockSelector.currentIndex()).filters:
                if not isinstance(f, Divider):
                    from model.report import block_signals
                    with block_signals(self.filterList):
                        item: QListWidgetItem = self.filterList.item(i)
                        if filts and f in filts:
                            if f.id == selected_filter.id:
                                if not isinstance(f, Mix) or f.mix_type in [MixType.ADD, MixType.SUBTRACT]:
                                    for n in f.nodes:
                                        self.__selected_node_names.add(n.name)
                                item.setSelected(True)
                        else:
                            item.setSelected(False)
                    i += 1
            self.__enable_edit_buttons_if_filters_selected()
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
            if self.__dsp:
                names = [n.text() for n in self.channelList.selectedItems()]
                for signal in self.__dsp.signals:
                    if signal.name in names:
                        result.append(MagnitudeData(signal.name, None, *signal.avg,
                                                    colour=get_filter_colour(len(result)),
                                                    linestyle='-' if mode == 'mag' else '--'))
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

    def closeEvent(self, event: QCloseEvent):
        ''' Stores the window size on close. '''
        self.prefs.set(JRIVER_GEOMETRY, self.saveGeometry())
        super().closeEvent(event)


class JRiverDSP:

    def __init__(self, filename: str, colours: Tuple[str, str] = None):
        self.__active_idx = 0
        self.__filename = filename
        self.__colours = colours
        start = time.time()
        self.__config_txt = Path(self.__filename).read_text()
        peq_block_order = get_peq_block_order(self.__config_txt)
        i, o = get_available_channels(self.__config_txt)
        self.__input_channel_indexes = i
        self.__output_channel_indexes = o
        self.__graphs: List[FilterGraph] = []
        self.__signals: Dict[str, Signal] = {}
        for block in peq_block_order:
            out_names = self.channel_names(output=True)
            in_names = out_names if self.__graphs else self.channel_names(output=False)
            self.__graphs.append(FilterGraph(block, in_names, out_names, self.__parse_peq(self.__config_txt, block)))
        end = time.time()
        logger.info(f"Parsed {filename} in {to_millis(start, end)}ms")

    @property
    def filename(self):
        return self.__filename

    def __init_signals(self) -> Dict[str, Signal]:
        names = [get_channel_name(c) for c in self.__output_channel_indexes]
        return {c: make_signal(c) for c in names}

    @property
    def signals(self) -> List[Signal]:
        return list(self.__signals.values()) if self.__signals else []

    @property
    def graph_count(self) -> int:
        return len(self.__graphs)

    def graph(self, idx) -> FilterGraph:
        return self.__graphs[idx]

    def as_dot(self, idx, vertical=True, selected_nodes=None) -> str:
        renderer = GraphRenderer(self.__graphs[idx], colours=self.__colours)
        return renderer.generate(vertical, selected_nodes=selected_nodes)

    def channel_names(self, short=True, output=False):
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
        individual_filters = [create_peq(d) for d in [self.__item_to_dicts(f) for f in filt_fragments[2:]] if d]
        return self.__coalesce_peq(individual_filters)

    @staticmethod
    def __coalesce_peq(individual_filters: List[Filter]) -> List[Filter]:
        '''
        Combines individual filters into CustomFilter instances based on divider text.
        :param individual_filters: the raw filters.
        :return: the coalesced filters.
        '''
        combined_filters: List[Filter] = []
        custom_name: Optional[str] = None
        custom_filters: List[SingleFilter] = []
        for f in individual_filters:
            if isinstance(f, Divider):
                custom_name = CustomFilter.get_custom_filter_name(f.text, custom_name is None)
                if custom_name is not None:
                    if custom_filters:
                        logger.debug(f"Coalesced {len(custom_filters)} into {custom_name}")
                        combined_filters.append(CustomFilter(custom_name, custom_filters))
                        custom_name = None
                    else:
                        custom_filters = []
                else:
                    logger.debug(f"Skipping divider {f.text}")
            elif custom_name:
                if isinstance(f, SingleFilter):
                    custom_filters.append(f)
                else:
                    logger.error(f"Unexpected filter type ignored as part of custom filter {custom_name} {f}")
            else:
                combined_filters.append(f)
        for i, f in enumerate(combined_filters):
            f.id = (i + 1) * 1000
            if isinstance(f, CustomFilter):
                for i1, f1 in enumerate(f.filters):
                    f1.id = f.id + 1 + i1
        return combined_filters

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

    def activate(self, active_idx: int) -> None:
        '''
        Activates the selected graph & generates signals accordingly.
        :param active_idx: the active graph index.
        '''
        self.__active_idx = active_idx
        self.__generate_signals()

    def __generate_signals(self) -> None:
        '''
        Creates a FilterPipeline for each channel and applies it to a unit impulse.
        '''
        start = time.time()
        signals: Dict[str, Signal] = self.__init_signals()
        branches: List[BranchFilterOp] = []
        incomplete: Dict[str, FilterPipe] = {}
        for c, pipe in self.active_graph.filter_pipes_by_channel.items():
            logger.info(f"Filtering {c} using {pipe}")
            while pipe is not None:
                if isinstance(pipe.op, BranchFilterOp):
                    branches.append(pipe.op)
                if not pipe.op.ready:
                    source = next((b for b in branches if b.is_source_for(pipe.op)), None)
                    if source:
                        pipe.op.accept(source.source_signal)
                if pipe.op.ready:
                    signals[c] = pipe.op.apply(signals[c])
                    pipe = pipe.next
                else:
                    incomplete[c] = pipe
                    pipe = None
        if incomplete:
            logger.error(f"Incomplete filter pipeline detected {incomplete}")
            # TODO
        self.__signals = signals
        end = time.time()
        logger.info(f"Generated {len(signals)} signals in {to_millis(start, end)} ms")

    @property
    def active_graph(self):
        return self.__graphs[self.__active_idx]

    def write_to_file(self, file=None) -> None:
        '''
        Writes the dsp config to the default file or the file provided.
        :param file: the file, if any.
        '''
        output_file = self.filename if file is None else file
        logger.info(f"Writing to {output_file}")
        new_txt = self.__config_txt
        for graph in self.__graphs:
            xml_filts = [filts_to_xml(f.get_all_vals()) for f in graph.filters]
            new_txt = include_filters_in_dsp(get_peq_key_name(graph.stage), new_txt, xml_filts)
        with open(output_file, mode='w', newline='\r\n') as f:
            f.write(new_txt)
        logger.info(f"Written new config to {output_file}")


class Filter(ABC):

    def __init__(self, short_name):
        self.__short_name = short_name
        self.__f_id = -1
        self.__nodes: List[Node] = []

    def reset(self):
        self.__nodes = []

    @property
    def id(self):
        return self.__f_id

    @id.setter
    def id(self, f_id: int):
        self.__f_id = f_id

    @property
    def nodes(self):
        return self.__nodes

    @property
    def short_name(self):
        return self.__short_name

    def short_desc(self):
        return self.short_name

    @property
    def enabled(self):
        return True

    def encode(self):
        return filts_to_xml(self.get_all_vals())

    @abstractmethod
    def get_all_vals(self) -> List[Dict[str, str]]:
        pass

    def get_editable_filter(self) -> Optional[SOS]:
        return None

    def get_filter(self) -> FilterOp:
        return NopFilterOp()

    def is_mine(self, idx):
        return True

    def can_merge(self, o: Filter):
        return False

    def can_split(self) -> bool:
        return False

    def split(self) -> List[Filter]:
        return [self]

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Filter):
            return self.get_all_vals() == o.get_all_vals()
        return False


class SingleFilter(Filter):

    def __init__(self, vals, short_name):
        super().__init__(short_name)
        self.__vals = vals
        self.__enabled = vals['Enabled'] == '1'
        self.__type_code = vals['Type']

    @property
    def enabled(self):
        return self.__enabled

    def get_all_vals(self) -> List[Dict[str, str]]:
        vals = {
            'Enabled': '1' if self.__enabled else '0',
            'Type': self.__type_code,
            **self.get_vals()
        }
        if self.key_order:
            return [{k: vals[k] for k in self.key_order}]
        else:
            return [vals]

    @property
    def key_order(self) -> List[str]:
        return []

    def get_vals(self) -> Dict[str, str]:
        return {}

    def print_disabled(self):
        return '' if self.enabled else f" *** DISABLED ***"

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            'Enabled': '1',
            'Type': cls.TYPE
        }


class ChannelFilter(SingleFilter):

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

    def can_merge(self, o: Filter):
        if isinstance(o, ChannelFilter):
            if self.__class__ == o.__class__:
                if pop_channels(self.get_all_vals()) == pop_channels(o.get_all_vals()):
                    repeating_channels = set(self.channels).intersection(set(o.channels))
                    return len(repeating_channels) == 0
        return False

    def can_split(self) -> bool:
        return len(self.channels) > 1

    def split(self) -> List[Filter]:
        return [create_peq({**self.get_all_vals()[0], 'Channels': str(c)}) for c in self.channels]

    def is_mine(self, idx):
        return idx in self.channels

    def pop_channel(self, channel_name: str):
        '''
        Removes a channel from the filter.
        :param channel_name: the (short) channel name to remove.
        '''
        if channel_name in self.__channel_names:
            self.__channel_names.remove(channel_name)
            self.__channels.remove(get_channel_idx(channel_name))
        else:
            raise ValueError(f"{channel_name} not found in {self}")

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            **super(ChannelFilter, cls).default_values(),
            'Channels': ''
        }


class GainQFilter(ChannelFilter):

    def __init__(self, vals, create_iir, short_name):
        super().__init__(vals, short_name)
        self.__create_iir = create_iir
        self.__gain = float(vals['Gain'])
        self.__frequency = float(vals['Frequency'])
        self.__q = self.from_jriver_q(float(vals['Q']), self.__gain)

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Slope', 'Q', 'Type', 'Gain', 'Frequency', 'Channels']

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
            'Q': f"{self.to_jriver_q(self.__q, self.__gain):.4g}",
            'Gain': f"{self.__gain:.7g}",
            'Frequency': f"{self.__frequency:.7g}",
            **super().get_vals()
        }

    @classmethod
    def from_jriver_q(cls,  q: float, gain: float) -> float:
        return q

    @classmethod
    def to_jriver_q(cls,  q: float, gain: float) -> float:
        return q

    def get_filter(self) -> FilterOp:
        sos = self.get_editable_filter().get_sos()
        if sos:
            return SosFilterOp(sos)
        else:
            return NopFilterOp()

    def get_editable_filter(self) -> Optional[SOS]:
        return self.__create_iir(48000, self.__frequency, self.__q, self.__gain, f_id=self.id)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__gain:+.7g} dB Q={self.__q:.4g} at {self.__frequency:.7g} Hz {self.print_channel_names()}{self.print_disabled()}"


class Peak(GainQFilter):
    TYPE = '3'

    def __init__(self, vals):
        super().__init__(vals, iir.PeakingEQ, 'Peak')


class LowShelf(GainQFilter):
    TYPE = '10'

    def __init__(self, vals):
        super().__init__(vals, iir.LowShelf, 'LS')

    @classmethod
    def from_jriver_q(cls, q: float, gain: float):
        return s_to_q(q, gain)

    @classmethod
    def to_jriver_q(cls, q: float, gain: float):
        return q_to_s(q, gain)


class HighShelf(GainQFilter):
    TYPE = '11'

    def __init__(self, vals):
        super().__init__(vals, iir.HighShelf, 'HS')

    @classmethod
    def from_jriver_q(cls, q: float, gain: float) -> float:
        return s_to_q(q, gain)

    @classmethod
    def to_jriver_q(cls, q: float, gain: float) -> float:
        return q_to_s(q, gain)


class Pass(ChannelFilter):

    def __init__(self, vals: dict, short_name: str,
                 one_pole_ctor: Callable[..., Union[FirstOrder_LowPass, FirstOrder_HighPass]],
                 two_pole_ctor: Callable[..., PassFilter],
                 many_pole_ctor: Callable[..., CompoundPassFilter]):
        super().__init__(vals, short_name)
        self.__order = int(int(vals['Slope']) / 6)
        self.__frequency = float(vals['Frequency'])
        self.__jriver_q = float(vals['Q'])
        self.__ctors = (one_pole_ctor, two_pole_ctor, many_pole_ctor)

    @classmethod
    def from_jriver_q(cls, q: float) -> float:
        return q / 2**0.5

    @classmethod
    def to_jriver_q(cls, q: float) -> float:
        return q * 2**0.5

    @property
    def freq(self) -> float:
        return self.__frequency

    @property
    def jriver_q(self) -> float:
        return self.__jriver_q

    @property
    def q(self):
        if self.order == 2:
            return self.from_jriver_q(self.jriver_q)
        else:
            return self.jriver_q

    @property
    def order(self) -> int:
        return self.__order

    def get_vals(self) -> Dict[str, str]:
        return {
            'Gain': '0',
            'Slope': f"{self.order * 6}",
            'Q': f"{self.jriver_q:.4g}",
            'Frequency': f"{self.__frequency:.7g}",
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Slope', 'Q', 'Type', 'Gain', 'Frequency', 'Channels']

    def get_filter(self) -> FilterOp:
        sos = self.get_editable_filter().get_sos()
        if sos:
            return SosFilterOp(sos)
        else:
            return NopFilterOp()

    def short_desc(self):
        q_suffix = ''
        if not math.isclose(self.jriver_q, 1.0):
            q_suffix = f" VarQ"
        if self.freq >= 1000:
            f = f"{self.freq/1000.0:.3g}k"
        else:
            f = f"{self.freq:g}"
        return f"{self.short_name}{self.order} BW {f}{q_suffix}"

    def get_editable_filter(self) -> Optional[SOS]:
        '''
        Creates a set of biquads which translate the non standard jriver Q into a real Q value.
        :return: the filters.
        '''
        if self.order == 1:
            return self.__ctors[0](48000, self.freq, f_id=self.id)
        elif self.order == 2:
            return self.__ctors[1](48000, self.freq, q=self.q, f_id=self.id)
        else:
            return self.__ctors[2](FilterType.BUTTERWORTH, self.order, 48000, self.freq, q_scale=self.q, f_id=self.id)

    def __repr__(self):
        return f"{self.__class__.__name__} Order={self.order} Q={self.q:.4g} at {self.freq:.7g} Hz {self.print_channel_names()}{self.print_disabled()}"


class LowPass(Pass):
    TYPE = '1'

    def __init__(self, vals):
        super().__init__(vals, 'LP', FirstOrder_LowPass, SecondOrder_LowPass, ComplexLowPass)


class HighPass(Pass):
    TYPE = '2'

    def __init__(self, vals):
        super().__init__(vals, 'HP', FirstOrder_HighPass, SecondOrder_HighPass, ComplexHighPass)


class Gain(ChannelFilter):
    TYPE = '4'

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

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Gain', 'Channels']

    def __repr__(self):
        return f"Gain {self.__gain:+.7g} dB {self.print_channel_names()}{self.print_disabled()}"

    def get_filter(self) -> FilterOp:
        return GainFilterOp(self.__gain)

    def short_desc(self):
        return f"{self.__gain:+.7g} dB"

    def get_editable_filter(self) -> Optional[SOS]:
        return iir.Gain(48000, self.gain, f_id=self.id)


class BitdepthSimulator(SingleFilter):
    TYPE = '13'

    def __init__(self, vals):
        super().__init__(vals, 'BITDEPTH')
        self.__bits = int(vals['Bits'])
        self.__dither = vals['Dither']

    def get_vals(self) -> Dict[str, str]:
        return {
            'Bits': str(self.__bits),
            'Dither': self.__dither
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Bits', 'Type', 'Dither']

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.__bits} bits){self.print_disabled()}"


class Delay(ChannelFilter):
    TYPE = '7'

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

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Delay', 'Type', 'Channels']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__delay:+.7g} ms {self.print_channel_names()}{self.print_disabled()}"

    def short_desc(self):
        return f"{self.__delay:+.7g}ms"

    def get_filter(self) -> FilterOp:
        return DelayFilterOp(self.__delay)


class Divider(SingleFilter):
    TYPE = '20'

    def __init__(self, vals):
        super().__init__(vals, '---')
        self.__text = vals['Text'] if 'Text' in vals else ''

    @property
    def text(self):
        return self.__text

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
    TYPE = '9'

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

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Hold', 'Type', 'Mode', 'Channels', 'Level', 'Release', 'Attack']

    def __repr__(self):
        return f"{LimiterMode(int(self.__mode)).name.capitalize()} {self.__class__.__name__} at {self.__level} dB {self.print_channel_names()}{self.print_disabled()}"


class LinkwitzTransform(ChannelFilter):
    TYPE = '8'

    def __init__(self, vals):
        super().__init__(vals, 'LT')
        self.__fp = float(vals['Fp'])
        self.__qp = float(vals['Qp'])
        self.__fz = float(vals['Fz'])
        self.__qz = float(vals['Qz'])
        self.__prevent_clipping = vals['PreventClipping']

    def get_vals(self) -> Dict[str, str]:
        return {
            'Fp': f"{self.__fp:.7g}",
            'Qp': f"{self.__qp:.4g}",
            'Fz': f"{self.__fz:.7g}",
            'Qz': f"{self.__qz:.4g}",
            'PreventClipping': self.__prevent_clipping,
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Fp', 'Qp', 'Type', 'Fz', 'Channels', 'Qz', 'PreventClipping']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__fz:.7g} Hz / {self.__qz:.4g} -> {self.__fp:.7g} Hz / {self.__qp:.4g} {self.print_channel_names()}{self.print_disabled()}"

    def get_filter(self) -> FilterOp:
        sos = iir.LinkwitzTransform(48000, self.__fz, self.__qz, self.__fp, self.__qp).get_sos()
        if sos:
            return SosFilterOp(sos)
        else:
            return NopFilterOp()


class LinkwitzRiley(SingleFilter):
    TYPE = '16'

    def __init__(self, vals):
        super().__init__(vals, 'LR')
        self.__freq = float(vals['Frequency'])

    def get_vals(self) -> Dict[str, str]:
        return {
            'Frequency': f"{self.__freq:.7g}"
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Frequency']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__freq:.7g} Hz{self.print_disabled()}"


class MidSideDecoding(SingleFilter):
    TYPE = '19'

    def __init__(self, vals):
        super().__init__(vals, 'MS Decode')

    def get_vals(self) -> Dict[str, str]:
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}{self.print_disabled()}"


class MidSideEncoding(SingleFilter):
    TYPE = '18'

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


class Mix(SingleFilter):
    TYPE = '6'

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
            'Gain': f"{self.__gain:.7g}",
            'Destination': self.__destination,
            'Mode': f"{self.__mode}"
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Source', 'Gain', 'Destination', 'Mode']

    def __repr__(self):
        mix_type = MixType(self.__mode)
        src_name = get_channel_name(int(self.__source))
        dst_name = get_channel_name(int(self.__destination))
        return f"{mix_type.name.capitalize()} {src_name} {mix_type.modifier} {dst_name} {self.__gain:+.7g} dB{self.print_disabled()}"

    def is_mine(self, idx):
        return self.src_idx == idx

    def short_desc(self):
        mix_type = MixType(self.__mode)
        if mix_type == MixType.MOVE or mix_type == MixType.MOVE:
            return f"{mix_type.name}\nto {get_channel_name(int(self.__destination))}"
        elif mix_type == MixType.SWAP:
            return f"{mix_type.name}\n{get_channel_name(int(self.__source))}-{get_channel_name(int(self.__destination))}"
        else:
            return super().short_desc()

    def get_filter(self) -> FilterOp:
        if self.mix_type == MixType.ADD:
            return AddFilterOp()
        elif self.mix_type == MixType.SUBTRACT:
            return SubtractFilterOp()
        else:
            return NopFilterOp()

    @classmethod
    def default_values(cls) -> Dict[str, str]:
        return {
            **super(Mix, cls).default_values(),
            'Source': '1',
            'Gain': '0',
            'Destination': '2'
        }


class Order(SingleFilter):
    TYPE = '12'

    def __init__(self, vals):
        super().__init__(vals, 'ORDER')
        self.__order = vals['Order'].split(',')
        self.__named_order = [get_channel_name(int(i)) for i in self.__order]

    def get_vals(self) -> Dict[str, str]:
        return {
            'Order': ','.join(self.__order)
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Order', 'Type']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__named_order}{self.print_disabled()}"


class Mute(ChannelFilter):
    TYPE = '5'

    def __init__(self, vals):
        super().__init__(vals, 'MUTE')
        self.__gain = float(vals['Gain'])

    def get_vals(self) -> Dict[str, str]:
        return {
            'Gain': f"{self.__gain:.7g}",
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Gain', 'Channels']


class Polarity(ChannelFilter):
    TYPE = '15'

    def __init__(self, vals):
        super().__init__(vals, 'INVERT')

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Channels']


class SubwooferLimiter(ChannelFilter):
    TYPE = '14'

    def __init__(self, vals):
        super().__init__(vals, 'SW Limiter')
        self.__level = float(vals['Level'])

    def get_vals(self) -> Dict[str, str]:
        return {
            'Level': f"{self.__level:.7g}",
            **super().get_vals()
        }

    @property
    def key_order(self) -> List[str]:
        return ['Enabled', 'Type', 'Channels', 'Level']

    def __repr__(self):
        return f"{self.__class__.__name__} {self.__level:+.7g} dB {self.print_channel_names()}{self.print_disabled()}"


class CustomFilter(Filter):

    def __init__(self, name, filters: List[SingleFilter]):
        super().__init__('CF')
        self.__name = name
        self.__prefix: SingleFilter = self.__make_divider(True)
        self.__filters = filters
        self.__suffix: SingleFilter = self.__make_divider(False)

    def __make_divider(self, start: bool):
        return Divider({
            'Enabled': '1',
            'Type': Divider.TYPE,
            'Text': f"***CUSTOM_{'START' if start else 'END'}|{self.__name}|***"
        })

    def short_desc(self):
        tokens = self.__name.split('/')
        freq = float(tokens[3])
        if freq >= 1000:
            f = f"{freq/1000.0:.3g}k"
        else:
            f = f"{freq:g}"
        return f"{tokens[0]}{tokens[2]} {tokens[1]} {f}"

    @property
    def filters(self) -> List[SingleFilter]:
        return self.__filters

    def get_all_vals(self) -> List[Dict[str, str]]:
        all_filters: List[SingleFilter] = [self.__prefix] + self.__filters + [self.__suffix]
        return [v for f in all_filters for v in f.get_all_vals()]

    def __repr__(self):
        return self.__name

    def is_mine(self, idx):
        return self.filters[0].is_mine(idx)

    def get_editable_filter(self) -> Optional[SOS]:
        return self.__decode_custom_filter()

    def get_filter(self) -> FilterOp:
        return SosFilterOp(self.get_editable_filter().get_sos())

    def __decode_custom_filter(self) -> SOS:
        '''
        Decodes a custom filter name into a filter.
        :param desc: the filter description.
        :return: the filter.
        '''
        tokens = self.__name.split('/')
        if len(tokens) == 5:
            f_type = FilterType(tokens[1])
            order = int(tokens[2])
            freq = float(tokens[3])
            q_scale = float(tokens[4])
            if tokens[0] == 'HP':
                return ComplexHighPass(f_type, order, 48000, freq, q_scale)
            elif tokens[0] == 'LP':
                return ComplexLowPass(f_type, order, 48000, freq, q_scale)
        raise ValueError(f"Unable to decode {self.__name}")

    @staticmethod
    def get_custom_filter_name(text: str, start: bool) -> Optional[str]:
        prefix = f"***CUSTOM_{'START' if start else 'END'}"
        return text.split('|')[1] if text.startswith(prefix) else None


def all_subclasses(cls):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])


filter_classes_by_type = {c.TYPE: c for c in all_subclasses(Filter) if hasattr(c, 'TYPE')}


def create_peq(vals: Dict[str, str]) -> Filter:
    '''
    :param vals: the vals from the encoded xml format.
    :param channels: the available channel names.
    :return: a filter type.
    '''
    return filter_classes_by_type[vals['Type']](vals)


def convert_filter_to_mc_dsp(filt: SOS, target_channels: str) -> Filter:
    '''
    :param filt: a filter.
    :param target_channels: the channels to output to.
    :return: a filter
    '''
    if isinstance(filt, BiquadWithQGain):
        if isinstance(filt, PeakingEQ):
            f_type = Peak
            q = filt.q
        else:
            q = q_to_s(filt.q, filt.gain)
            f_type = LowShelf if isinstance(filt, LS) else HighShelf
        return f_type({
            'Enabled': '1',
            'Slope': '12',
            'Q': f"{q:.4g}",
            'Type': f_type.TYPE,
            'Gain': f"{filt.gain:.7g}",
            'Frequency': f"{filt.freq:.7g}",
            'Channels': target_channels
        })
    elif isinstance(filt, G):
        return Gain({
            'Enabled': '1',
            'Type': Gain.TYPE,
            'Gain': f"{filt.gain:.7g}",
            'Channels': target_channels
        })
    elif isinstance(filt, LT):
        return LinkwitzTransform({
            'Enabled': '1',
            'Type': LinkwitzTransform.TYPE,
            'Fz': filt.f0,
            'Qz': filt.q0,
            'Fp': filt.fp,
            'Qp': filt.qp,
            'PreventClipping': 0,
            'Channels': target_channels
        })
    elif isinstance(filt, CompoundPassFilter):
        if filt.type == FilterType.BUTTERWORTH and filt.order in [4, 6, 8]:
            pass_type = HighPass if isinstance(filt, ComplexHighPass) else LowPass
            vals = __make_high_order_mc_pass_filter(filt, pass_type.TYPE, pass_type.to_jriver_q, target_channels)
            return pass_type(vals)
        else:
            return __make_mc_custom_pass_filter(filt, target_channels)
    elif isinstance(filt, PassFilter):
        pass_type = HighPass if isinstance(filt, SecondOrder_HighPass) else LowPass
        return pass_type(__make_mc_pass_filter(filt, pass_type.TYPE, pass_type.to_jriver_q, target_channels))
    elif isinstance(filt, FirstOrder_LowPass) or isinstance(filt, FirstOrder_HighPass):
        pass_type = HighPass if isinstance(filt, FirstOrder_HighPass) else LowPass
        return pass_type(__make_mc_pass_filter(filt, pass_type.TYPE, pass_type.to_jriver_q, target_channels))
    else:
        raise ValueError(f"Unsupported filter type {filt}")


def __make_mc_custom_pass_filter(p_filter: CompoundPassFilter, target_channels: str) -> CustomFilter:
    pass_type = HighPass if isinstance(p_filter, ComplexHighPass) else LowPass
    mc_filts = [pass_type(__make_mc_pass_filter(f, pass_type.TYPE, pass_type.to_jriver_q, target_channels))
                for f in p_filter.filters]
    type_code = 'HP' if pass_type == HighPass else 'LP'
    encoded = f"{type_code}/{p_filter.type.value}/{p_filter.order}/{p_filter.freq:.7g}/{p_filter.q_scale:.4g}"
    return CustomFilter(encoded, mc_filts)


def __make_high_order_mc_pass_filter(f: CompoundPassFilter, filt_type: str, convert_q: Callable[[float], float],
                                     target_channels: str) -> Dict[str, str]:
    return {
        'Enabled': '1',
        'Slope': f"{f.order * 6}",
        'Type': filt_type,
        'Q': f"{convert_q(f.q_scale):.4g}",
        'Frequency': f"{f.freq:.7g}",
        'Gain': '0',
        'Channels': target_channels
    }


def __make_mc_pass_filter(f: Union[FirstOrder_LowPass, FirstOrder_HighPass, PassFilter],
                          filt_type: str, convert_q: Callable[[float], float], target_channels: str) -> Dict[str, str]:
    return {
        'Enabled': '1',
        'Slope': f"{f.order * 6}",
        'Type': filt_type,
        'Q': f"{convert_q(f.q):.4g}" if hasattr(f, 'q') else '1',
        'Frequency': f"{f.freq:.7g}",
        'Gain': '0',
        'Channels': target_channels
    }


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


def extract_filters(config_txt: str, key_name: str):
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


def filts_to_xml(vals: List[Dict[str, str]]) -> str:
    '''
    Formats key-value pairs into a jriver dsp config file compatible str fragment.
    :param vals: the key-value pairs.
    :return: the txt snippet.
    '''
    return ''.join(filt_to_xml(f) for f in vals)


def filt_to_xml(vals: Dict[str, str]) -> str:
    '''
    Converts a set of filter values to a jriver compatible xml fragment.
    :param vals: the values.
    :return: the xml fragment.
    '''
    items = [f"<Item Name=\"{k}\">{v}</Item>" for k, v in vals.items()]
    catted_items = '\n'.join(items)
    prefix = '<XMLPH version="1.1">'
    suffix = '</XMLPH>'
    txt_length = len(prefix) + len(''.join(items)) + len(suffix)
    new_line_len = (len(items) + 1) * 2
    total_len = txt_length + new_line_len
    xml_frag = f"({total_len}:{prefix}\n{catted_items}\n{suffix})"
    # print(f"{filter_classes_by_type[vals['Type']].__name__} ({vals['Type']}): {offset}")
    return xml_frag


def include_filters_in_dsp(peq_block_name: str, config_txt: str, xml_filts: List[str], replace: bool = True) -> str:
    '''
    :param peq_block_name: the peq block to process.
    :param config_txt: the dsp config in txt form.
    :param xml_filts: the filters to include.
    :param replace: if true, replace existing filters. if false, append.
    :return: the new config txt.
    '''
    root, filt_element = extract_filters(config_txt, peq_block_name)
    # before_value, after_value, filt_section = extract_value_section(config_txt, self.__block)
    # separate the tokens, which are in (TOKEN) blocks, from within the Value element
    filt_fragments = [v + ')' for v in filt_element.text.split(')') if v]
    if len(filt_fragments) < 2:
        raise ValueError('Invalid input file - Unexpected <Value> format')
    # find the filter count and replace it with the new filter count
    new_filt_count = sum([x.count('<XMLPH version') for x in xml_filts])
    if not replace:
        new_filt_count = int(filt_fragments[1][1:-1].split(':')[1]) + new_filt_count
    filt_fragments[1] = f"({len(str(new_filt_count))}:{new_filt_count})"
    # append the new filters to any existing ones or replace
    if replace:
        new_filt_section = ''.join(filt_fragments[0:2]) + ''.join(xml_filts)
    else:
        new_filt_section = ''.join(filt_fragments) + ''.join(xml_filts)
    # replace the value block in the original string
    filt_element.text = new_filt_section
    config_txt = et.tostring(root, encoding='UTF-8', xml_declaration=True).decode('utf-8')
    return config_txt


class Node:
    '''
    A single node in a filter chain, linked in both directions to parent(s) and children, if any.
    '''

    def __init__(self, rank: int, name: str, filt: Optional[Filter], channel: str):
        self.name = name
        self.rank = rank
        self.__filt = filt
        self.__filter_op = filt.get_filter() if filt else NopFilterOp()
        self.__filter_op.node_id = name
        self.channel = channel
        self.visited = False
        if self.filt is None and self.channel is None:
            raise ValueError('Must have either filter or channel')
        self.__down_edges: List[Node] = []
        self.__up_edges: List[Node] = []

    def add(self, node: Node, link_branch: bool = False):
        if node not in self.downstream:
            self.downstream.append(node)
            node.upstream.append(self)
            if link_branch is True:
                self.__filter_op = BranchFilterOp(node.filter_op, self.name)

    def has_editable_filter(self):
        return self.__filt and self.__filt.get_editable_filter() is not None

    @property
    def editable_filter(self) -> Optional[SOS]:
        '''
        :return: a beqd filter, if one can be provided by this filter.
        '''
        if self.__filt:
            return self.__filt.get_editable_filter()
        return None

    @property
    def editable_node_chain(self) -> Tuple[int, List[Node]]:
        '''
        A contiguous chain of filters which can be edited in the filter dialog as one unit.
        :return: idx of this node in the chain, the list of nodes.
        '''
        chain = []
        pos = 0
        if self.has_editable_filter():
            t = self
            while len(t.upstream) == 1 and t.upstream[0].has_editable_filter():
                chain.insert(0, t.upstream[0])
                t = t.upstream[0]
            pos = len(chain)
            chain.append(self)
            t = self
            while len(t.downstream) == 1 and t.downstream[0].has_editable_filter():
                chain.append(t.downstream[0])
                t = t.downstream[0]
        return pos, chain

    @property
    def filt(self) -> Optional[Filter]:
        return self.__filt

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

    @property
    def filter_op(self) -> FilterOp:
        return self.__filter_op

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


class FilterGraph:

    def __init__(self, stage: int, input_channels: List[str], output_channels: List[str], filts: List[Filter]):
        self.__editing: Optional[Tuple[str, List[Node]], int] = None
        self.__stage = stage
        self.__filts = filts
        self.__output_channels = output_channels
        self.__input_channels = input_channels
        self.__nodes_by_name: Dict[str, Node] = {}
        self.__nodes_by_channel: Dict[str, Node] = {}
        self.__filter_pipes_by_channel: Dict[str, FilterPipe] = {}
        self.__regen()

    @property
    def stage(self):
        return self.__stage

    def __regen(self):
        '''
        Regenerates the graph.
        '''
        self.__nodes_by_name = {}
        for f in self.__filts:
            f.reset()
        self.__nodes_by_channel = self.__generate_nodes()
        self.__filter_pipes_by_channel = self.__generate_filter_paths()

    def __generate_nodes(self) -> Dict[str, Node]:
        '''
        Parses the supplied filters into a linked set of nodes per channel.
        :return: the linked node per channel.
        '''
        return self.__link(self.__create_nodes())

    @property
    def filter_pipes_by_channel(self) -> Dict[str, FilterPipe]:
        return self.__filter_pipes_by_channel

    def get_filter_at_node(self, node_name: str) -> Optional[Filter]:
        '''
        Locates the filter for the given node.
        :param node_name: the node to search for.
        :return: the filter, if any.
        '''
        node = self.get_node(node_name)
        if node and node.filt:
            return node.filt
        return None

    def get_filter_by_id(self, f_id: int) -> Optional[Filter]:
        '''
        Locates the filter with the given id.
        :param f_id: the filter id.
        :return: the filter, if any.
        '''
        return next((f for f in self.__filts if f.id == f_id), None)

    def get_node(self, node_name: str) -> Optional[Node]:
        '''
        Locates the named node.
        :param node_name: the node to search for.
        :return: the node, if any.
        '''
        return self.__nodes_by_name.get(node_name, None)

    def reorder(self, start: int, end: int, to: int) -> None:
        '''
        Moves the filters at indexes start to end to a new position & regenerates the graph.
        :param start: the starting index.
        :param end: the ending index.
        :param to: the position to move to.
        '''
        logger.info(f"Moved rows {start}:{end+1} to idx {to}")
        new_filters = self.filters[0: start] + self.filters[end+1:]
        to_insert = self.filters[start: end+1]
        for i, f in enumerate(to_insert):
            new_filters.insert(to + i - (1 if start < to else 0), f)
        logger.debug(f"Order: {[f for f in new_filters]}")
        self.__filts = new_filters
        self.__regen()

    @property
    def filters(self) -> List[Filter]:
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
        transforms each filter into a node, one per channel the filter is applicable to.
        :return: nodes by channel name.
        '''
        # create a channel/filter grid
        by_channel: Dict[str, List[Node]] = {c: [Node(0, f"IN:{c}", None, c)] if c not in SHORT_USER_CHANNELS else []
                                             for c in self.__output_channels}
        i = 1
        for idx, f in enumerate(self.__filts):
            if not isinstance(f, Divider) and f.enabled:
                for channel_name, nodes in by_channel.items():
                    channel_idx = get_channel_idx(channel_name)
                    if f.is_mine(channel_idx):
                        # mix is added as a node to both channels
                        if isinstance(f, Mix):
                            dst_channel_name = get_channel_name(f.dst_idx)
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
        if node.name in self.__nodes_by_name:
            logger.warning(f"Duplicate node name detected in {channel_name}!!! {node.name}")
        self.__nodes_by_name[node.name] = node
        filt.nodes.append(node)
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
        '''
        Applies mix operations to the nodes.
        :param by_channel: the simply linked nodes.
        :param orphaned_nodes: any orphaned nodes.
        :return: the remixed nodes.
        '''
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

    def __remix_orphans(self, by_channel, orphaned_nodes) -> None:
        '''
        Allows orphaned nodes to be considered in the remix.
        :param by_channel: the nodes by channel.
        :param orphaned_nodes: orphaned nodes.
        '''
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
        '''
        Applies a mix to a particular node.
        :param node: the node involved in the mix.
        :param by_channel: the entire graph so far.
        :param orphaned_nodes: unlinked node.
        '''
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
                               orphaned_nodes: Dict[str, List[Node]]) -> None:
        '''
        Applies an add or subtract mix operation if the filter owns the supplied node.
        :param by_channel: the filter graph.
        :param f: the mix filter.
        :param node: the node to remix.
        :param orphaned_nodes: unlinked nodes.
        '''
        if f.is_mine(get_channel_idx(node.channel)):
            dst_channel_name = get_channel_name(f.dst_idx)
            try:
                dst_node = self.__find_owning_node_in_channel(by_channel[dst_channel_name], f, dst_channel_name)
                node.add(dst_node, link_branch=True)
            except ValueError:
                dst_node = self.__find_owning_node_in_orphans(dst_channel_name, f, orphaned_nodes)
                if dst_node:
                    dst_node.visited = True
                    node.add(dst_node, link_branch=True)
                else:
                    logger.debug(f"No match for {f} in {dst_channel_name}, presumed orphaned")
                    node.visited = False

    def __find_owning_node_in_orphans(self, dst_channel_name, f, orphaned_nodes):
        return next((n for n in orphaned_nodes.get(dst_channel_name, [])
                     if self.__owns_filter(n, f, dst_channel_name)), None)

    def __copy_or_replace_node(self, by_channel: Dict[str, Node], channel_name: str, downstream: List[Node], f: Mix,
                               node: Node, orphaned_nodes: Dict[str, List[Node]]) -> List[Node]:
        '''
        Applies a copy or replace operation to the supplied node.
        :param by_channel: the filter graph.
        :param channel_name: the channel being mixed.
        :param downstream: the downstream nodes.
        :param f: the mix filter.
        :param node: the node that is the subject of the mix.
        :param orphaned_nodes: unlinked nodes.
        :return: the nodes that are now downstream of this node.
        '''
        src_channel_name = get_channel_name(f.src_idx)
        if src_channel_name == channel_name and node.channel == channel_name:
            dst_channel_name = get_channel_name(f.dst_idx)
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
        '''
        Applies a swap operation to the supplied node.
        :param by_channel: the filter graph.
        :param channel_name: the channel being mixed.
        :param downstream: the downstream nodes.
        :param f: the mix filter.
        :param node: the node that is the subject of the mix.
        :param orphaned_nodes: unlinked nodes.
        :return: the nodes that are now downstream of this node.
        '''
        src_channel_name = get_channel_name(f.src_idx)
        if src_channel_name == channel_name:
            dst_channel_name = get_channel_name(f.dst_idx)
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
        # TODO compare by id?
        return node.filt and node.filt == match and node.channel == owning_channel_name

    def __generate_filter_paths(self) -> Dict[str, FilterPipe]:
        '''
        Converts the filter graph into a filter pipeline per channel.
        :return: a FilterPipe by output channel.
        '''
        output_nodes = self.__get_output_nodes()
        filter_pipes: Dict[str, FilterPipe] = {}
        for channel_name, output_node in output_nodes.items():
            parent = self.__get_parent(output_node)
            filters: List[FilterOp] = []
            while parent is not None:
                f = parent.filter_op
                if f:
                    filters.append(f)
                parent = self.__get_parent(parent)
            filter_pipes[channel_name] = coalesce_ops(filters[::-1])
        return filter_pipes

    @staticmethod
    def __get_parent(node: Node) -> Optional[Node]:
        '''
        Provides the actual parent node for this node. If the node has multiple upstreams then it picks out the right
        parent node for an add operation to ensure we include the added (or subtracted channel).
        :param node: the node.
        :return: the parent node, if any.
        '''
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
        '''
        :param node: the node.
        :param upstream: the upstream node.
        :return: true if the upstream is a add or subtract mix operation landing in this node from a different channel.
        '''
        f = upstream.filt
        return isinstance(f, Mix) \
               and (f.mix_type == MixType.ADD or f.mix_type == MixType.SUBTRACT) \
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

    def start_edit(self, channel: str, node_chain: List[Node], insert_at: int):
        self.__editing = (channel, node_chain, insert_at)

    def end_edit(self, new_filters: CompleteFilter):
        '''
        Replaces the nodes identified by the node_chain with the provided set of new filters.
        :param new_filters: the filters.
        '''
        if self.__editing:
            node_chain: List[Node]
            channel_name, node_chain, insert_at = self.__editing
            channel_idx = str(get_channel_idx(channel_name))
            old_filters: List[Tuple[Filter, str]] = [(n.filt, n.channel) for n in node_chain]
            new_chain_filter: Optional[Filter] = None
            last_match = -1
            must_regen = False
            offset = 0
            # for each new filter
            # look for a matching old filter
            #   if match
            #     if skipped past old filters, delete them
            #   insert filter
            for i, f in enumerate(new_filters):
                new_filter = convert_filter_to_mc_dsp(f, channel_idx)
                if last_match < len(old_filters):
                    handled = False
                    for j in range(last_match + 1, len(old_filters)):
                        old_filter, filter_channel = old_filters[j]
                        old_filt_vals = old_filter.get_all_vals()
                        if pop_channels(new_filter.get_all_vals()) == pop_channels(old_filt_vals):
                            if (j - last_match) > 1:
                                offset -= self.__delete_filters(old_filters[last_match + 1:j])
                                must_regen = True
                            insert_at = self.__get_filter_idx(old_filter) + 1
                            offset = 0
                            last_match = j
                            new_chain_filter = old_filter
                            handled = True
                            break
                    if not handled:
                        must_regen = True
                        self.insert(new_filter, insert_at + offset)
                        new_chain_filter = new_filter
                        offset += 1
                else:
                    must_regen = True
                    self.insert(new_filter, insert_at + offset)
                    new_chain_filter = new_filter
                    offset += 1
            if last_match + 1 < len(old_filters):
                if self.__delete_filters(old_filters[last_match + 1:]):
                    must_regen = True
                    if not new_chain_filter:
                        new_chain_filter = old_filters[:last_match + 1][0][0]
            if must_regen:
                self.__regen()
                new_chain_node: Node = next((n for n in self.__collect_all_nodes(self.__nodes_by_channel)
                                             if n.channel == channel_name and n.filt and n.filt.id == new_chain_filter.id))
                editable_chain: List[Node] = new_chain_node.editable_node_chain[1]
                self.__editing = (channel_name, editable_chain, self.__get_filter_idx(editable_chain[0].filt) + 1)
            return must_regen

    def __get_filter_idx(self, to_match: Filter) -> int:
        return next((i for i, f in enumerate(self.__filts) if f.id == to_match.id))

    def __delete_filters(self, to_delete: List[Tuple[Filter, str]]) -> int:
        '''
        Removes the provided channel specific filters.
        :param to_delete: the channel-filter combinations to eliminate.
        :return: the number of deleted filters.
        '''
        deleted = 0
        for filt_to_delete, filt_channel_to_delete in to_delete:
            logger.debug(f"Deleting {filt_channel_to_delete} from {filt_to_delete}")
            if isinstance(filt_to_delete, ChannelFilter):
                filt_to_delete.pop_channel(filt_channel_to_delete)
                if not filt_to_delete.channels:
                    self.__delete(filt_to_delete)
                    deleted += 1
            else:
                self.__delete(filt_to_delete)
                deleted += 1
        return deleted

    def __delete(self, filt_to_delete):
        self.__filts.pop(self.__get_filter_idx(filt_to_delete))

    def clear_filters(self) -> None:
        '''
        Removes all filters.
        '''
        self.__filts = []
        self.__regen()

    def delete_channel(self, channel_filter: ChannelFilter, channel: str) -> bool:
        '''
        Removes the channel from the filter, deleting the filter itself if it has no channels left.
        :param channel_filter: the filter.
        :param channel: the channel.
        :returns true if the filter is deleted.
        '''
        logger.debug(f"Deleting {channel} from {channel_filter}")
        channel_filter.pop_channel(channel)
        deleted_channel = False
        if not channel_filter.channels:
            self.__delete(channel_filter)
            deleted_channel = True
        self.__regen()
        return deleted_channel

    def delete(self, filters: List[Filter]) -> None:
        '''
        Removes the provided list of filters.
        :param filters: the filters to remove.
        '''
        self.__filts = [f for f in self.__filts if f not in filters]
        self.__regen()

    def insert(self, to_insert: Filter, at: int, regen=True) -> None:
        '''
        Inserts a filter a specified position.
        :param to_insert: the filter to insert.
        :param at: the position to insert at.
        :param regen: react to the insertion by regenerating the graph.
        '''
        next_filt_id = self.__filts[at].id if at < len(self.__filts) else 2**31
        prev_filt_id = 0 if at == 0 else self.__filts[at-1].id
        self.__filts.insert(at, to_insert)
        # TODO improve this
        to_insert.id = prev_filt_id + int((next_filt_id - prev_filt_id) / 10)
        if to_insert.id >= next_filt_id:
            raise ValueError(f"Unable to insert filter at {at}, attempting to insert {to_insert.id} before {next_filt_id}")
        if regen:
            self.__regen()

    def replace(self, old_filter: Filter, new_filter: Filter) -> bool:
        '''
        Replaces the specified filter.
        :param old_filter: the filter to replace.
        :param new_filter:  the replacement filter.
        :return: true if it was replaced.
        '''
        try:
            self.__filts[self.__get_filter_idx(old_filter)] = new_filter
            self.__regen()
            return True
        except:
            return False


class GraphRenderer:

    def __init__(self, graph: FilterGraph, colours: Tuple[str, str] = None):
        self.__graph = graph
        self.__colours = colours

    def generate(self, vertical: bool, selected_nodes: Optional[Iterable[str]] = None) -> str:
        '''
        Generates a graphviz rendering of the graph.
        :param vertical: if true, use top to bottom alignment.
        :param selected_nodes: any nodes to highlight.
        :return: the rendered dot notation.
        '''
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

    def __create_channel_nodes(self, channel, nodes, selected_nodes: Optional[Iterable[str]] = None):
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

    def __add_node_definitions(self, selected_nodes: Optional[Iterable[str]] = None) -> Tuple[str, Dict[str, str]]:
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

    def __init__(self):
        self.node_id = None

    @abstractmethod
    def apply(self, input_signal: Signal) -> Signal:
        pass

    @property
    def ready(self):
        return True

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name[:-8] if class_name.endswith('FilterOp') else class_name} [{self.node_id}]"


class FilterPipe:

    def __init__(self, op: FilterOp, parent: Optional[FilterPipe]):
        self.op = op
        self.next: Optional[FilterPipe] = None
        self.parent = parent

    @property
    def ready(self):
        return self.op.ready

    def __repr__(self):
        if self.next:
            return f"{self.op} -> {self.next}"
        else:
            return f"{self.op}"


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
                    tmp = FilterPipe(last_sos, None)
                    root = tmp
                else:
                    tmp.next = FilterPipe(last_sos, parent=tmp)
                    tmp = tmp.next
                last_sos = None
            if root is None:
                tmp = FilterPipe(op, None)
                root = tmp
            else:
                tmp.next = FilterPipe(op, tmp)
                tmp = tmp.next
    if last_sos:
        if root is None:
            root = FilterPipe(last_sos, None)
        else:
            tmp.next = FilterPipe(last_sos, tmp)
    return root


class NopFilterOp(FilterOp):

    def __init__(self):
        super().__init__()

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal


class BranchFilterOp(FilterOp):

    def __init__(self, branch: FilterOp, node_id: str):
        super().__init__()
        self.node_id = node_id
        self.__branch = branch
        self.__source_signal: Optional[Signal] = None

    def apply(self, input_signal: Signal) -> Signal:
        self.__source_signal = input_signal
        return input_signal

    @property
    def source_signal(self) -> Optional[Signal]:
        return self.__source_signal

    def is_source_for(self, target: FilterOp) -> bool:
        return self.__branch == target


class SosFilterOp(FilterOp):

    def __init__(self, sos: List[List[float]]):
        super().__init__()
        self.__sos = sos

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.sosfilter(self.__sos)

    def extend(self, more_sos: SosFilterOp):
        self.__sos += more_sos.__sos


class DelayFilterOp(FilterOp):

    def __init__(self, delay_millis: float, fs: int = 48000):
        super().__init__()
        self.__shift_samples = int((delay_millis / 1000) / (1.0 / fs))

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.shift(self.__shift_samples)


class AddFilterOp(FilterOp):

    def __init__(self):
        super().__init__()
        self.__other: Optional[Signal] = None

    def accept(self, signal: Signal):
        if self.ready:
            raise ValueError(f"Attempting to reuse AddFilterOp")
        self.__other = signal

    @property
    def ready(self):
        return self.__other is not None

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.add(self.__other.samples)


class SubtractFilterOp(FilterOp):

    def __init__(self):
        super().__init__()
        self.__other: Optional[Signal] = None

    def accept(self, signal: Signal):
        if self.ready:
            raise ValueError(f"Attempting to reuse AddFilterOp")
        self.__other = signal

    @property
    def ready(self):
        return self.__other is not None

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.subtract(self.__other.samples)


class GainFilterOp(FilterOp):

    def __init__(self, gain_db: float):
        super().__init__()
        self.__gain_db = gain_db

    def apply(self, input_signal: Signal) -> Signal:
        return input_signal.adjust_gain(10 ** (self.__gain_db / 20.0))


def collect_nodes(node: Node, arr: List[Node]) -> List[Node]:
    if node not in arr:
        arr.append(node)
    for d in node.downstream:
        collect_nodes(d, arr)
    return arr


def make_signal(channel: str):
    fs = 48000
    return Signal(channel, unit_impulse(fs*4, 'mid') * 23453.66, fs=fs)


class JRiverFilterPipelineDialog(QDialog, Ui_jriverGraphDialog):

    def __init__(self, dot: str, on_change, parent):
        super(JRiverFilterPipelineDialog, self).__init__(parent)
        self.setupUi(self)
        self.source.setPlainText(dot)
        if on_change:
            self.source.textChanged.connect(lambda: on_change(self.source.toPlainText()))


def pop_channels(vals: List[Dict[str, str]]):
    '''
    :param vals: a set of filter values.
    :return: the values without the Channel key.
    '''
    return [{k: v for k, v in d.items() if k != 'Channels'} for d in vals]


def render_dot(txt):
    from graphviz import Source
    return Source(txt, format='svg').pipe()


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
            self.millis.setValue(float(self.__vals['Delay']))
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
                for item in self.channelList.findItems(c, Qt.MatchCaseSensitive):
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
        self.__vals = vals if vals else {}
        self.__vals['Mode'] = str(mix_type.value)
        self.setWindowTitle(f"{mix_type.name.capitalize()}")
        self.gain.valueChanged.connect(lambda v: self.__set_val('Gain', f"{v:.7g}"))
        self.source.currentTextChanged.connect(lambda v: self.__set_val('Source', str(get_channel_idx(v))))
        self.destination.currentTextChanged.connect(lambda v: self.__set_val('Destination', str(get_channel_idx(v))))
        self.__configure_channel_list(channels)
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
            self.gain.setValue(float(self.__vals['Gain']))
        self.__validators.append(lambda: self.source.currentText() != self.destination.currentText())

    def __enable_accept(self):
        self.buttonBox.button(QDialogButtonBox.Save).setEnabled(all(v() for v in self.__validators))


class JRiverChannelOnlyFilterDialog(QDialog, Ui_jriverChannelSelectDialog):

    def __init__(self, parent: QDialog, channels: List[str], on_save: Callable[[Dict[str, str]], None],
                 vals: Dict[str, str] = None, title: str = 'Polarity', multi: bool = True):
        super(JRiverChannelOnlyFilterDialog, self).__init__(parent)
        self.setupUi(self)
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


class JRiverParser:

    def __init__(self, block=0, channels=('Subwoofer',)):
        self.__block = get_peq_key_name(block)
        self.__target_channels = ';'.join([str(JRIVER_CHANNELS.index(c)) for c in channels])

    def convert(self, dst, filt: ComplexFilter, **kwargs):
        from model.minidsp import flatten_filters
        flat_filts: List[SOS] = flatten_filters(filt)
        config_txt = Path(dst).read_text()
        if len(flat_filts) > 0:
            logger.info(f"Copying {len(flat_filts)} to {dst}")
            # generate the xml formatted filters
            xml_filts = [filts_to_xml(convert_filter_to_mc_dsp(f, self.__target_channels).get_all_vals())
                         for f in flat_filts]
            config_txt = include_filters_in_dsp(self.__block, config_txt, xml_filts, replace=False)
        else:
            logger.warning(f"Nop for empty filter file {dst}")
        return config_txt, False

    @staticmethod
    def file_extension():
        return '.dsp'

    @staticmethod
    def newline():
        return '\r\n'
