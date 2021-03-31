from __future__ import annotations

import itertools
import logging
import os
import sys
import time
import xml.etree.ElementTree as et
from abc import ABC, abstractmethod
from builtins import isinstance
from collections import defaultdict, Sequence
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Callable, Union, Set, Iterable, overload, Type

import math
import qtawesome as qta
from qtpy.QtCore import QPoint, QModelIndex, Qt
from qtpy.QtGui import QColor, QPalette, QKeySequence, QCloseEvent
from qtpy.QtWidgets import QDialog, QFileDialog, QMenu, QAction, QListWidgetItem, QAbstractItemView, \
    QDialogButtonBox, QMessageBox, QInputDialog
from scipy.signal import unit_impulse

from model import iir, JRIVER_SHORT_CHANNELS, JRIVER_CHANNELS, SHORT_USER_CHANNELS, USER_CHANNELS, \
    JRIVER_SHORT_NAMED_CHANNELS
from model.filter import FilterModel, FilterDialog
from model.iir import s_to_q, SOS, CompleteFilter, SecondOrder_HighPass, PeakingEQ, LowShelf as LS, Gain as G, \
    LinkwitzTransform as LT, CompoundPassFilter, ComplexHighPass, BiquadWithQGain, q_to_s, SecondOrder_LowPass, \
    ComplexLowPass, FilterType, FirstOrder_LowPass, ComplexFilter as CF, PassFilter, FirstOrder_HighPass
from model.limits import DecibelRangeCalculator, PhaseRangeCalculator
from model.log import to_millis
from model.magnitude import MagnitudeModel
from model.preferences import JRIVER_GEOMETRY, JRIVER_GRAPH_X_MIN, JRIVER_GRAPH_X_MAX, JRIVER_DSP_DIR, Preferences, \
    get_filter_colour
from model.signal import Signal
from model.xy import MagnitudeData
from ui.jriver import Ui_jriverDspDialog
from ui.channel_select import Ui_channelSelectDialog
from ui.jriver_delay_filter import Ui_jriverDelayDialog
from ui.jriver_mix_filter import Ui_jriverMixDialog
from ui.pipeline import Ui_jriverGraphDialog



