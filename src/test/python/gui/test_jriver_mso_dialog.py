'''
The JRiver MSO import dialog's channel check: it must judge an MSO filter's channels against the same
channel pool the dsp itself uses (legacy numbered pre-35.0.39, Atmos+Extra from 35.0.39), otherwise a
filter on a 16 channel format's LW/RW is refused as "Unsupported channels".
'''
import ui.beq  # noqa: F401 -- must precede model.jriver.ui, see AGENTS.md's circular-import note

import json

import pytest
from qtpy.QtWidgets import QDialogButtonBox, QWidget

from model.jriver.formats import OUTPUT_FORMATS, get_channel_indexes
from model.jriver.parser import from_mso
from model.jriver.ui import MSODialog


def _mso_filter(chans):
    return from_mso(json.dumps({'mso_filters': [
        {'_type': 'PeakingEQ', 'chans': chans, 'gain': -3.0, 'fc': 40.0, 'q': 1.0},
    ]}))


@pytest.mark.parametrize('use_atmos_channels', [True, False])
def test_has_channels_follows_the_channel_pool(use_atmos_channels):
    fmt = OUTPUT_FORMATS['SIXTEEN']
    assert fmt.has_channels(get_channel_indexes(['LW', 'RW']), use_atmos_channels) is use_atmos_channels
    assert fmt.has_channels(get_channel_indexes(['C9', 'C10']), use_atmos_channels) is not use_atmos_channels


@pytest.mark.parametrize('use_atmos_channels,accepted', [(True, True), (False, False)])
def test_mso_filter_on_wide_channels(qtbot, use_atmos_channels, accepted):
    parent = QWidget()
    qtbot.addWidget(parent)
    dialog = MSODialog(_mso_filter(['LW', 'RW']), lambda _: None, OUTPUT_FORMATS['SIXTEEN'], parent,
                       use_atmos_channels=use_atmos_channels)
    qtbot.addWidget(dialog)

    assert dialog.buttonBox.button(QDialogButtonBox.StandardButton.Ok).isEnabled() is accepted
    if accepted:
        assert dialog.status.text() == ''
    else:
        assert dialog.status.text() == 'Unsupported channels for 16 channels - LW, RW'
