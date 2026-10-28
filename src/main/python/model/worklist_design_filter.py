'''
The work list's *Design* filter and the out-of-date count on the drift banner -- design/design-version.md ("Caller
behaviour" 3). A mixin of `model.worklist.WorkListWindow`:

* **Design: Any / Declined / Out of date** beside the source combo: the titles whose design the designer declined, or that
  a redesign now could answer differently (`Selection.declined` / `out_of_date`, as the CLI and the service's titles
  filter select them), so `current_selection()` and the buttons work on the same titles the table lists.
* **What "now" is**: the profile's designer, if it has HTTP endpoints, is asked (`pipeline.library.setup.designer_identity`, the service's own
  question) on a worker whenever the setup is read. Until it answers, or if no endpoint that is up says its design
  version, *Out of date* is not offered and the banner counts nothing; its tooltip says why.
* **The count**: the drift banner also says how many designs are out of date, with *Show them*, which picks the filter.
  It is worked out from the rows the window holds, so it follows every refresh.
'''
import logging
from dataclasses import replace
from typing import Dict, List, Optional

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal
from qtpy.QtWidgets import QComboBox, QPushButton

from model.worklist_model import DESIGN_ANY, DESIGN_DECLINED, DESIGN_OUT_OF_DATE, design_accepts
from pipeline.designer.identity import CurrentIdentity
from pipeline.library.selection import Selection
from pipeline.library.setup import configured_values, designer_endpoints, designer_identity

logger = logging.getLogger('worklist')

NOT_ASKED = 'The designer has not been asked yet what it designs with.'
NO_IDENTITY = 'No endpoint of the designer that is up says its design version, so no design can be judged out of date.'


class _IdentitySignals(QObject):
    finished = Signal(object)


class IdentityJob(QRunnable):
    ''' Asks the designer's endpoints what a redesign now would be made with; the answer is (token, identity or None). '''

    def __init__(self, token: int, values: dict, config: dict):
        super().__init__()
        self.signals = _IdentitySignals()
        self._token, self._values, self._config = token, values, config

    def run(self):
        try:
            current = designer_identity(self._values, self._config)
        except Exception:   # a courtesy: a designer that cannot be asked leaves the filter unoffered
            logger.exception('Could not ask the designer what it designs with')
            current = None
        self.signals.finished.emit((self._token, current))


def out_of_date_text(count: int, current: CurrentIdentity) -> str:
    ''' The banner's sentence about the designs a redesign now could answer differently. '''
    designs = '1 design was' if count == 1 else f'{count:,} designs were'
    return (f'{designs} made with another design version or settings than the designer uses now '
            f'({current.describe()}); a redesign could answer {"it" if count == 1 else "them"} differently.')


class WorkListDesignFilter:
    '''
    The mixin: it uses `_setup`, `_model`, `_proxy`, `_closed`, `_refresh_view()`, `_drift_ids` and the widgets
    `filterLayout`, `sourceCombo`, `driftBannerLayout` and `driftDismissButton` of the window.
    '''

    def _init_design_filter(self) -> None:
        ''' The state, before setupUi reads the properties. '''
        self._identity_token = 0
        self._identity_jobs: Dict[int, IdentityJob] = {}
        self._identity_reason = NOT_ASKED

    def _configure_design_filter(self) -> None:
        self.designFilterCombo = QComboBox(self.sourceCombo.parentWidget())
        self.designFilterCombo.setObjectName('designFilterCombo')
        for text, value in (('Any design', DESIGN_ANY), ('Declined', DESIGN_DECLINED), ('Out of date', DESIGN_OUT_OF_DATE)):
            self.designFilterCombo.addItem(text, value)
        self.designFilterCombo.setToolTip('Only the titles whose design the designer declined, or that a redesign now '
                                          'could answer differently (another design version or settings).')
        self.filterLayout.insertWidget(self.filterLayout.indexOf(self.sourceCombo) + 1, self.designFilterCombo)
        self.designFilterCombo.currentIndexChanged.connect(self._on_design_filter)
        self.driftOutOfDateButton = QPushButton('Show them', self.driftBannerButton.parentWidget())
        self.driftOutOfDateButton.setObjectName('driftOutOfDateButton')
        self.driftOutOfDateButton.setToolTip('List only the designs that are out of date')
        self.driftOutOfDateButton.clicked.connect(lambda: self.set_design_filter(DESIGN_OUT_OF_DATE))
        self.driftBannerLayout.insertWidget(self.driftBannerLayout.indexOf(self.driftDismissButton),
                                            self.driftOutOfDateButton)
        self._offer_out_of_date()

    @property
    def current_identity(self) -> Optional[CurrentIdentity]:
        ''' What a redesign now would be made with, as the designer last said; None if it has not or cannot. '''
        return self._proxy.current_identity

    @property
    def design_filter(self) -> str:
        return self._proxy.design_filter

    def set_design_filter(self, design: str) -> None:
        ''' Picks *Any*, *Declined* or *Out of date* (DESIGN_*); out of date only once the designer has said. '''
        if design == DESIGN_OUT_OF_DATE and self.current_identity is None:
            raise ValueError(self._identity_reason)
        self.designFilterCombo.setCurrentIndex(self.designFilterCombo.findData(design))

    def _on_design_filter(self, _index: int) -> None:
        self._proxy.set_design_filter(self.designFilterCombo.currentData())
        self._refresh_view()

    def out_of_date_ids(self) -> List[str]:
        ''' The titles the window holds whose design a redesign now could answer differently (empty until the designer says). '''
        current = self.current_identity
        return [r.id for r in self._model.rows if design_accepts(DESIGN_OUT_OF_DATE, r, current)] if current else []

    def design_selection(self, selection: Selection) -> Selection:
        ''' `selection` narrowed by the *Design* filter, as `current_selection()` returns it. '''
        design = self._proxy.design_filter
        if design == DESIGN_DECLINED:
            return replace(selection, declined=True)
        if design == DESIGN_OUT_OF_DATE:
            return replace(selection, out_of_date=True, current=self.current_identity)
        return selection

    def refresh_identity(self) -> None:
        ''' Asks (on a worker) the profile's designer what it designs with now; an answer to an older question is dropped. '''
        self._identity_token += 1
        token, setup = self._identity_token, self._setup
        if self._closed or setup.profile is None or setup.settings is None or not setup.settings.designer:
            self._set_identity(None, NO_IDENTITY)
            return
        config = dict(setup.profile.config or {})
        try:
            values = configured_values(config, 'run')
            values['designer'] = setup.settings.designer
            declared = designer_endpoints(values, config)
        except ValueError as error:
            self._set_identity(None, str(error))
            return
        if declared is None or not declared.endpoints:   # the manual designer, or one with no address: nothing to ask
            self._set_identity(None, NO_IDENTITY)
            return
        job = IdentityJob(token, values, config)
        self._identity_jobs[token] = job
        job.signals.finished.connect(self._on_identity)
        QThreadPool.globalInstance().start(job)

    def _on_identity(self, result) -> None:
        token, current = result
        self._identity_jobs.pop(token, None)
        if token != self._identity_token or self._closed:
            return
        self._set_identity(current, '' if current else NO_IDENTITY)

    def _set_identity(self, current: Optional[CurrentIdentity], reason: str) -> None:
        self._identity_reason = reason
        if current is None and self._proxy.design_filter == DESIGN_OUT_OF_DATE:
            self.designFilterCombo.setCurrentIndex(self.designFilterCombo.findData(DESIGN_ANY))
        self._proxy.set_current_identity(current)
        self._offer_out_of_date()
        self._refresh_view()
        self._refresh_drift_banner()

    def _offer_out_of_date(self) -> None:
        ''' *Out of date* is offered only once the designer has said what it designs with. '''
        item = self.designFilterCombo.model().item(self.designFilterCombo.findData(DESIGN_OUT_OF_DATE))
        current = self.current_identity
        item.setEnabled(current is not None)
        item.setToolTip(f'Designed otherwise than {current.describe()}' if current else self._identity_reason)
