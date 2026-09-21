'''The built-in manual design path: queue an editable, flat filter.'''
from pipeline.designer.contract import BiquadSpec, DesignCandidate, DesignResponse

MANUAL_DESIGNER = 'manual'


def manual_response() -> DesignResponse:
    '''One 0 dB PEQ is a valid no-op and gives the editor an obvious first row.'''
    return DesignResponse(contract_version='1.0', candidates=[DesignCandidate(
        filters=[BiquadSpec(type='peaking_eq', freq_hz=20.0, gain_db=0.0, q=1.0)],
        confidence=0.0, mv_adjust_db=0.0, method='non_parametric',
        commentary={'Manual filter': 'No automatic design was applied. Open the project and edit this flat filter.'})])
