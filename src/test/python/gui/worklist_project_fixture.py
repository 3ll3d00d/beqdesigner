'''
`.beq` projects for the title page tests: the real files `pipeline.publish.project` writes into `<work_dir>/<id>/`, and the
edits a person makes to them in the main window (`test_pipeline_publish_project`'s helpers), so what the page reads is what the
pipeline and the app leave.
'''
import os
from typing import Optional, Tuple

from model.iir import CompleteFilter, LowShelf
from pipeline.config import AnalysisConfig
from pipeline.orchestrate import Session
from pipeline.publish.project import write_mono_project, write_multichannel_project
from test_pipeline_publish_project import _HUMAN_FILTER, _hand_edit_filter, _read_raw, _write_mono_wav, \
    _write_multichannel_wav, _write_raw

PIPELINE_FILTER = CompleteFilter(fs=1000, filters=[LowShelf(1000, 20, 0.7, 4.5)])


def write_projects(work_dir, title_id: str, multichannel: bool = False) -> Tuple[str, Optional[str]]:
    '''
    The mono project (and, if asked, the multichannel one, with the `multichannel.wav` that makes the title a multichannel
    title) exactly as the pipeline writes them.
    :return: (mono project path, multichannel project path or None)
    '''
    directory = os.path.join(str(work_dir), title_id)
    os.makedirs(directory, exist_ok=True)
    session = Session(AnalysisConfig())
    mono_wav = os.path.join(directory, 'mono.wav')
    _write_mono_wav(mono_wav)
    mono = os.path.join(directory, f'{title_id}.mono.beq')
    write_mono_project(session, mono_wav, PIPELINE_FILTER, mono)
    if not multichannel:
        return mono, None
    mc_wav = os.path.join(directory, 'multichannel.wav')
    _write_multichannel_wav(mc_wav, (1000, 2000, 3000, 4000, 5000, 6000))
    mc = os.path.join(directory, f'{title_id}.multichannel.beq')
    write_multichannel_project(session, mc_wav, PIPELINE_FILTER, '5.1', mc)
    return mono, mc


def edit_project(path: str) -> None:
    ''' A person changed the filter in the main window and saved the project over this file. '''
    _hand_edit_filter(path, _HUMAN_FILTER)


def resave_unchanged(path: str) -> None:
    ''' Opened and saved again without changing anything: the main window's save carries no pipeline stamp. '''
    data = _read_raw(path)
    (data[0]['channels'][0] if data[0]['_type'] == 'BassManagedSignalData' else data[0]).pop(
        'pipeline_filter_hash', None)
    _write_raw(path, data)


def corrupt_project(path: str) -> None:
    with open(path, 'wb') as f:
        f.write(b'this is not a gzip file')
