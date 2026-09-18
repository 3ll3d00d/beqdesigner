'''
pipeline/publish/project.py: `.beq` project files as the published source --
design/library-sync-pipeline-plan.md §3.3/§3.3.1, Appendix B (chunk 2).

Output 1 (§3.3) is a mono `.beq` project file per title, and (optionally) a
linked multichannel one -- the same gzip+JSON shape app.py's
exportProject()/importProject() use, built with model.codec.signaldata_to_json()
per signal. §3.3.1's key correction: once that project exists, a human
editing it in the interactive app is what must reach beqcatalogue, not the
review queue's frozen designer output -- so this module also owns the
edit-detection mechanism (a `pipeline_filter_hash` stamped onto the
master/mono signal's dict, an additive key the interactive app's own
generic signaldata_to_json() never re-emits on a human's re-save) and the
two-sided (mono vs. multichannel) resolution/conflict logic
publish_reviewed_queue() needs.
'''
import os
from typing import Optional, Sequence, Tuple

from model.iir import CompleteFilter
from model.signal import SingleChannelSignalData

from pipeline.orchestrate import Session


def _filter_hash(filters: CompleteFilter) -> str:
    ''' sha256 of the filter's canonical JSON -- what pipeline_filter_hash stores/compares against. '''
    import hashlib
    import json
    return hashlib.sha256(json.dumps(filters.to_json(), sort_keys=True).encode('utf-8')).hexdigest()


def write_project(path: str, signals: Sequence[SingleChannelSignalData], filter_hash: Optional[str] = None) -> None:
    '''
    Writes a .beq project -- the same gzip+JSON shape app.py's exportProject()/importProject() use
    (model.codec.signaldata_to_json() per signal, no BassManagedSignalData wrapper). If filter_hash is
    given, stamps it as an extra 'pipeline_filter_hash' key on signals[0]'s dict -- an additive key
    signaldata_from_json()/signalmodel_from_json() simply don't look for, so it round-trips fine through
    the interactive app's own load path but is never re-emitted by a human's re-export (app.py's
    exportProject() only ever calls the generic signaldata_to_json(), which has no concept of this key) --
    that asymmetry is the edit-detection mechanism (see read_project_filter()).
    '''
    import gzip
    import json
    from model.codec import signaldata_to_json
    output = [signaldata_to_json(s) for s in signals]
    if filter_hash is not None and output:
        output[0]['pipeline_filter_hash'] = filter_hash
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with gzip.open(path, 'wb') as f:
        f.write(json.dumps(output).encode('utf-8'))


def write_mono_project(session: Session, mono_wav_path: str, filters: CompleteFilter, out_path: str) -> None:
    sig = session.load(mono_wav_path)
    session.set_filters(sig, filters)
    write_project(out_path, [sig], filter_hash=_filter_hash(filters))


def write_multichannel_project(session: Session, multichannel_wav_path: str, filters: CompleteFilter,
                               channel_layout_name: str, out_path: str) -> None:
    '''
    Loads every channel (Session.load_channel_signals()), applies `filters` to the first (the master),
    and enslave()s every other channel -- including LFE, which needs no special handling beyond already
    being named correctly by load_channel_signals() -- to it.
    '''
    channels = session.load_channel_signals(multichannel_wav_path, channel_layout_name=channel_layout_name)
    master = channels[0]
    session.set_filters(master, filters)
    for slave in channels[1:]:
        master.enslave(slave)
    write_project(out_path, channels, filter_hash=_filter_hash(filters))


def read_project_filter(path: str) -> Tuple[CompleteFilter, bool]:
    '''
    :return: (the master/mono signal's current filter, True if pipeline_filter_hash matches a fresh hash
        of it -- i.e. still exactly what the pipeline last wrote -- False if a human has changed it since,
        or the file predates this mechanism and never had a hash).
    '''
    import gzip
    import json
    from model.codec import filter_from_json
    with gzip.open(path, 'rb') as f:
        data = json.loads(f.read().decode('utf-8'))
    master = data[0]
    filt = filter_from_json(master['filter_presets'][master['active_filter_preset']])
    stored = master.get('pipeline_filter_hash')
    return filt, stored is not None and stored == _filter_hash(filt)


def _is_safe_to_overwrite(path: Optional[str]) -> bool:
    ''' True if path doesn't exist yet, or exists and is still pipeline-pure (no human edit to lose). '''
    if path is None or not os.path.isfile(path):
        return True
    _, is_pure = read_project_filter(path)
    return is_pure


def write_title_projects_if_safe(session: Session, mono_wav_path: str, filters: CompleteFilter, mono_out_path: str,
                                 multichannel_wav_path: Optional[str] = None, channel_layout_name: str = 'unknown',
                                 multichannel_out_path: Optional[str] = None) -> dict:
    '''
    The one entry point both design_and_queue() and publish_reviewed_queue() call. Writes each project
    only if it's safe to (§3.3.1's hash gate) -- an existing human-edited file is left untouched, the other
    one (if any) still gets written/updated normally. Both targets are independent; one being unsafe never
    blocks the other.
    :return: {'mono': True|False, 'multichannel': True|False|None} -- True=written, False=skipped (existing
        file was human-edited), None=not applicable (no multichannel_wav_path given).
    '''
    result = {'mono': False, 'multichannel': None}
    if _is_safe_to_overwrite(mono_out_path):
        write_mono_project(session, mono_wav_path, filters, mono_out_path)
        result['mono'] = True
    if multichannel_wav_path is not None:
        if _is_safe_to_overwrite(multichannel_out_path):
            write_multichannel_project(session, multichannel_wav_path, filters, channel_layout_name,
                                       multichannel_out_path)
            result['multichannel'] = True
        else:
            result['multichannel'] = False
    return result


class ProjectFilterConflict(Exception):
    ''' Raised by resolve_published_filter() when both the mono and multichannel projects were
    independently edited and now disagree -- this plan deliberately does not guess which one wins. '''
    def __init__(self, mono_filter: CompleteFilter, multichannel_filter: CompleteFilter):
        super().__init__('mono and multichannel projects have independently edited, conflicting filters')
        self.mono_filter = mono_filter
        self.multichannel_filter = multichannel_filter


def resolve_published_filter(mono_path: str, multichannel_path: Optional[str] = None) -> Tuple[CompleteFilter, bool]:
    '''
    §3.3.1's three-way resolution. :return: (the filter to publish, True if it came from a human edit on
    either side, False if both projects are still pipeline-pure).
    :raises ProjectFilterConflict: if both projects were independently edited and disagree.
    '''
    mono_filter, mono_pure = read_project_filter(mono_path)
    if multichannel_path is None or not os.path.isfile(multichannel_path):
        return mono_filter, not mono_pure
    mc_filter, mc_pure = read_project_filter(multichannel_path)
    if not mono_pure and not mc_pure:
        if mono_filter.to_json() != mc_filter.to_json():
            raise ProjectFilterConflict(mono_filter, mc_filter)
        return mono_filter, True
    if not mc_pure:
        return mc_filter, True
    return mono_filter, not mono_pure
