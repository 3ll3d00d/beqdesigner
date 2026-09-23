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
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from model.codec import bassmanagedsignaldata_to_json, signaldata_to_json
from model.iir import CompleteFilter
from model.preferences import BASS_MANAGEMENT_LPF_FS, BASS_MANAGEMENT_LPF_POSITION
from model.signal import BassManagedSignalData, SingleChannelSignalData

from pipeline.orchestrate import Session


def _filter_hash(filters: CompleteFilter) -> str:
    ''' sha256 of the filter's canonical JSON -- what pipeline_filter_hash stores/compares against. '''
    import hashlib
    import json
    return hashlib.sha256(json.dumps(filters.to_json(), sort_keys=True).encode('utf-8')).hexdigest()


def _master_json(project: dict) -> dict:
    '''The filter-bearing channel in a mono, legacy flat, or bass-managed project.'''
    return project['channels'][0] if project['_type'] == 'BassManagedSignalData' else project


def write_project(path: str, signals: Sequence[SingleChannelSignalData | BassManagedSignalData],
                  filter_hash: Optional[str] = None) -> None:
    '''
    Writes a .beq project -- the same gzip+JSON shape app.py's exportProject()/importProject() use
    (model.codec's signal serializers). If filter_hash is given, stamps it as an extra
    'pipeline_filter_hash' key on the first filter-bearing channel's dict -- an additive key
    signaldata_from_json()/signalmodel_from_json() simply don't look for, so it round-trips fine through
    the interactive app's own load path but is never re-emitted by a human's re-export (app.py's
    exportProject() only ever calls the generic signaldata_to_json(), which has no concept of this key) --
    that asymmetry is the edit-detection mechanism (see read_project_filter()).
    '''
    import gzip
    import json
    output = [bassmanagedsignaldata_to_json(s) if isinstance(s, BassManagedSignalData) else signaldata_to_json(s)
              for s in signals]
    if filter_hash is not None and output:
        _master_json(output[0])['pipeline_filter_hash'] = filter_hash
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
    preferences = session.preferences
    composite = BassManagedSignalData(channels, preferences.get(BASS_MANAGEMENT_LPF_FS),
                                      preferences.get(BASS_MANAGEMENT_LPF_POSITION), preferences)
    write_project(out_path, [composite], filter_hash=_filter_hash(filters))


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
    master = _master_json(data[0])
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


@dataclass(frozen=True)
class PublishedFilter:
    filter: CompleteFilter
    edited_side: Optional[str]  # None: both projects still pipeline-pure. 'mono'/'multichannel': that project
                                # carries a human edit the other lacks. 'both': edited on both sides, identically.


def resolve_published_projects(mono_path: str, multichannel_path: Optional[str] = None) -> PublishedFilter:
    '''
    §3.3.1's three-way resolution, saying *which* project carried the edit.
    :raises ProjectFilterConflict: if both projects were independently edited and disagree.
    '''
    mono_filter, mono_pure = read_project_filter(mono_path)
    if multichannel_path is None or not os.path.isfile(multichannel_path):
        return PublishedFilter(mono_filter, None if mono_pure else 'mono')
    mc_filter, mc_pure = read_project_filter(multichannel_path)
    if not mono_pure and not mc_pure:
        if mono_filter.to_json() != mc_filter.to_json():
            raise ProjectFilterConflict(mono_filter, mc_filter)
        return PublishedFilter(mono_filter, 'both')
    if not mc_pure:
        return PublishedFilter(mc_filter, 'multichannel')
    return PublishedFilter(mono_filter, None if mono_pure else 'mono')


def preview_published_projects(mono_path: str, multichannel_path: Optional[str], candidate: CompleteFilter
                               ) -> PublishedFilter:
    '''
    What resolve_published_projects() will return once publishing has run write_title_projects_if_safe() with
    `candidate`, worked out without writing anything: a project that is missing, or still pipeline-pure, is one
    publishing would (re)write from the candidate, so it counts as holding `candidate`; a hand-edited one holds its
    edit. Lets discovery ask "would publishing change the catalogue?" without touching the working directory.
    :param multichannel_path: the multichannel project's path if the title has a multichannel extraction (whether or
        not the project has been written yet), else None.
    :raises ProjectFilterConflict: as resolve_published_projects().
    '''
    def read(path: Optional[str]) -> Tuple[CompleteFilter, bool]:
        if path is None or not os.path.isfile(path):
            return candidate, True
        current, pure = read_project_filter(path)
        return (candidate if pure else current), pure

    mono_filter, mono_pure = read(mono_path)
    if multichannel_path is None:
        return PublishedFilter(mono_filter, None if mono_pure else 'mono')
    mc_filter, mc_pure = read(multichannel_path)
    if not mono_pure and not mc_pure:
        if mono_filter.to_json() != mc_filter.to_json():
            raise ProjectFilterConflict(mono_filter, mc_filter)
        return PublishedFilter(mono_filter, 'both')
    if not mc_pure:
        return PublishedFilter(mc_filter, 'multichannel')
    return PublishedFilter(mono_filter, None if mono_pure else 'mono')


def edited_projects(mono_path: Optional[str], multichannel_path: Optional[str] = None) -> Optional[str]:
    '''
    Whether a person has changed a title's filter in its `.beq` project since the pipeline wrote it -- the "edited
    project" fact bulk accept must not override (design.md §12.9). Reads only what exists: a project that has not
    been written yet is not edited.
    :return: None if every project is still exactly what the pipeline wrote; else `mono`, `multichannel` or `both`.
    :raises OSError/ValueError/KeyError: if a project exists but cannot be read.
    '''
    edited = [name for name, path in (('mono', mono_path), ('multichannel', multichannel_path))
              if path and os.path.isfile(path) and not read_project_filter(path)[1]]
    if not edited:
        return None
    return edited[0] if len(edited) == 1 else 'both'


def resolve_published_filter(mono_path: str, multichannel_path: Optional[str] = None) -> Tuple[CompleteFilter, bool]:
    '''
    :return: (the filter to publish, True if it came from a human edit on either side, False if both projects
        are still pipeline-pure). See resolve_published_projects() for which side.
    :raises ProjectFilterConflict: if both projects were independently edited and disagree.
    '''
    published = resolve_published_projects(mono_path, multichannel_path)
    return published.filter, published.edited_side is not None


def align_projects(session: Session, published: PublishedFilter, mono_path: str, mono_wav_path: str,
                   multichannel_path: Optional[str] = None, multichannel_wav_path: Optional[str] = None,
                   channel_layout_name: str = 'unknown') -> List[str]:
    '''
    Writes a human's edit into the *other* project, so the two files agree with what was published rather than
    one silently going stale (§3.3.1). Only ever rewrites a pipeline-pure project -- by construction of
    PublishedFilter.edited_side the sibling of an edited side is pure (or absent), so no edit is lost.
    :return: which projects were rewritten, e.g. ['multichannel'].
    '''
    if published.edited_side == 'mono' and multichannel_path and multichannel_wav_path \
            and os.path.isfile(multichannel_path):
        write_multichannel_project(session, multichannel_wav_path, published.filter, channel_layout_name,
                                   multichannel_path)
        return ['multichannel']
    if published.edited_side == 'multichannel':
        write_mono_project(session, mono_wav_path, published.filter, mono_path)
        return ['mono']
    return []
