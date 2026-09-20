'''
Reading a title's stage states off its outputs -- design/library-sync/workflow-rework/design.md §12.5/§12.6.

Discovery's "read" half. Given a title (a LibraryItem, or a TV SeasonGroup) this looks at what is on disk -- the
extract manifest, the queue entry, the `.beq` projects, the two catalogue repos -- and fills in a state.StageStates for
state.derive_needs(). It **never extracts, designs, publishes or commits**, and it uses the very functions the run
uses to decide what to do (`extract_status()`, `design_status()`, `current_publish_digest()`), so "what would run"
cannot drift from "what runs".

What it reads, and what it does not:

- the extract manifest and the wav's existence (files this program wrote);
- the queue entry -- but only when its file changed since the last scan (the index caches a summary keyed by mtime and
  size), because an entry carries the whole average curve;
- for an accepted or published entry, the projects and the artwork file, to work out the *current* published digest --
  again only when something changed;
- one `git status` and one `git diff` per repo (state.repo_state), however many titles;
- **never** the media: a JRiver item's fingerprint is the library's own, and a filesystem item's is one `stat`.
'''
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from pipeline.config import AnalysisConfig
from pipeline.designer.contract import Coverage
from pipeline.library.design_cache import design_status
from pipeline.library.extract_cache import extract_params_hash, extract_status, read_manifest, \
    source_fingerprint
from pipeline.library.season import DEFAULT_TV_MODE, SeasonGroup, Unit, track_fingerprint
from pipeline.library.source import LibraryItem
from pipeline.library.state import StageStates
from pipeline.publish.catalogue import catalogue_paths
from pipeline.publish.git import RepoState, RepoTarget, repo_state
from pipeline.publish.report import ReportSpec
from pipeline.review import QueueEntry, read_entry

_ANALYSIS_FIELDS = ('target_fs', 'resolution', 'avg_window', 'peak_window')


def analysis_from_values(values: Mapping[str, Any]) -> AnalysisConfig:
    ''' An AnalysisConfig from a flat mapping of options, or one nested under `analysis:`; flat wins. '''
    configured = values.get('analysis') or {}
    fields = {name: values.get(name, configured.get(name)) for name in _ANALYSIS_FIELDS}
    return AnalysisConfig(**{name: value for name, value in fields.items() if value is not None})


def report_spec_from_values(values: Mapping[str, Any]) -> Optional[ReportSpec]:
    '''
    The report image's layout from a `report_spec:` mapping of `ReportSpec` fields (in `run:`/`sync:`), or None -- the
    default -- if there is none. Publish and discovery must be given the same one: it is in the published digest.
    :raises ValueError: for a field ReportSpec does not have or a value of the wrong kind.
    '''
    given = values.get('report_spec')
    if not given:
        return None
    if not isinstance(given, Mapping):
        raise ValueError(f'report_spec must be a mapping of report layout fields, got {given!r}')
    try:
        return ReportSpec(**given)
    except TypeError as error:
        raise ValueError(f'report_spec is not valid: {error}')


@dataclass(frozen=True)
class ScanSettings:
    '''
    The parts of a run and a publish that decide what state a title is in. They must match what `run`, `publish` and
    `commit` are given, or discovery describes work those commands would not do.
    '''
    work_dir: str
    queue_dir: str
    designer: str = ''
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    coverage: Coverage = 'complete_programme'
    keep_multichannel: bool = False
    tv_mode: str = DEFAULT_TV_MODE
    xml_repo: str = ''
    xml_dir: str = ''
    images_repo: str = ''
    image_dir: str = ''
    meta_defaults: Optional[dict] = None
    # what `publish` is given besides the above, because each is in the published digest (publish_digest()). Unset --
    # the default -- keeps a digest recorded before they were counted valid, so give them exactly as `publish` gets them.
    image_owner: str = ''
    image_repo_name: str = ''
    report_spec: Optional[ReportSpec] = None    # None: the default ReportSpec()

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> 'ScanSettings':
        ''' From the flat options the CLI merges from a config file's `run:` and `sync:` sections and its flags. '''
        def text(name: str) -> str:
            return str(values.get(name) or '')

        return cls(
            work_dir=text('work_dir'), queue_dir=text('queue_dir'), designer=text('designer'),
            config=analysis_from_values(values), coverage=values.get('coverage') or 'complete_programme',
            keep_multichannel=bool(values.get('keep_multichannel', False)),
            tv_mode=values.get('tv_mode') or DEFAULT_TV_MODE, xml_repo=text('xml_repo'), xml_dir=text('xml_dir'),
            images_repo=text('images_repo'), image_dir=text('image_dir'),
            meta_defaults=values.get('meta_defaults') or None, image_owner=text('image_owner'),
            image_repo_name=text('image_repo_name'), report_spec=report_spec_from_values(values))

    @classmethod
    def from_profile(cls, profile) -> 'ScanSettings':
        ''' From a Profile: `run:` wins over `sync:` (as it does for the directories), the profile's own paths win. '''
        sync, run = profile.config.get('sync') or {}, profile.config.get('run') or {}
        values = {**sync, **run}
        values.update({name: getattr(profile, name) for name in
                       ('work_dir', 'queue_dir', 'xml_repo', 'xml_dir', 'images_repo', 'image_dir')
                       if getattr(profile, name)})
        return cls.from_values(values)

    @property
    def has_image(self) -> bool:
        return bool(self.images_repo)


# --- fingerprints and failure memory -----------------------------------------------------------------------------

def safe_fingerprint(item: LibraryItem) -> str:
    ''' source_fingerprint(), or '' -- "unknown" -- for an item that has none of its own and cannot be stat'd. '''
    try:
        return source_fingerprint(item)
    except OSError:
        return ''


def unit_fingerprint(unit: Unit) -> str:
    ''' An item's fingerprint; a season's is one over its episodes' (and which episodes they are). '''
    if isinstance(unit, SeasonGroup):
        parts = [[m.episodes[0], safe_fingerprint(m)] for m in unit.members]
        return hashlib.sha256(json.dumps(parts).encode('utf-8')).hexdigest()[:32]
    return safe_fingerprint(unit)


SEASON_FINGERPRINT_PREFIX = 'season:'


def season_source_fingerprint(group: SeasonGroup) -> str:
    '''
    What a season's source is, cheaply, from its members' own fingerprints (an item listing's, or one `stat` each) and
    which episodes they are; the same computation at design time (recorded as the entry's `source_fingerprint`) and at
    scan time, so a re-ripped episode is seen as "source changed since accepted" without the season being extracted
    again. Prefixed, so it cannot be mistaken for the joined track's fingerprint an older entry recorded.
    :return: '' -- unknown -- if any member's fingerprint is unknown (an offline source): nothing is compared then.
    '''
    parts = [[m.episodes[0], safe_fingerprint(m)] for m in group.members]
    if any(not fingerprint for _, fingerprint in parts):
        return ''
    return SEASON_FINGERPRINT_PREFIX + hashlib.sha256(json.dumps(parts).encode('utf-8')).hexdigest()[:32]


def failure_key(stage: str, item: LibraryItem, *, config: AnalysisConfig, designer: str, coverage: str,
                keep_multichannel: bool) -> str:
    '''
    What a failure was against, other than the source itself: the settings the stage ran with. A failure is only
    remembered while both this and the source fingerprint are unchanged (design.md §12.5, "Failure memory").
    '''
    if stage == 'extract':
        payload = {'params': extract_params_hash(item, config, True), 'keep_multichannel': bool(keep_multichannel)}
    else:
        payload = {'designer': designer, 'config': asdict(config), 'coverage': coverage,
                   'audio_stream': item.audio_stream, 'playlist_name': item.playlist_name}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class FailureMemory:
    stage: str          # 'extract' | 'design'
    message: str
    fingerprint: str    # the source fingerprint it failed against
    key: str            # failure_key() -- the settings it failed against


def failure_applies(memory: FailureMemory, item: LibraryItem, fingerprint: str, *, config: AnalysisConfig,
                    designer: str, coverage: str, keep_multichannel: bool) -> bool:
    '''
    True while a remembered failure is still what would happen again: the source (its fingerprint) and the settings
    (failure_key()) are exactly those it failed with. The one rule for "do not retry this" -- discovery (which shows
    the title as *failed*) and `run` (which skips it unless asked to retry) both use it, so they cannot disagree.
    '''
    key = failure_key(memory.stage, item, config=config, designer=designer, coverage=coverage,
                      keep_multichannel=keep_multichannel)
    return memory.fingerprint == fingerprint and memory.key == key


# --- queue entries -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EntryFacts:
    '''
    The parts of a QueueEntry discovery needs -- everything but the curve and the candidates' filters -- plus the
    file's mtime and size, which is what tells the index its cached copy is still good. Quacks like a QueueEntry for
    design_status() (`status`, `design_fingerprint`).
    '''
    status: str
    chosen: Optional[int]
    confidence: Optional[float]
    candidates: int
    decline: str
    design_fingerprint: Optional[str]
    source_fingerprint: Optional[str]
    published_digest: Optional[str]
    art_path: Optional[str]
    meta: Dict[str, Any]
    mtime_ns: int
    size: int
    inode: int = 0      # with the change time, so an entry rewritten in place with a same-length value, within a
    ctime_ns: int = 0   # coarse filesystem's one mtime tick, is still noticed (write_queue_entry replaces the file)

    @classmethod
    def of(cls, entry: QueueEntry, mtime_ns: int, size: int, inode: int = 0, ctime_ns: int = 0) -> 'EntryFacts':
        return cls(entry.status, entry.chosen_candidate_index,
                   entry.candidates[0].confidence if entry.candidates else None, len(entry.candidates),
                   (entry.decline_message or entry.decline_reason or '') if not entry.candidates else '',
                   entry.design_fingerprint, entry.source_fingerprint, entry.published_digest, entry.art_path,
                   dict(entry.meta), mtime_ns, size, inode, ctime_ns)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Optional['EntryFacts']:
        '''
        :return: None if the text is not a summary this version wrote (stale, corrupt): a cache miss, so the entry is
            read again, and never an error that would stop a scan.
        '''
        try:
            return cls(**json.loads(text))
        except (TypeError, ValueError, KeyError):
            return None


def _stat_signature(path: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    ''' (mtime_ns, size, inode, ctime_ns): what says a file is the one seen last time. '''
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino, stat.st_ctime_ns


def read_entry_facts(queue_dir: str, entry_id: str, cached: Optional[EntryFacts] = None) -> Optional[EntryFacts]:
    '''
    :param cached: what was read last time; used as it is if the entry's file has the same mtime, size, inode and
        change time.
    :return: None if the title has no queue entry (or an unreadable one -- which is not this function's to report).
    '''
    signature = _stat_signature(os.path.join(queue_dir, f'{entry_id}.json')) if queue_dir else None
    if signature is None:
        return None
    if cached is not None and (cached.mtime_ns, cached.size, cached.inode, cached.ctime_ns) == signature:
        return cached
    try:
        return EntryFacts.of(read_entry(queue_dir, entry_id), *signature)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def metadata_problems(meta: Mapping[str, Any], meta_defaults: Optional[Mapping[str, Any]] = None) -> Tuple[str, ...]:
    ''' pipeline.metadata.validate() for an entry's metadata as it would be published; empty if it is complete. '''
    from pipeline.metadata import BeqMetadata, validate
    try:
        return tuple(validate(BeqMetadata(**{'title': '', 'year': '', **(meta_defaults or {}), **meta})))
    except (TypeError, AttributeError, ValueError) as error:  # a null title, an unknown field: for a person to fix
        return (f'metadata is not valid: {error}',)


# --- the evaluator -----------------------------------------------------------------------------------------------

@dataclass
class Evaluation:
    ''' Everything discovery learned about one title, ready to become a row. '''
    states: StageStates
    fingerprint: str
    facts: Optional[EntryFacts] = None
    title: str = ''
    year: str = ''
    in_catalogue: bool = False
    digest_key: str = ''            # what the cached digest below was computed from
    current_digest: str = ''
    clear_failure: bool = False     # the remembered failure no longer applies (the source or settings changed)


_PUBLISHED = ('accepted', 'published')
_COMMIT_RANK = {'uncommitted': 3, 'committed': 2, 'unknown': 1, 'pushed': 0}


def _rel(path: str) -> str:
    return path.replace(os.sep, '/')  # git reports paths with forward slashes


class Evaluator:
    '''
    One per scan. It holds what is the same for every title -- the directory listings, the repos' states -- so that
    each title costs its own manifest and (if its file changed) its own entry, and nothing else.
    '''

    def __init__(self, settings: ScanSettings, *, xml_index: Optional[Mapping[Tuple[str, bool], Set[str]]] = None,
                 own_ids: FrozenSet[str] = frozenset()):
        '''
        :param xml_index: (TMDB id, is tv) -> the XML file stems that have it, from the local XML repo.
        :param own_ids: ids this profile publishes under; an XML with one of those names is ours, not someone else's.
        '''
        self.settings = settings
        self.__dirs: FrozenSet[str] = frozenset(
            name for name in os.listdir(settings.work_dir) if not name.startswith('.')
        ) if settings.work_dir and os.path.isdir(settings.work_dir) else frozenset()
        self.__xml_index = xml_index or {}
        self.__own_ids = own_ids
        self.__repo_states: Dict[str, RepoState] = {}

    # extract ------------------------------------------------------------------------------------------------------

    def _extract_item(self, item: LibraryItem, fingerprint: str, mono_only: bool = False) -> Tuple[str, dict]:
        ''' :return: (extract state, the manifest read) '''
        if item.id not in self.__dirs:
            return 'none', {}
        item_dir = os.path.join(self.settings.work_dir, item.id)
        manifest = read_manifest(item_dir)
        config = self.settings.config
        mono = extract_status(item, item_dir, config, True, fingerprint=fingerprint, manifest=manifest)
        states = [mono.state]
        if self.settings.keep_multichannel and not mono_only and manifest.get('source_channel_count') != 1:
            states.append(extract_status(item, item_dir, config, False, fingerprint=fingerprint,
                                         manifest=manifest).state)
        if mono.state == 'none':
            return 'none', manifest
        return ('current' if all(s == 'current' for s in states) else 'stale'), manifest

    def _extract_season(self, group: SeasonGroup) -> Tuple[str, Optional[str], List[str]]:
        '''
        A season is as extracted as its episodes. An episode that was never extracted is tolerated once some were:
        a run leaves out an episode that will not extract and designs the rest, and would otherwise be reported as
        needing extraction for ever.
        :return: (state, the joined track's fingerprint if every extracted episode is current, else None, the
            extracted episodes' ids)
        '''
        states, wavs, extracted = [], [], []
        for member in group.members:
            state, _ = self._extract_item(member, safe_fingerprint(member), mono_only=True)  # a season keeps no multichannel
            states.append(state)
            if state != 'none':
                extracted.append(member.id)
            if state == 'current':
                wavs.append((member.episodes[0], os.path.join(self.settings.work_dir, member.id, 'mono.wav')))
        if not extracted:
            return 'none', None, []
        if 'stale' in states:
            return 'stale', None, extracted
        return 'current', track_fingerprint(sorted(wavs)), extracted

    # publish and commit ---------------------------------------------------------------------------------------------

    def _repo_state(self, path: str) -> RepoState:
        if path not in self.__repo_states:
            self.__repo_states[path] = repo_state(RepoTarget(path))
        return self.__repo_states[path]

    def _commit_state(self, entry_id: str) -> Tuple[str, str]:
        '''
        :return: (commit state, why, if it is not obvious) of a published title, worst of its XML and image. With no XML
            repository configured there is nothing to commit to: `none`, which `derive_needs` does not ask to commit.
        '''
        settings = self.settings
        if not settings.xml_repo:
            return 'none', ''   # nothing to commit to: not applicable, so a published title is not "commit" for ever
        xml_path, image_path = catalogue_paths(entry_id, settings.xml_dir, settings.image_dir)
        worst, detail = 'pushed', ''
        for repo, path in ((settings.xml_repo, xml_path), (settings.images_repo, image_path)):
            if not repo:
                continue
            state = self._repo_state(repo)
            path = _rel(path)
            if state.uncommitted is None:
                found, why = 'unknown', 'cannot read git in ' + repo
            elif path in state.uncommitted:
                found, why = 'uncommitted', ''
            elif state.unpushed is None:
                found, why = 'unknown', 'cannot tell whether it is pushed (the branch has no upstream and was never pushed)'
            else:
                found, why = ('committed' if path in state.unpushed else 'pushed'), ''
            if _COMMIT_RANK[found] > _COMMIT_RANK[worst]:
                worst, detail = found, why
        return worst, detail

    def _digest(self, entry_id: str, facts: EntryFacts, previous_key: str, previous_digest: str,
                previous_conflict: bool) -> Tuple[str, str, bool]:
        ''' :return: (digest key, current digest, project conflict), from the cache if nothing it depends on changed. '''
        from pipeline.publish.project import ProjectFilterConflict
        from pipeline.review import current_publish_digest, project_paths
        settings = self.settings
        mono = mc = mc_wav = None
        if settings.work_dir:
            _, mono, mc, mc_wav = project_paths(settings.work_dir, entry_id)
        key = hashlib.sha256(json.dumps([
            facts.mtime_ns, facts.size, facts.inode, facts.ctime_ns, _stat_signature(facts.art_path),
            _stat_signature(mono), _stat_signature(mc), bool(mc_wav and os.path.isfile(mc_wav)),
            settings.meta_defaults, settings.has_image, bool(settings.work_dir), settings.image_owner,
            settings.image_repo_name, asdict(settings.report_spec) if settings.report_spec else None],
            sort_keys=True, default=str).encode('utf-8')).hexdigest()
        if previous_key == key:
            return key, previous_digest, previous_conflict
        try:
            entry = read_entry(settings.queue_dir, entry_id)
            return key, current_publish_digest(entry, meta_defaults=settings.meta_defaults,
                                               work_dir=settings.work_dir or None, has_image=settings.has_image,
                                               report_spec=settings.report_spec,
                                               image_owner=settings.image_owner or None,
                                               image_repo_name=settings.image_repo_name or None), False
        except ProjectFilterConflict:
            return key, '', True
        except Exception:  # a digest that cannot be worked out must not stop discovery: it just cannot compare
            return key, '', False

    def _publish_and_commit(self, entry_id: str, facts: EntryFacts, previous: Optional[Mapping]) -> Dict[str, Any]:
        ''' The publish and commit states of an entry that is accepted or published. '''
        out = {'publish': 'not_written', 'commit': 'none', 'out_of_date': '', 'commit_detail': '',
               'project_conflict': False, 'digest_key': '', 'current_digest': ''}
        if facts.status not in _PUBLISHED or facts.chosen is None:
            return {**out, 'publish': 'none'}
        key, digest, conflict = self._digest(
            entry_id, facts, (previous or {}).get('digest_key') or '', (previous or {}).get('current_digest') or '',
            bool((previous or {}).get('conflict')))
        out.update(digest_key=key, current_digest=digest, project_conflict=conflict)
        if facts.status != 'published':
            return out
        out['publish'] = 'written'
        settings = self.settings
        if settings.xml_repo and not os.path.isfile(
                os.path.join(settings.xml_repo, catalogue_paths(entry_id, settings.xml_dir, settings.image_dir)[0])):
            out.update(publish='out_of_date', out_of_date='its file is missing from the repository')
        elif facts.published_digest and digest and facts.published_digest != digest:
            out.update(publish='out_of_date', out_of_date='changed since it was published')
        if out['publish'] == 'written':
            out['commit'], out['commit_detail'] = self._commit_state(entry_id)
        return out

    # the title --------------------------------------------------------------------------------------------------

    def in_catalogue(self, item: Optional[LibraryItem], meta: Mapping[str, Any]) -> bool:
        '''
        True if the XML repo already holds this title's TMDB id under a file this profile did not publish. The XML has
        no movie/tv marker, so the match is on the id together with whether a season is set (design.md §12.14).
        '''
        tmdb = str(((item.external_ids.get('tmdb') if item else None) or meta.get('the_movie_db') or '')).strip()
        if not tmdb:
            return False
        is_tv = bool((item and item.kind == 'tv') or meta.get('season'))
        return any(stem not in self.__own_ids for stem in self.__xml_index.get((tmdb, is_tv), ()))

    def evaluate(self, unit: Unit, previous: Optional[Mapping] = None,
                 failure: Optional[FailureMemory] = None) -> Evaluation:
        '''
        :param previous: the title's row from the last scan, for its cached entry summary and digest.
        :param failure: the failure remembered for this title, if any.
        '''
        settings = self.settings
        season = isinstance(unit, SeasonGroup)
        item = unit.item if season else unit
        fingerprint = unit_fingerprint(unit)
        cached = EntryFacts.from_json(previous['entry_summary']) if previous and previous.get('entry_summary') else None
        # (a summary this version cannot read is a cache miss)
        facts = read_entry_facts(settings.queue_dir, item.id, cached)

        if season:
            extract, track, extracted = self._extract_season(unit)
            design_source = track
            changed_source = season_source_fingerprint(unit)   # from the members' own fingerprints: no need to extract
            multichannel = [False]
        else:
            extract, manifest = self._extract_item(item, fingerprint)
            design_source, changed_source = fingerprint, fingerprint
            count = manifest.get('source_channel_count')
            kept = settings.keep_multichannel and 'multichannel_source_fingerprint' in manifest and count != 1
            multichannel = [kept] + ([not kept] if count is None and kept else [])

        if design_source is None:  # a season with nothing (current) to join
            design = 'none' if facts is None else 'protected' if facts.status in _PUBLISHED else 'stale'
        else:
            statuses = [design_status(item, facts, settings.designer, settings.config, settings.coverage,
                                      multichannel=mc, source=design_source) for mc in multichannel]
            design = next((s for s in statuses if s.current), statuses[0]).state

        failed_message, clear = '', False
        if failure is not None:
            if failure_applies(failure, item, fingerprint, config=settings.config, designer=settings.designer,
                               coverage=settings.coverage, keep_multichannel=settings.keep_multichannel):
                failed_message = failure.message
                if failure.stage == 'extract':
                    extract = 'failed'
                else:
                    design = 'failed'
            else:
                clear = True

        publish = self._publish_and_commit(item.id, facts, previous) if facts is not None else {
            'publish': 'none', 'commit': 'none', 'out_of_date': '', 'commit_detail': '', 'project_conflict': False,
            'digest_key': '', 'current_digest': ''}
        meta = facts.meta if facts is not None else {}
        problems = metadata_problems(meta, settings.meta_defaults) \
            if facts is not None and facts.status in ('pending', 'accepted', 'published') else ()
        recorded = facts.source_fingerprint if facts is not None and facts.status in _PUBLISHED else None
        if season and recorded and not recorded.startswith(SEASON_FINGERPRINT_PREFIX):
            changed_source = design_source   # recorded before seasons had their own: the joined track's, if it is known
        source_changed = bool(recorded and changed_source and changed_source != recorded)
        states = StageStates(
            extract=extract, design=design, review='none' if facts is None else
            ('accepted' if facts.status == 'published' else facts.status), publish=publish['publish'],
            commit=publish['commit'], failure=failed_message, project_conflict=publish['project_conflict'],
            source_changed=source_changed, metadata_problems=problems, decline=facts.decline if facts else '',
            confidence=facts.confidence if facts else None, candidates=facts.candidates if facts else 0,
            out_of_date=publish['out_of_date'], commit_detail=publish['commit_detail'])
        return Evaluation(
            states, fingerprint, facts, title=str(meta.get('title') or item.title or item.display_name or ''),
            year=str(meta.get('year') or item.year or ''), in_catalogue=self.in_catalogue(item, meta),
            digest_key=publish['digest_key'], current_digest=publish['current_digest'], clear_failure=clear)

    def evaluate_from_outputs(self, entry_id: str, previous: Optional[Mapping] = None) -> Optional[Evaluation]:
        '''
        A title's states from its outputs alone, for rebuilding a lost index without listing any source. With no item
        to compare against, the source is assumed unchanged: an extraction is current if it is recorded and its wav is
        there, a design if the entry carries a fingerprint. The next scan puts right anything that is not so.
        :return: None if the title has no queue entry (a rebuild has nothing to say about it).
        '''
        facts = read_entry_facts(self.settings.queue_dir, entry_id)
        if facts is None:
            return None
        item_dir = os.path.join(self.settings.work_dir, entry_id)
        manifest = read_manifest(item_dir) if entry_id in self.__dirs else {}
        extract = 'current' if (manifest.get('mono_source_fingerprint') is not None
                                and os.path.isfile(os.path.join(item_dir, 'mono.wav'))) else 'none'
        design = 'protected' if facts.status in _PUBLISHED else 'current' if facts.design_fingerprint else 'stale'
        publish = self._publish_and_commit(entry_id, facts, previous)
        meta = facts.meta
        problems = metadata_problems(meta, self.settings.meta_defaults) \
            if facts.status in ('pending', 'accepted', 'published') else ()
        states = StageStates(
            extract=extract, design=design, review='accepted' if facts.status == 'published' else facts.status,
            publish=publish['publish'], commit=publish['commit'], project_conflict=publish['project_conflict'],
            metadata_problems=problems, decline=facts.decline, confidence=facts.confidence,
            candidates=facts.candidates, out_of_date=publish['out_of_date'], commit_detail=publish['commit_detail'])
        return Evaluation(states, str(manifest.get('mono_source_fingerprint') or ''), facts,
                          title=str(meta.get('title') or entry_id), year=str(meta.get('year') or ''),
                          in_catalogue=self.in_catalogue(None, meta), digest_key=publish['digest_key'],
                          current_digest=publish['current_digest'])
