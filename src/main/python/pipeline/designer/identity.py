'''
Which answer a designer gives (design/design-version.md, contract 1.4): its design identity, the pair
(`design_version`, `design_settings`), and whether a design made with one is out of date against what the designers
that are up now offer.

A designer reports its identity on each response and on `/health`. A queue entry made before 1.4 has only the build its
designer named; `backfill_version()` maps the builds this library has seen to the versions beqforge declared for its past
releases. A designer that reports no version leaves every entry's state unknown, and nothing else changes.
'''
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple

# beqforge's past releases, as beqforge declared them (its plans/design-version.md, checked against its tags and source
# digests 2026-10-19): the tokens a build string carries (the container's `src:` digest, a frozen build's revision and
# `src:` digest) and the design version that release would have reported. 0.2.4 and 0.2.5 changed no answer.
BEQFORGE_RELEASES: Tuple[Tuple[str, Tuple[str, ...], int], ...] = (
    ('0.2.0', ('src:eeb06d1cb9a6', '9f07a6700b22'), 1),
    ('0.2.1', ('src:c1ae12bafc27', '7a1702fa94de'), 2),
    ('0.2.2', ('src:745e6c61342a', 'fd9ae22846a6'), 3),
    ('0.2.3', ('src:3f0cbd9447c7', '2f1a0ccefd69', 'src:35c4437ecbb6'), 4),
    ('0.2.4', ('src:d81a799f2b7b', 'a110332e314d', 'src:04ac01018d49'), 4),
    ('0.2.5', ('src:76b2edd7e060', 'e1a2c24b1172', 'src:008f1c30d377'), 4),
)


def backfill_version(build: Optional[str]) -> Optional[int]:
    ''' The design version a build named before 1.4 would have reported; None for a build not in the table. '''
    if not build:
        return None
    for _, tokens, version in BEQFORGE_RELEASES:
        if any(token in build for token in tokens):
            return version
    return None


def valid_version(value: Any) -> Optional[int]:
    ''' A design version as the contract allows it (an int, at least 0, not a bool), else None. '''
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def valid_text(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class DesignIdentity:
    '''
    What a design was made with.
    :param version: its design version; None: unknown (an older designer, or a build not in the backfill table).
    :param settings: its `design_settings`; None: the designer said none.
    :param build: the build it named, for people.
    :param backfilled: the version came from the build (a design made before 1.4), so `settings` was never said.
    '''
    version: Optional[int]
    settings: Optional[str] = None
    build: Optional[str] = None
    backfilled: bool = False

    @classmethod
    def of(cls, version: Any = None, settings: Any = None, build: Any = None) -> 'DesignIdentity':
        ''' From what an entry holds: its own version if it has one, else the build's backfilled one. '''
        version, settings, build = valid_version(version), valid_text(settings), valid_text(build)
        if version is not None:
            return cls(version, settings, build)
        backfilled = backfill_version(build)
        return cls(backfilled, None, build, backfilled is not None)

    def describe(self) -> str:
        if self.version is None:
            return 'design version unknown'
        said = f'design version {self.version}'
        return said + (' (from its build)' if self.backfilled else '')


@dataclass(frozen=True)
class EndpointIdentity:
    '''
    What one endpoint's `/health` says it designs with. `settings_detail`: its non-default settings, if it says.
    `release`: the release it is (e.g. 0.2.10), for people; None when neither it nor its build says.
    '''
    url: str
    version: int
    settings: Optional[str] = None
    settings_detail: Optional[Mapping[str, Any]] = None
    build: Optional[str] = None
    release: Optional[str] = None


# a build that starts with a release (`0.2.10+src:…`, `v0.2.10 (frozen build)`), not a revision
_RELEASE_BUILD = re.compile(r'^v?(\d+\.\d+\.\d+)(?=$|[+ ])')


def release_of(release: Any, build: Optional[str]) -> Optional[str]:
    ''' The release a `/health` names: its `designer_release`, else the one its build starts with; None when neither. '''
    if valid_text(release):
        return release.strip()
    match = _RELEASE_BUILD.match(build or '')
    return match.group(1) if match else None


def identity_from_health(url: str, body: Any) -> Optional[EndpointIdentity]:
    ''' The identity a `/health` answer reports, or None when it reports no (valid) design version. '''
    if not isinstance(body, Mapping):
        return None
    version = valid_version(body.get('design_version'))
    if version is None:
        return None
    detail = body.get('design_settings_detail')
    build = valid_text(body.get('designer_build'))
    return EndpointIdentity(url, version, valid_text(body.get('design_settings')),
                            dict(detail) if isinstance(detail, Mapping) else None, build,
                            release_of(body.get('designer_release'), build))


@dataclass(frozen=True)
class CurrentIdentity:
    '''
    What a redesign now would be made with: the identity of the endpoint that is up with the highest design version.
    :param disagreement: the endpoints that are up but would answer differently, in words; empty when they agree.
    '''
    version: int
    settings: Optional[str] = None
    settings_detail: Optional[Mapping[str, Any]] = None
    disagreement: Tuple[str, ...] = ()

    def describe(self) -> str:
        said = f'design version {self.version}'
        if self.settings_detail:
            said += ' with ' + ', '.join(f'{k}={v}' for k, v in sorted(self.settings_detail.items()))
        return said


def current_identity(endpoints: Iterable[Optional[EndpointIdentity]]) -> Optional[CurrentIdentity]:
    '''
    The current identity of the endpoints that are up, as their `/health` reported it (None for one that reported no
    version); None when none of them reports one.
    '''
    reported = [e for e in endpoints if e is not None]
    if not reported:
        return None
    best = max(reported, key=lambda e: e.version)
    differ = [e for e in reported if (e.version, e.settings) != (best.version, best.settings)]
    disagreement = tuple(_says(e) for e in [*differ, best]) if differ else ()
    return CurrentIdentity(best.version, best.settings, best.settings_detail, disagreement)


def _says(endpoint: EndpointIdentity) -> str:
    ''' What an endpoint designs with, in words: its non-default settings if it lists them, else their digest. '''
    said = f'{endpoint.url} designs with version {endpoint.version}'
    if endpoint.settings_detail is not None:
        return said + (f', settings {dict(endpoint.settings_detail)}' if endpoint.settings_detail else
                       ', default settings')
    return said + (f', settings {endpoint.settings}' if endpoint.settings else '')


def is_out_of_date(identity: DesignIdentity, current: CurrentIdentity) -> bool:
    '''
    True if a design made with `identity` could come out differently if made now with `current`: its version is unknown or
    lower, or its settings differ. A design from before 1.4 never said its settings: it is current only when the current
    designer runs with every setting at its default (it says so with an empty `design_settings_detail`).
    '''
    if identity.version is None or identity.version < current.version:
        return True
    if identity.version > current.version:
        return True   # made by a newer designer than any up now: a redesign would not reproduce it
    if identity.backfilled:
        return current.settings_detail is None or bool(current.settings_detail)
    return identity.settings != current.settings


def out_of_date_reasons(identity: DesignIdentity, current: CurrentIdentity) -> List[str]:
    ''' Why it is out of date, in words; empty when it is not. '''
    if not is_out_of_date(identity, current):
        return []
    if identity.version is None:
        return ['designed by a designer that did not say its design version']
    if identity.version != current.version:
        return [f'designed with version {identity.version}; the designer now designs with {current.version}']
    return ['designed with different designer settings']
