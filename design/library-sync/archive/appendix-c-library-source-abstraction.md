# Library sync plan -- archived handoff spec

> Part of the library sync plan -- **start at the index**: [`../../library-sync-pipeline-plan.md`](../../library-sync-pipeline-plan.md).
> Contains Appendix C (chunk 4). Section numbers are global across the plan; the index maps every `§` to its file.
> Status of what this file describes: **built and shipped** -- kept as the record of the original spec; read only if you need the detail

## Appendix C -- Chunk 4 handoff spec: `LibrarySource` abstraction

Self-contained enough to implement without reading the rest of this
document (§3 above has the same shapes plus the JRiver-specific
rationale, but nothing in this chunk depends on JRiver at all). This is
the smallest chunk in the plan: a pure interface + an in-process
registry, no implementation behind it yet -- chunk 8 (JRiver, pending
chunk 3's real-server mapping fixture) is the first real implementation, but chunks 5-7
(extract cache, design cache, orchestration) only need this interface
to exist, not anything concrete implementing it, so this unblocks all
of them.

### C.1 Why

`run_library()` (chunk 7) needs to iterate "whatever a library gives
us" without knowing whether that's JRiver, Kodi, Plex, or a test
double. This chunk defines that contract -- a `LibraryItem` (the data
one title's row needs to carry through extract/design/publish) and a
`LibrarySource` protocol (`list_items(**query) -> Iterable[LibraryItem]`)
-- plus a registry to bind a configured instance to a name, the exact
same shape `pipeline.designer.registry` already uses for designers (a
GUI registers configured instances at startup, a CLI registers from a
config file at process start -- see `pipeline/README.md`'s "In the GUI,
Preferences -> Designers... registered under a `http:`-prefixed name on
every app startup").

### C.2 Scope

**In scope:**
- New package `pipeline/library/` (`__init__.py`, empty -- matches
  `pipeline/designer/__init__.py`'s convention).
- `pipeline/library/source.py` -- `LibraryItem`, `LibrarySource`.
- `pipeline/library/registry.py` -- `register_source`/`unregister_source`/
  `get_source`/`registered_sources`.

**Explicitly out of scope:**
- Any real `LibrarySource` implementation (JRiver -- chunk 8, blocked;
  Kodi/Plex -- not built at all, per §3.2).
- Wiring this into `run_library()`/`design_if_needed()`/anything else
  -- those are chunks 5-7's job; this chunk only needs to exist for
  them to import against, and this chunk's own tests use a small
  fake/test-only `LibrarySource` to exercise the shape, not a real one.

### C.3 Exact code

`pipeline/library/source.py`:

```python
'''
LibrarySource: the pluggable contract a library-scale extract+design run (pipeline/library/run.py, not yet
built) iterates over. One implementation ships against JRiver (blocked on a wire-format spike); Kodi/Plex
are named future implementations of this same interface, not built here (design/library-sync-pipeline-plan.md
§3.2).
'''
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol


@dataclass(frozen=True)
class LibraryItem:
    '''
    One title as a library source sees it -- enough for extract+design+metadata resolution to run against
    without needing to know which concrete source produced it.
    '''
    id: str                                  # stable across runs -- also the QueueEntry.id / cache key.
                                              # Must survive a rename/re-scan; a source should prefer its
                                              # own persistent key (e.g. JRiver's Media ID) over a
                                              # filename-derived one.
    source_path: str                         # ffmpeg input: a container file path, or a BDMV root directory
    display_name: str
    title: Optional[str] = None
    year: Optional[str] = None
    kind: str = 'movie'                      # 'movie' or 'tv' -- forwarded to TMDB resolution
    external_ids: dict = field(default_factory=dict)  # {'tmdb': '603', 'imdb': 'tt0133093'} -- whatever
                                              # identifiers the source can supply; empty if it can't
                                              # (design/library-sync-pipeline-plan.md §3.1.1)
    audio_stream: int = 0
    playlist_name: Optional[str] = None      # BD only, forwarded to Session.extract()
    art_path: Optional[str] = None           # a local poster/cover file the source already has, if any
                                              # (§3.1.3)
    meta: dict = field(default_factory=dict) # extra BeqMetadata ctor kwargs the source can supply directly
                                              # -- merged in on top of, and losing to nothing from, whatever
                                              # TMDB resolution produces
    fingerprint: str = ''                    # source-specific change marker (§4.1) -- opaque to callers


class LibrarySource(Protocol):
    def list_items(self, **query) -> Iterable[LibraryItem]:
        ''' `query`'s accepted keyword arguments are entirely source-specific (e.g. a JRiver search-string
        query) -- there is no fixed cross-source query schema, matching how model/batch.py's existing
        FileSearch takes source-specific glob patterns today. '''
        ...
```

`pipeline/library/registry.py`:

```python
'''
A small in-process registry binding a library source name to an already-configured instance --
pipeline.designer.registry's exact pattern (design/api-headless-pipeline.md D8): the cheapest binding
first, no HTTP binding needed unless something actually requires one.
'''
from typing import Dict, List

from pipeline.library.source import LibrarySource

_registry: Dict[str, LibrarySource] = {}


def register_source(name: str, source: LibrarySource) -> None:
    '''
    :param name: a unique name for the source, e.g. 'jriver'.
    :param source: an already-configured LibrarySource instance -- any per-connection configuration (a
        JRiver server's ip/auth, say) is resolved by the caller *before* registering, the same way
        pipeline.designer.http_binding.http_designer(url) is called once to produce a ready callable
        before pipeline.designer.registry.register_designer() ever sees it.
    '''
    _registry[name] = source


def unregister_source(name: str) -> None:
    _registry.pop(name, None)


def get_source(name: str) -> LibrarySource:
    '''
    :raises KeyError: if no source is registered under that name.
    '''
    if name not in _registry:
        raise KeyError(f"No library source registered as '{name}' -- registered: {sorted(_registry.keys())}")
    return _registry[name]


def registered_sources() -> List[str]:
    return sorted(_registry.keys())
```

### C.4 Tests to add

New `src/test/python/test_pipeline_library_source.py` -- nothing else
in the codebase exercises this yet (chunks 5-7 will), so this chunk's
own tests are what prove the shape works, using a small fake source
defined in the test file itself:

```python
class _FakeSource:
    def __init__(self, items):
        self._items = items
    def list_items(self, **query):
        return [i for i in self._items if not query.get('title_contains') or query['title_contains'] in i.title]
```

- `test_library_item_defaults` -- construct a `LibraryItem` with only
  the three required fields, assert every optional field's default
  (`title=None`, `kind='movie'`, `external_ids=={}`, `meta=={}`, etc.).
- `test_register_and_get_source_round_trips` -- `register_source('fake',
  _FakeSource([...]))`, `get_source('fake')` returns the same instance,
  `list_items()` on it works.
- `test_get_source_unregistered_raises_keyerror` -- includes the
  registered-names list in the message (matching
  `get_designer()`'s exact error-message shape).
- `test_unregister_source_removes_it` -- register, unregister, assert
  `get_source()` now raises.
- `test_registered_sources_is_sorted` -- register a few out of
  alphabetical order, assert `registered_sources()` comes back sorted.
- `test_list_items_accepts_source_specific_query_kwargs` -- via
  `_FakeSource`, confirms `list_items(**query)`'s duck-typed,
  no-fixed-schema shape works as intended (e.g.
  `list_items(title_contains='Player')` filters as expected).
- A `pipeline_qt_boundary`-style AST-scan test confirming
  `pipeline/library/source.py`/`registry.py` import no `qtpy` --
  matching the per-module convention every other new `pipeline/`
  module in this plan has added one of (see chunk 2's
  `test_pipeline_publish_project_module_has_no_qtpy_import`).

### C.5 Acceptance checklist

- [x] `pipeline/library/__init__.py`, `source.py`, `registry.py` created.
- [x] All of C.4's tests pass; full suite still green.
- [x] No `qtpy` import in either new module.
