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
    # Source-provided descriptions, in ffmpeg audio-stream order.  This is deliberately separate from ``meta``:
    # it is operational data used when a reviewer selects another stream, not BEQ metadata to publish.
    audio_stream_details: tuple = ()
    playlist_name: Optional[str] = None      # BD only, forwarded to Session.extract()
    art_path: Optional[str] = None           # a local poster/cover file the source already has, if any
                                              # (§3.1.3)
    art_candidates: tuple = ()               # where such a file might be, in order of preference, for a source
                                              # that must not touch the disk while listing (JRiver's):
                                              # artwork.resolve_art() checks them at design time
    meta: dict = field(default_factory=dict) # extra BeqMetadata ctor kwargs the source can supply directly
                                              # -- merged in on top of, and losing to nothing from, whatever
                                              # TMDB resolution produces
    fingerprint: str = ''                    # source-specific change marker (§4.1) -- opaque to callers
    season: Optional[str] = None             # TV: the season number, when the source knows it
    episodes: tuple = ()                     # TV: the episode numbers (ints) this item covers -- one for an
                                              # episode, several for a season grouped into a single track
                                              # (design/library-sync-pipeline-plan.md §11.9)
    source_path_problem: Optional[str] = None  # non-sensitive source-path diagnostic, if a source knows it cannot
                                                # be opened on this host (e.g. an unmapped JRiver Windows path)


class LibrarySource(Protocol):
    def list_items(self, **query) -> Iterable[LibraryItem]:
        ''' `query`'s accepted keyword arguments are entirely source-specific -- there is no fixed
        cross-source query schema. A JRiver source, for example, can expose a preconfigured browse node
        and accept no per-call query at all, while a future source might accept its own filter syntax. '''
        ...
