'''
TV seasons treated as a single track -- design/library-sync-pipeline-plan.md §11.9.

With `tv_mode='season'` the TV items of one series and season become one unit: each episode is extracted as usual,
their (mono, decimated) audio is joined into one track, that track is designed once, and the result is published as
a filter for the whole season, with every episode that went into it named as in scope. `tv_mode='episode'` leaves
items alone, so each episode gets its own filter and its own metadata.
'''
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pipeline.library.source import LibraryItem

logger = logging.getLogger('library_season')

TV_MODES = ('episode', 'season')
DEFAULT_TV_MODE = 'episode'
_TRACK_MANIFEST = 'season_track.json'
_BLOCK_FRAMES = 1 << 20


@dataclass(frozen=True)
class SeasonGroup:
    '''
    One series' season as a single unit. `item` is the synthetic library item the design and review queue see;
    `members` are the real episode items it is made from, in episode order.
    '''
    item: LibraryItem
    members: Tuple[LibraryItem, ...]


Unit = Union[LibraryItem, SeasonGroup]


def _groupable(item: LibraryItem) -> bool:
    return item.kind == 'tv' and bool(item.title and item.season) and len(item.episodes) == 1


def season_item_id(title: str, season: str) -> str:
    '''
    A filesystem-safe, stable id for a series' season, e.g. `some-show-s01-1a2b3c`. It depends only on the series
    title and season number, so the same season keeps its cache and review entry from run to run and whichever
    episodes are present.
    '''
    slug = re.sub(r'[^a-z0-9]+', '-', title.casefold()).strip('-')[:40] or 'show'
    digest = hashlib.sha256(f"{title.casefold().strip()}|{int(season)}".encode('utf-8')).hexdigest()[:6]
    return f"{slug}-s{int(season):02d}-{digest}"


def _build_item(members: Sequence[LibraryItem]) -> LibraryItem:
    first = members[0]
    external_ids = {}
    for member in members:
        for identifier, value in member.external_ids.items():
            external_ids.setdefault(identifier, value)
    return LibraryItem(
        id=season_item_id(first.title, first.season),
        source_path=first.source_path,
        display_name=f"{first.title} Season {int(first.season)}",
        title=first.title,
        year=next((m.year for m in members if m.year), None),
        kind='tv',
        external_ids=external_ids,
        art_path=next((m.art_path for m in members if m.art_path), None),
        season=first.season,
        episodes=tuple(m.episodes[0] for m in members),
    )


def plan_units(items: Sequence[LibraryItem], tv_mode: str = DEFAULT_TV_MODE) -> List[Unit]:
    '''
    :return: what to process, in the order the items came. In 'season' mode the episodes of one series and season
        are replaced by a single SeasonGroup, placed where its first episode was; items that are not a numbered
        episode of a titled series pass through untouched. Two items for the same episode (say, two copies) count
        once -- the first wins -- since joining both would double it.
    :raises ValueError: for an unknown mode.
    '''
    if tv_mode not in TV_MODES:
        raise ValueError(f"tv_mode must be one of {', '.join(TV_MODES)}, got {tv_mode!r}")
    if tv_mode == 'episode':
        return list(items)
    order: List[Union[LibraryItem, Tuple[str, str]]] = []
    seasons: Dict[Tuple[str, str], Dict[int, LibraryItem]] = {}
    for item in items:
        if not _groupable(item):
            order.append(item)
            continue
        key = (item.title.casefold().strip(), str(int(item.season)))
        if key not in seasons:
            seasons[key] = {}
            order.append(key)
        episode = item.episodes[0]
        if episode in seasons[key]:
            logger.warning('%s is a second item for %s season %s episode %s; ignored', item.id, item.title,
                           item.season, episode)
            continue
        seasons[key][episode] = item
    units: List[Unit] = []
    for entry in order:
        if isinstance(entry, LibraryItem):
            units.append(entry)
        else:
            members = tuple(seasons[entry][episode] for episode in sorted(seasons[entry]))
            units.append(SeasonGroup(_build_item(members), members))
    return units


def _track_fingerprint(member_wavs: Sequence[Tuple[int, str]]) -> str:
    ''' Changes whenever an episode is added, dropped or re-extracted (its wav's stat moves). '''
    parts = []
    for episode, path in member_wavs:
        stat = os.stat(path)
        parts.append([episode, os.path.basename(os.path.dirname(path)), stat.st_mtime_ns, stat.st_size])
    return hashlib.sha256(json.dumps(parts).encode('utf-8')).hexdigest()


def _join(member_wavs: Sequence[Tuple[int, str]], out_path: str) -> None:
    import soundfile as sf
    writer = None
    try:
        for episode, path in member_wavs:
            with sf.SoundFile(path) as source:
                if writer is None:
                    writer = sf.SoundFile(out_path, mode='w', samplerate=source.samplerate,
                                          channels=source.channels, subtype=source.subtype, format='WAV')
                elif (source.samplerate, source.channels) != (writer.samplerate, writer.channels):
                    raise ValueError(f"episode {episode}'s audio ({source.samplerate} Hz, {source.channels} ch) "
                                     f"does not match the others ({writer.samplerate} Hz, {writer.channels} ch)")
                for block in source.blocks(blocksize=_BLOCK_FRAMES, dtype='float64'):
                    writer.write(block)
    finally:
        if writer is not None:
            writer.close()


def season_track_if_needed(member_wavs: Sequence[Tuple[int, str]], target_dir: str,
                           force: bool = False) -> Tuple[str, str, bool]:
    '''
    Joins the episodes' wavs, in episode order, into `<target_dir>/mono.wav` -- the file the design and the project
    files read, exactly as for a single episode. Skipped when the same episodes' wavs are unchanged.
    :param member_wavs: (episode number, wav path) per episode.
    :return: (track path, its fingerprint, True if it was already up to date).
    :raises ValueError: if the episodes' audio has different sample rates or channel counts, or none are given.
    '''
    if not member_wavs:
        raise ValueError('a season track needs at least one episode')
    ordered = sorted(member_wavs)
    fingerprint = _track_fingerprint(ordered)
    os.makedirs(target_dir, exist_ok=True)
    track = os.path.join(target_dir, 'mono.wav')
    manifest_path = os.path.join(target_dir, _TRACK_MANIFEST)
    if not force and os.path.isfile(track) and os.path.isfile(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            if json.load(f).get('fingerprint') == fingerprint:
                return track, fingerprint, True
    _join(ordered, track)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({'fingerprint': fingerprint, 'episodes': [e for e, _ in ordered]}, f)
    return track, fingerprint, False


def invalidate_season_track(target_dir: str) -> bool:
    ''' Forgets the joined track's fingerprint, so the next season_track_if_needed() joins the episodes again. '''
    manifest_path = os.path.join(target_dir, _TRACK_MANIFEST)
    if not os.path.isfile(manifest_path):
        return False
    os.remove(manifest_path)
    return True


def with_extracted(group: SeasonGroup, episodes: Sequence[int], fingerprint: str) -> LibraryItem:
    ''' The group's item narrowed to the episodes that actually made it into the track. '''
    return replace(group.item, episodes=tuple(episodes), fingerprint=fingerprint)
