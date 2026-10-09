'''
Which physical disk a title is read from, so extraction can read each disk one title at a time (`run.disks`).

On a pool of independent disks (JBOD: Unraid, mergerfs) each title lives on one disk, so titles on different disks
extract side by side at each disk's full speed, while two on the same disk make it seek between them and both slow
down. The pool's filesystem says where a file is in an extended attribute -- Unraid's `system.LOCATION` ("disk2"),
mergerfs's `user.mergerfs.basepath` -- so the profile names the attribute and the run reads it once per title:

    run:
      parallelism: {extract: 6}
      disks: {xattr: system.LOCATION, per_disk: 1}

A title whose disk cannot be read (no attribute, another filesystem, a platform without xattrs) is not held back by
disk, only by `parallelism.extract`. A title on several disks (a season whose episodes are spread, or Unraid's
comma-separated list) waits until every one of them has room.
'''
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Mapping, Optional

logger = logging.getLogger('library_disks')

DEFAULT_PER_DISK = 1


@dataclass(frozen=True)
class DiskLimit:
    xattr: str
    per_disk: int = DEFAULT_PER_DISK


def disk_limit(value: Any = None) -> Optional[DiskLimit]:
    ''' Validate the optional `run.disks` profile mapping. :return: None when absent: disks are not considered. '''
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('run.disks must be a mapping: {xattr: NAME, per_disk: N}')
    unknown = set(value) - {'xattr', 'per_disk'}
    if unknown:
        raise ValueError(f'unknown run.disks key(s): {", ".join(sorted(map(str, unknown)))}')
    name = value.get('xattr')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('run.disks.xattr must name the extended attribute that says which disk a file is on')
    per_disk = value.get('per_disk', DEFAULT_PER_DISK)
    if isinstance(per_disk, bool) or not isinstance(per_disk, int) or per_disk < 1:
        raise ValueError('run.disks.per_disk must be a whole number of titles, 1 or more')
    return DiskLimit(name.strip(), per_disk)


def disks_of(paths: Iterable[str], xattr: str) -> FrozenSet[str]:
    ''' The disks the files at `paths` are on, by the attribute `xattr`; empty if none of them says. '''
    found = set()
    for path in paths:
        try:
            value = os.getxattr(path, xattr)
        except (OSError, AttributeError) as error:   # AttributeError: no xattrs on this platform (Windows, macOS)
            logger.debug(f'No {xattr} on {path}: {error}')
            continue
        text = value.decode('utf-8', errors='replace').strip().strip('\x00')
        found.update(disk.strip() for disk in text.split(',') if disk.strip())
    return frozenset(found)


class DiskSlots:
    ''' How many extractions are reading each disk, and whether a title on some disks may start. '''

    def __init__(self, per_disk: int):
        self.__per_disk = per_disk
        self.__reading: Counter = Counter()

    def free(self, disks: FrozenSet[str]) -> bool:
        return all(self.__reading[disk] < self.__per_disk for disk in disks)

    def busy(self, disks: FrozenSet[str]) -> FrozenSet[str]:
        ''' Those of `disks` with no room. '''
        return frozenset(disk for disk in disks if self.__reading[disk] >= self.__per_disk)

    def take(self, disks: FrozenSet[str]) -> None:
        self.__reading.update(disks)

    def give_back(self, disks: FrozenSet[str]) -> None:
        self.__reading.subtract(disks)
