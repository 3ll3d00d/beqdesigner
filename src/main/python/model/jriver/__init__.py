from __future__ import annotations

from typing import Iterable

JRIVER_FS: int = 48000


def flatten(items: Iterable) -> Iterable:
    if isinstance(items, Iterable):
        for x in items:
            if isinstance(x, Iterable):
                yield from flatten(x)
            else:
                yield x
    else:
        yield items


class RoutingError(ValueError):
    pass


class ImpossibleRoutingError(RoutingError):

    def __init__(self, msg: str):
        super().__init__(msg)


class UnsupportedRoutingError(RoutingError):

    def __init__(self, msg: str):
        super().__init__(msg)
