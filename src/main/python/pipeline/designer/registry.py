'''
A small in-process registry binding a designer name to a callable matching
the design/designer-interface.md contract -- D8 in design/api-headless-
pipeline.md: start with a single in-process Python callable, registered by
name, and defer a subprocess/HTTP binding until something actually needs it.
'''
from typing import Callable, Dict, Set

from pipeline.designer.contract import DesignRequest, DesignResponse

DesignerCallable = Callable[[DesignRequest], DesignResponse]

_registry: Dict[str, DesignerCallable] = {}
_takes_sources: Set[str] = set()


def register_designer(name: str, designer: DesignerCallable, takes_sources: bool = False) -> None:
    '''
    :param name: a unique name for the designer, e.g. 'beqanalyser.rolloff_v1'.
    :param designer: a callable matching def design(request: DesignRequest) -> DesignResponse.
    :param takes_sources: the callable also takes `sources=` (pipeline.designer.sources.AudioSources) -- an HTTP designer
        that can be sent arrays by reference (designer-interface.md §7.1, 1.2). Every other designer is called with the
        request alone.
    '''
    _registry[name] = designer
    if takes_sources:
        _takes_sources.add(name)
    else:
        _takes_sources.discard(name)


def unregister_designer(name: str) -> None:
    _registry.pop(name, None)
    _takes_sources.discard(name)


def get_designer(name: str) -> DesignerCallable:
    '''
    :raises KeyError: if no designer is registered under that name.
    '''
    if name not in _registry:
        raise KeyError(f"No designer registered as '{name}' -- registered: {sorted(_registry.keys())}")
    return _registry[name]


def takes_sources(name: str) -> bool:
    ''' Whether the designer registered as `name` is called with `sources=` as well as the request. '''
    return name in _takes_sources


def registered_designers() -> list:
    return sorted(_registry.keys())
