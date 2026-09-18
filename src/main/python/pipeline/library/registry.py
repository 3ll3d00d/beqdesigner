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
