'''
A small in-process registry binding a designer name to a callable matching
the design/designer-interface.md contract -- D8 in design/api-headless-
pipeline.md: start with a single in-process Python callable, registered by
name, and defer a subprocess/HTTP binding until something actually needs it.
'''
from typing import Callable, Dict

from pipeline.designer.contract import DesignRequest, DesignResponse

DesignerCallable = Callable[[DesignRequest], DesignResponse]

_registry: Dict[str, DesignerCallable] = {}


def register_designer(name: str, designer: DesignerCallable) -> None:
    '''
    :param name: a unique name for the designer, e.g. 'beqanalyser.rolloff_v1'.
    :param designer: a callable matching def design(request: DesignRequest) -> DesignResponse.
    '''
    _registry[name] = designer


def unregister_designer(name: str) -> None:
    _registry.pop(name, None)


def get_designer(name: str) -> DesignerCallable:
    '''
    :raises KeyError: if no designer is registered under that name.
    '''
    if name not in _registry:
        raise KeyError(f"No designer registered as '{name}' -- registered: {sorted(_registry.keys())}")
    return _registry[name]


def registered_designers() -> list:
    return sorted(_registry.keys())
