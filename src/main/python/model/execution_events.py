"""Structured, Qt-free events for a library work-list execution."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Callable, Iterator, Optional, Tuple


@dataclass(frozen=True)
class ExecutionEvent:
    """One progress or process event, suitable for retaining and rendering later."""

    run_id: str
    title_id: str
    stage: str
    kind: str
    timestamp: float
    message: str = ''
    command: Tuple[str, ...] = ()
    stdout: str = ''
    stderr: str = ''
    exit_code: Optional[int] = None
    current: Optional[int] = None
    total: Optional[int] = None


@dataclass(frozen=True)
class _EventContext:
    run_id: str
    emit: Callable[[ExecutionEvent], None]
    title_id: str = ''
    stage: str = ''


_CURRENT: ContextVar[Optional[_EventContext]] = ContextVar('library_run_event_context', default=None)


@contextmanager
def execution_event_context(run_id: str, emit: Optional[Callable[[ExecutionEvent], None]], *,
                            title_id: str = '', stage: str = '') -> Iterator[None]:
    """Route events in this execution context to ``emit``; a missing callback is a no-op."""
    context = _EventContext(run_id, emit, title_id, stage) if emit is not None else None
    token = _CURRENT.set(context)
    try:
        yield
    finally:
        _CURRENT.reset(token)


@contextmanager
def event_scope(*, title_id: Optional[str] = None, stage: Optional[str] = None) -> Iterator[None]:
    """Temporarily attach title/stage identity to events emitted by lower-level process helpers."""
    context = _CURRENT.get()
    if context is None:
        yield
        return
    token = _CURRENT.set(_EventContext(context.run_id, context.emit,
                                       context.title_id if title_id is None else title_id,
                                       context.stage if stage is None else stage))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def emit_execution_event(kind: str, *, message: str = '', command=(), stdout: str = '', stderr: str = '',
                         exit_code: Optional[int] = None, current: Optional[int] = None,
                         total: Optional[int] = None) -> None:
    """Emit a structured event if a library run installed an observer in this context."""
    context = _CURRENT.get()
    if context is None:
        return
    context.emit(ExecutionEvent(
        context.run_id, context.title_id, context.stage, kind, time.time(), message,
        tuple(str(part) for part in command), stdout, stderr, exit_code, current, total))
