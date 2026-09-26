"""Structured, Qt-free events for a library work-list execution."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import re
import shlex
import time
from typing import Callable, Iterator, Optional, Tuple

MAX_EVENT_TEXT = 8192
_SECRET = re.compile(r'(?i)(api[_-]?key|access[_-]?token|password|secret)([=: ]+)([^\s&]+)')
_URL_CREDENTIALS = re.compile(r'(https?://)[^/@\s]+:[^/@\s]+@', re.IGNORECASE)
_AUTH_HEADER = re.compile(r'(?i)(authorization\s*[:=]\s*(?:bearer|basic)\s+)[^\s,;]+')
_COOKIE_HEADER = re.compile(r'(?i)((?:set-)?cookie\s*[:=]\s*)[^\r\n]+')
_SECRET_ARG = re.compile(r'^--?(?:api[_-]?key|access[_-]?token|password|secret)$', re.IGNORECASE)
MAX_COMMAND_ARGS = 256
MAX_COMMAND_CHARS = 32 * 1024


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


# --- redaction: what is kept and shown of an event never carries a credential --------------------------------------

def redact_text(text: str) -> str:
    ''' `text` with credentials in URLs, authorization and cookie headers, and api keys, tokens, passwords and secrets masked. '''
    text = _URL_CREDENTIALS.sub(r'\1[REDACTED]@', text)
    text = _AUTH_HEADER.sub(r'\1[REDACTED]', text)
    text = _COOKIE_HEADER.sub(r'\1[REDACTED]', text)
    return _SECRET.sub(r'\1\2[REDACTED]', text)


def redact_argv(argv) -> tuple:
    ''' A command line, redacted, trimmed to a bounded size; the value after a secret-looking option is masked whole. '''
    safe, redact_next = [], False
    remaining = MAX_COMMAND_CHARS
    for raw in list(argv)[:MAX_COMMAND_ARGS]:
        arg = str(raw)
        if redact_next:
            safe.append('[REDACTED]')
            redact_next = False
            continue
        arg = redact_text(arg)[:min(MAX_EVENT_TEXT, remaining)]
        safe.append(arg)
        remaining -= len(arg)
        if remaining <= 0:
            safe[-1] += '…'
            break
        if _SECRET_ARG.match(arg):
            redact_next = True
    if len(argv) > MAX_COMMAND_ARGS:
        safe.append('[additional arguments trimmed]')
    return tuple(safe)


def event_text(event: ExecutionEvent) -> str:
    ''' The event as the lines a person reads (the work list's Details, the pipeline service's job log). '''
    stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
    heading = ' '.join(part for part in (stamp, event.stage, event.kind) if part)
    lines = [f'[{heading}] {event.message}' if event.message else f'[{heading}]']
    if event.command:
        lines.append('Command: ' + shlex.join(redact_text(arg) for arg in event.command))
    if event.exit_code is not None:
        lines.append(f'Exit code: {event.exit_code}')
    if event.stdout:
        lines.append('stdout:\n' + redact_text(event.stdout))
    if event.stderr:
        lines.append('stderr:\n' + redact_text(event.stderr))
    if event.current is not None or event.total is not None:
        lines.append(f'Progress: {event.current or 0} / {event.total or 0}')
    return '\n'.join(lines)


def redacted_event(event: ExecutionEvent) -> ExecutionEvent:
    ''' The event with its message, command and output redacted and each trimmed to MAX_EVENT_TEXT, fit to keep and show. '''
    if not (event.stdout or event.stderr or event.message or event.command):
        return event
    return ExecutionEvent(event.run_id, event.title_id, event.stage, event.kind, event.timestamp,
                          redact_text(event.message)[:MAX_EVENT_TEXT],
                          tuple(arg[:MAX_EVENT_TEXT] for arg in redact_argv(event.command)),
                          redact_text(event.stdout)[:MAX_EVENT_TEXT], redact_text(event.stderr)[:MAX_EVENT_TEXT],
                          event.exit_code, event.current, event.total)
