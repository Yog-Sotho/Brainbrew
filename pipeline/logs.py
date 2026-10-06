"""
Logging, configured once for the web app, the CLI and job threads.

structlog and the standard library share one handler, so library logs
(openai, httpx, datasets) and Brainbrew's own events come out in the same
format: human-readable on a terminal, one JSON object per line in containers
or when BRAINBREW_LOG_FORMAT=json. BRAINBREW_LOG_LEVEL sets the level.

`run_log(run)` additionally copies every log line emitted while a run is
executing (in that thread and the asyncio tasks it starts) into
runs/<id>/run.log as JSON lines.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import structlog

LEVEL_ENV = "BRAINBREW_LOG_LEVEL"
FORMAT_ENV = "BRAINBREW_LOG_FORMAT"  # json | console | auto (default)
_HANDLER_NAME = "brainbrew"
_NOISY = ("httpx", "httpx2", "httpcore", "openai", "urllib3", "filelock", "fsspec", "datasets")
_lock = threading.Lock()
_configured = False

_SHARED: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]


def _in_container() -> bool:
    return Path("/.dockerenv").exists() or bool(os.getenv("KUBERNETES_SERVICE_HOST"))


def use_json(fmt: str | None = None) -> bool:
    fmt = (fmt or os.getenv(FORMAT_ENV) or "auto").strip().lower()
    if fmt in {"json", "console"}:
        return fmt == "json"
    return _in_container() or not sys.stderr.isatty()


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever sys.stderr is *now*, so a replaced stderr is never stale."""

    @property  # type: ignore[override]
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


def _formatter(json_lines: bool) -> logging.Formatter:
    renderer: Any = (structlog.processors.JSONRenderer() if json_lines
                     else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            renderer,
        ],
    )


def configure_logging(level: str | None = None, fmt: str | None = None, *, force: bool = False) -> None:
    """Set up structlog + stdlib logging. Safe to call many times; only the first call (or force) acts."""
    global _configured
    with _lock:
        if _configured and not force:
            return
        structlog.configure(
            processors=[*_SHARED, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        handler = _StderrHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(_formatter(use_json(fmt)))
        root = logging.getLogger()
        for old in [h for h in root.handlers if h.get_name() == _HANDLER_NAME]:
            root.removeHandler(old)
        # The console shows the requested level; the root stays at INFO or lower so
        # per-run logs (run_log) always keep the run's INFO lines.
        console_level = logging.getLevelName((level or os.getenv(LEVEL_ENV) or "INFO").upper())
        handler.setLevel(console_level)
        root.addHandler(handler)
        root.setLevel(min(console_level, logging.INFO))
        for name in _NOISY:
            logging.getLogger(name).setLevel(logging.WARNING)
        _configured = True


class _RunFilter(logging.Filter):
    """Pass only records emitted while `run_id` is bound in the current context."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(structlog.contextvars.get_contextvars().get("run_id") == self.run_id)


@contextmanager
def run_log(log_path: Path, run_id: str) -> Iterator[None]:
    """Bind run_id for this thread's logs and copy them to *log_path* (JSON lines)."""
    configure_logging()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(_formatter(json_lines=True))
    handler.addFilter(_RunFilter(run_id))
    root = logging.getLogger()
    root.addHandler(handler)
    tokens = structlog.contextvars.bind_contextvars(run_id=run_id)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
        root.removeHandler(handler)
        handler.close()
