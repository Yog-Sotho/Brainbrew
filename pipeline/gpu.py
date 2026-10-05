"""
The single GPU slot.

LoRA training needs the whole GPU, so only one training stage may run at a
time on this machine, whether it comes from a web session or the CLI. The slot
is a file lock in the runs folder, which works across threads and processes.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from filelock import FileLock, Timeout

from pipeline.runs import RunCancelled, runs_base

LOCK_NAME = ".gpu.lock"


@contextmanager
def gpu_slot(
    cancel: threading.Event | None = None,
    on_wait: Callable[[], None] | None = None,
    poll_s: float = 1.0,
) -> Iterator[None]:
    """Hold the GPU slot for the duration of the block.

    While another holder has it, *on_wait* is called once and the wait checks
    *cancel* every *poll_s* seconds, so a queued run can still be cancelled.
    """
    base = runs_base()
    base.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(base / LOCK_NAME), thread_local=False)
    waited = False
    while True:
        if cancel is not None and cancel.is_set():
            raise RunCancelled()
        try:
            lock.acquire(timeout=poll_s)
            break
        except Timeout:
            if not waited and on_wait is not None:
                on_wait()
            waited = True
    try:
        yield
    finally:
        lock.release()
