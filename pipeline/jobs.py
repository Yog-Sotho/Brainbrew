"""
Background job runner for generation runs.

The web app hands each run to one process-wide JobRunner, so a run keeps going
when the browser tab is closed or refreshed, several sessions can run at once
(BRAINBREW_MAX_JOBS, default 2) and any session can cancel a run. Training
inside a run waits for the machine's single GPU slot (pipeline/gpu.py).

The run's manifest is the durable record; the in-memory Job adds live
progress. A manifest left "running" by a process that is gone is reported as
interrupted (see `run_state`).
"""
from __future__ import annotations

import os
import socket
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from config import DistillationConfig
from pipeline.runs import ACTIVE_STATES, RunCancelled, RunDir, utc_now

logger = structlog.get_logger(__name__)

MAX_JOBS_ENV = "BRAINBREW_MAX_JOBS"
MAX_FINISHED_KEPT = 200  # finished jobs remembered in memory; their manifests stay on disk


def max_jobs_from_env() -> int:
    raw = os.getenv(MAX_JOBS_ENV, "2").strip()
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{MAX_JOBS_ENV} must be a positive whole number, got {raw!r}.")
    return int(raw)


@dataclass
class Job:
    """One run handed to the runner."""

    run: RunDir
    owner: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    progress: int = 0
    stage: str = "Queued"
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    error: str | None = None
    future: Future[None] | None = None

    @property
    def run_id(self) -> str:
        return self.run.run_id

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def on_progress(self, pct: int, stage: str) -> None:
        self.progress, self.stage = pct, stage


Runner = Callable[..., Any]


def _default_runner() -> Runner:
    from orchestrator import run_distillation

    return run_distillation


class JobRunner:
    """A small thread pool for runs. One instance per server process."""

    def __init__(self, max_jobs: int | None = None, runner: Runner | None = None) -> None:
        self.max_jobs = max_jobs or max_jobs_from_env()
        self._executor = ThreadPoolExecutor(max_workers=self.max_jobs, thread_name_prefix="brainbrew-run")
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, cfg: DistillationConfig, run: RunDir, owner: str | None = None) -> Job:
        """Queue *run* (its source.txt must already be written) and return its Job."""
        job = Job(run=run, owner=owner)
        run.update_manifest(status="queued", queued_at=utc_now(), pid=os.getpid(),
                            host=socket.gethostname(), **({"owner": owner} if owner else {}))
        with self._lock:
            self._forget_old_jobs()
            self._jobs[run.run_id] = job
        job.future = self._executor.submit(self._execute, job, cfg)
        logger.info("Run queued", run_id=run.run_id)
        return job

    def _execute(self, job: Job, cfg: DistillationConfig) -> None:
        if job.cancel_event.is_set():  # cancelled while queued
            job.status, job.stage = "cancelled", "Cancelled"
            job.run.update_manifest(status="cancelled", finished_at=utc_now(), stage="Cancelled")
            return
        job.status, job.stage = "running", "Starting"
        runner = self._runner or _default_runner()
        try:
            runner(cfg, job.run.source, job.on_progress, run=job.run, cancel=job.cancel_event, owner=job.owner)
        except RunCancelled:
            job.status, job.stage = "cancelled", "Cancelled"
        except Exception as exc:  # recorded in the manifest by the orchestrator too
            job.status, job.error = "failed", str(exc)[:1000]
            logger.warning("Run failed", run_id=job.run_id, error=job.error)
        else:
            job.status, job.progress, job.stage = "succeeded", 100, "Done"

    def _forget_old_jobs(self) -> None:
        """Keep memory bounded on a long-running server (caller holds the lock)."""
        finished = [rid for rid, j in self._jobs.items() if not j.active]
        for rid in finished[: max(0, len(finished) - MAX_FINISHED_KEPT)]:
            del self._jobs[rid]

    def get(self, run_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        """Ask a queued or running job to stop. Returns False if it is not active here."""
        job = self.get(run_id)
        if job is None or not job.active:
            return False
        job.cancel_event.set()
        logger.info("Cancel requested", run_id=run_id)
        return True

    def active_jobs(self) -> list[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.active]

    def shutdown(self, wait: bool = True) -> None:
        for job in self.active_jobs():
            job.cancel_event.set()
        self._executor.shutdown(wait=wait, cancel_futures=False)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def run_state(manifest: dict[str, Any], runner: JobRunner | None = None) -> str:
    """The run's status as a user should see it.

    A run the manifest calls active but that no live process owns (the server
    restarted, or the CLI was killed) is "interrupted"; one that another live
    process on this host is running is "running elsewhere".
    """
    status = str(manifest.get("status") or "unknown")
    if status not in ACTIVE_STATES:
        return status
    run_id = str(manifest.get("run_id") or "")
    if runner is not None and (job := runner.get(run_id)) is not None:
        return job.status
    pid, host = manifest.get("pid"), manifest.get("host")
    if not isinstance(pid, int) or (host and host != socket.gethostname()):
        return status if status == "created" else "unknown"
    if pid == os.getpid():
        return "interrupted"  # this process would know about it
    return "running elsewhere" if _pid_alive(pid) else "interrupted"


def list_runs(base: Path, owner: str | None = None, only_owner: bool = False) -> list[RunDir]:
    """Run folders under *base*, newest first; with *only_owner*, only those *owner* started."""
    from pipeline.runs import open_run

    runs: list[RunDir] = []
    if not base.is_dir():
        return runs
    for path in sorted(base.iterdir(), reverse=True):
        try:
            run = open_run(path.name, base)
        except (ValueError, FileNotFoundError):
            continue
        if only_owner and run.read_manifest().get("owner") != owner:
            continue
        runs.append(run)
    return runs


_runner: JobRunner | None = None
_runner_lock = threading.Lock()


def get_runner() -> JobRunner:
    """The process-wide runner shared by every web session."""
    global _runner
    with _runner_lock:
        if _runner is None:
            _runner = JobRunner()
            _register_shutdown()
        return _runner


def _cancel_on_exit() -> None:
    """At interpreter shutdown, cancel running jobs so they end as "cancelled"."""
    if _runner is not None:
        _runner.shutdown(wait=False)


def _register_shutdown() -> None:
    # Python joins worker threads at exit *before* ordinary atexit handlers run,
    # so a plain atexit hook would wait for every run to finish. threading's own
    # exit hooks run first (concurrent.futures uses the same mechanism).
    register = getattr(threading, "_register_atexit", None)
    if register is not None:
        register(_cancel_on_exit)
    else:  # pragma: no cover - other Python implementations
        import atexit

        atexit.register(_cancel_on_exit)
