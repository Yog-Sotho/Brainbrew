"""
tests/test_jobs.py

Phase 3 operability: the background job runner, run states, the single GPU
slot, logging, pricing, per-user run visibility and the version.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import tomllib
from pathlib import Path

import pytest

from config import DistillationConfig
from pipeline import jobs, logs, pricing
from pipeline.gpu import gpu_slot
from pipeline.jobs import JobRunner, list_runs, run_state
from pipeline.runs import RunCancelled, create_run, runs_base
from pipeline.version import __version__

CFG = DistillationConfig(teacher_model="m", base_url="http://localhost:8000/v1")


def _wait(pred, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


class _FakeRun:
    """Stands in for run_distillation: reports progress, honours cancel, can block."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Semaphore(0)
        self.running = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.fail = False

    def __call__(self, cfg, source, progress, *, run, cancel, owner):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        self.started.release()
        try:
            progress(40, "Working")
            while not self.release.is_set():
                if cancel.is_set():
                    raise RunCancelled()
                time.sleep(0.01)
            if self.fail:
                raise RuntimeError("boom")
        finally:
            with self.lock:
                self.running -= 1


class TestJobRunner:

    def test_runs_in_background_with_live_progress(self):
        fake = _FakeRun()
        runner = JobRunner(max_jobs=2, runner=fake)
        job = runner.submit(CFG, create_run(), owner="ann@example.com")
        fake.started.acquire(timeout=5)
        _wait(lambda: job.progress == 40)
        assert job.active and job.stage == "Working"
        assert job.run.read_manifest()["owner"] == "ann@example.com"
        fake.release.set()
        job.future.result(timeout=5)
        assert job.status == "succeeded" and not job.active and job.progress == 100

    def test_concurrency_is_bounded(self):
        fake = _FakeRun()
        runner = JobRunner(max_jobs=2, runner=fake)
        jobs = [runner.submit(CFG, create_run()) for _ in range(4)]
        fake.started.acquire(timeout=5)
        fake.started.acquire(timeout=5)
        time.sleep(0.1)
        assert fake.running == 2 and sum(j.status == "queued" for j in jobs) == 2
        fake.release.set()
        for j in jobs:
            j.future.result(timeout=5)
        assert fake.peak == 2 and all(j.status == "succeeded" for j in jobs)

    def test_cancel_running_and_queued(self):
        fake = _FakeRun()
        runner = JobRunner(max_jobs=1, runner=fake)
        running, queued = runner.submit(CFG, create_run()), runner.submit(CFG, create_run())
        fake.started.acquire(timeout=5)
        assert runner.cancel(queued.run_id) and runner.cancel(running.run_id)
        running.future.result(timeout=5)
        queued.future.result(timeout=5)
        assert running.status == queued.status == "cancelled"
        assert queued.run.read_manifest()["status"] == "cancelled"
        assert not runner.cancel(running.run_id)  # no longer active
        assert not runner.cancel("20990101-000000-abcdef")

    def test_failure_is_recorded(self):
        fake = _FakeRun()
        fake.fail = True
        fake.release.set()
        runner = JobRunner(max_jobs=1, runner=fake)
        job = runner.submit(CFG, create_run())
        job.future.result(timeout=5)
        assert job.status == "failed" and job.error == "boom"

    def test_shutdown_cancels_active_jobs(self):
        fake = _FakeRun()
        runner = JobRunner(max_jobs=1, runner=fake)
        job = runner.submit(CFG, create_run())
        fake.started.acquire(timeout=5)
        runner.shutdown(wait=True)
        assert job.status == "cancelled"

    @pytest.mark.parametrize("raw", ["0", "-1", "two", "1.5", ""])
    def test_bad_max_jobs_is_a_clear_error(self, monkeypatch, raw):
        monkeypatch.setenv("BRAINBREW_MAX_JOBS", raw)
        with pytest.raises(ValueError, match="BRAINBREW_MAX_JOBS must be a positive whole number"):
            JobRunner(runner=_FakeRun())

    def test_max_jobs_from_env(self, monkeypatch):
        monkeypatch.setenv("BRAINBREW_MAX_JOBS", " 3 ")
        runner = JobRunner(runner=_FakeRun())
        assert runner.max_jobs == 3
        runner.shutdown()

    def test_finished_jobs_are_forgotten_beyond_the_limit(self, monkeypatch):
        monkeypatch.setattr(jobs, "MAX_FINISHED_KEPT", 2)
        fake = _FakeRun()
        fake.release.set()
        runner = JobRunner(max_jobs=1, runner=fake)
        done = []
        for _ in range(4):
            job = runner.submit(CFG, create_run())
            job.future.result(timeout=5)
            done.append(job)
        fake.release.clear()
        active = runner.submit(CFG, create_run())
        fake.started.acquire(timeout=5)
        kept = [j.run_id for j in done if runner.get(j.run_id)]
        # Two finished jobs were already remembered when the fifth was submitted.
        assert kept == [done[2].run_id, done[3].run_id]
        assert runner.get(active.run_id) is active
        runner.shutdown()

    def test_exit_cancels_running_jobs(self, tmp_path):
        # At interpreter exit, a running job is cancelled instead of being waited for.
        import subprocess
        import sys

        marker = tmp_path / "outcome"
        code = (
            "import pathlib, sys, threading\n"
            "from config import DistillationConfig\n"
            "from pipeline import jobs\n"
            "from pipeline.runs import RunCancelled, create_run\n"
            "started = threading.Event()\n"
            "def run(cfg, source, progress, *, run, cancel, owner):\n"
            "    started.set()\n"
            "    if cancel.wait(60):\n"
            "        pathlib.Path(sys.argv[1]).write_text('cancelled')\n"
            "        raise RunCancelled()\n"
            "    pathlib.Path(sys.argv[1]).write_text('ran to the end')\n"
            "jobs._runner = jobs.JobRunner(max_jobs=1, runner=run)\n"
            "jobs._register_shutdown()\n"
            "cfg = DistillationConfig(teacher_model='m', base_url='http://localhost:8000/v1')\n"
            "jobs._runner.submit(cfg, create_run())\n"
            "started.wait(30)\n"
        )
        env = {**os.environ, "BRAINBREW_RUNS_DIR": str(runs_base())}
        start = time.monotonic()
        subprocess.run([sys.executable, "-c", code, str(marker)], env=env, check=True, timeout=60,
                       cwd=Path(__file__).resolve().parent.parent)
        assert marker.read_text() == "cancelled"
        assert time.monotonic() - start < 30


class TestRunState:

    def test_final_states_pass_through(self):
        for status in ("succeeded", "failed", "cancelled"):
            assert run_state({"status": status}) == status

    def test_dead_process_means_interrupted(self):
        dead_pid = 2**22 + 12345  # beyond the default pid range
        assert run_state({"status": "running", "pid": dead_pid, "host": os.uname().nodename}) == "interrupted"

    def test_this_process_without_a_job_means_interrupted(self):
        assert run_state({"status": "running", "pid": os.getpid()}) == "interrupted"

    def test_live_other_process(self):
        assert run_state({"status": "running", "pid": os.getppid()}) == "running elsewhere"

    def test_runner_knows_best(self):
        fake = _FakeRun()
        runner = JobRunner(max_jobs=1, runner=fake)
        job = runner.submit(CFG, create_run())
        fake.started.acquire(timeout=5)
        assert run_state(job.run.read_manifest(), runner) == "running"
        fake.release.set()
        job.future.result(timeout=5)


class TestListRuns:

    def test_newest_first_and_owner_filter(self):
        a, b = create_run(), create_run()
        a.update_manifest(owner="ann")
        b.update_manifest(owner="bob")
        (runs_base() / "not-a-run").mkdir()
        assert [r.run_id for r in list_runs(runs_base())] == sorted([a.run_id, b.run_id], reverse=True)
        assert [r.run_id for r in list_runs(runs_base(), "ann", only_owner=True)] == [a.run_id]

    def test_missing_base(self, tmp_path):
        assert list_runs(tmp_path / "nope") == []


class TestGpuSlot:

    def test_one_holder_at_a_time(self):
        order: list[str] = []
        first_in = threading.Event()

        def first():
            with gpu_slot():
                order.append("first in")
                first_in.set()
                time.sleep(0.3)
                order.append("first out")

        def second():
            first_in.wait(5)
            waited = []
            with gpu_slot(on_wait=lambda: waited.append(1), poll_s=0.05):
                order.append("second in")
            assert waited == [1]

        threads = [threading.Thread(target=first), threading.Thread(target=second)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        assert order == ["first in", "first out", "second in"]

    def test_excludes_other_processes(self, tmp_path):
        # The web server and the CLI are separate processes; the slot must exclude across them.
        import subprocess
        import sys

        marker = tmp_path / "held"
        code = (
            "import time, pathlib, sys\n"
            "from pipeline.gpu import gpu_slot\n"
            "with gpu_slot():\n"
            "    pathlib.Path(sys.argv[1]).write_text(str(time.time()))\n"
            "    time.sleep(1.0)\n"
        )
        env = {**os.environ, "BRAINBREW_RUNS_DIR": str(runs_base())}
        child = subprocess.Popen([sys.executable, "-c", code, str(marker)], env=env,
                                 cwd=Path(__file__).resolve().parent.parent)
        _wait(marker.exists, timeout=30)
        start = time.time()
        with gpu_slot(poll_s=0.05):
            waited = time.time() - start
        assert child.wait(timeout=30) == 0
        assert waited > 0.5, "the slot was entered while another process held it"

    def test_cancel_while_waiting(self):
        cancel = threading.Event()
        held = threading.Event()
        done = threading.Event()

        def holder():
            with gpu_slot():
                held.set()
                done.wait(5)

        t = threading.Thread(target=holder)
        t.start()
        held.wait(5)
        cancel.set()
        with pytest.raises(RunCancelled), gpu_slot(cancel, poll_s=0.05):
            pass
        done.set()
        t.join(5)


class TestLogs:

    def test_json_lines_and_run_log(self, tmp_path, capsys):
        logs.configure_logging(fmt="json", force=True)
        log_path = tmp_path / "run.log"
        with logs.run_log(log_path, "run-1"):
            logging.getLogger("some.library").warning("from stdlib")
            logs.structlog.get_logger("brainbrew").info("from structlog", n=1)
        logging.getLogger("x").warning("after the run")
        lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert [line["event"] for line in lines] == ["from stdlib", "from structlog"]
        assert all(line["run_id"] == "run-1" and "timestamp" in line for line in lines)
        err = capsys.readouterr().err.strip().splitlines()
        assert json.loads(err[-1])["event"] == "after the run"
        logs.configure_logging(force=True)

    def test_quiet_console_still_fills_the_run_log(self, tmp_path, capsys):
        # The CLI asks for WARNING on the console; the run log must still get INFO lines.
        logs.configure_logging(level="WARNING", fmt="json", force=True)
        log_path = tmp_path / "run.log"
        with logs.run_log(log_path, "run-2"):
            logs.structlog.get_logger("brainbrew").info("progress note")
        assert json.loads(log_path.read_text(encoding="utf-8"))["event"] == "progress note"
        assert "progress note" not in capsys.readouterr().err
        logs.configure_logging(force=True)

    def test_configure_is_idempotent(self):
        logs.configure_logging(force=True)
        logs.configure_logging()
        named = [h for h in logging.getLogger().handlers if h.get_name() == "brainbrew"]
        assert len(named) == 1

    @pytest.mark.parametrize(("fmt", "expected"), [("json", True), ("console", False)])
    def test_format_choice(self, fmt, expected):
        assert logs.use_json(fmt) is expected


class TestPricing:

    def test_longest_match_wins(self):
        assert pricing.price("gpt-4o-mini-2024-07-18") == pricing.PRICES["gpt-4o-mini"]
        assert pricing.price("openai/gpt-4o") == pricing.PRICES["gpt-4o"]
        assert pricing.price("Qwen/Qwen3-4B") is None

    def test_usage_cost(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
        assert pricing.usage_cost("gpt-4o-mini", usage) == pytest.approx(0.75)
        assert pricing.usage_cost("my-model", usage) is None
        assert pricing.usage_cost("my-model", usage, local=True) == 0.0

    def test_local_hosts(self):
        assert pricing.is_local("http://127.0.0.1:8000/v1") and not pricing.is_local("https://api.example.com/v1")
        assert not pricing.is_local(None)


class TestVisibleRun:

    def test_owner_scoping_when_login_is_on(self, monkeypatch):
        from ui import common

        mine, theirs = create_run(), create_run()
        mine.update_manifest(owner="ann@example.com")
        theirs.update_manifest(owner="bob@example.com")
        monkeypatch.setattr(common, "login_required", lambda: True)
        monkeypatch.setattr(common, "current_owner", lambda: "ann@example.com")
        assert common.visible_run(mine.run_id) == mine
        assert common.visible_run(theirs.run_id) is None
        assert common.visible_run("../etc") is None

    def test_everything_visible_in_single_user_mode(self, monkeypatch):
        from ui import common

        run = create_run()
        monkeypatch.setattr(common, "login_required", lambda: False)
        assert common.visible_run(run.run_id) == run


def test_version_matches_pyproject():
    data = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == __version__


class TestServerHfToken:

    @pytest.mark.parametrize(("repo", "namespace", "login", "allowed"), [
        ("anyone/data", "", False, True),        # single user: the visitor is the operator
        ("ops/data", "ops", True, True),
        ("victim/data", "ops", True, False),
        ("opsx/data", "ops", True, False),       # prefix must be the whole namespace
        ("ops/data", "", True, False),           # no namespace configured: never with login
        (None, "ops", True, False),
    ])
    def test_scope(self, repo, namespace, login, allowed):
        from ui.common import server_hf_token_allowed

        assert server_hf_token_allowed(repo, namespace, login) is allowed


class TestCustomEndpoints:

    @pytest.mark.parametrize(("flag", "login", "allowed"), [
        (None, None, True),          # single user: the visitor is the operator
        (None, "1", False),          # shared server: fail closed
        ("1", "1", True),            # operator opted in
        ("0", None, False),
        ("false", None, False),
        ("yes", "1", True),
    ])
    def test_default_depends_on_login(self, monkeypatch, flag, login, allowed):
        from ui.common import custom_endpoints_allowed

        for var, value in (("BRAINBREW_ALLOW_CUSTOM_ENDPOINTS", flag), ("BRAINBREW_REQUIRE_LOGIN", login)):
            if value is None:
                monkeypatch.delenv(var, raising=False)
            else:
                monkeypatch.setenv(var, value)
        assert custom_endpoints_allowed() is allowed
