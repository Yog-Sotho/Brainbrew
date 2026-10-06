"""
tests/test_cli.py

The headless CLI (Phase 3.4) end to end: real config handling, real pipeline
and run folders, with only the model server replaced by FakeOpenAI.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx2
import pytest
from typer.testing import CliRunner

import cli
import orchestrator
from engine import ChatClient, EndpointSettings
from pipeline.records import read_records
from pipeline.runs import RunCancelled, open_run
from tests.fake_openai import FakeOpenAI

TEXT = "\n\n".join(
    f"Section {i}. " + (f"Topic {i} covers distinct material about subject number {i}. ") * 12 for i in range(4)
)
runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "HF_TOKEN", "BRAINBREW_DEFAULT_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def doc(tmp_path: Path) -> Path:
    p = tmp_path / "notes.txt"
    p.write_text(TEXT, encoding="utf-8")
    return p


@contextmanager
def _fake_server(fake: FakeOpenAI | None = None):
    fake = fake or FakeOpenAI()
    made: list[EndpointSettings] = []

    def make(settings: EndpointSettings) -> ChatClient:
        made.append(settings)
        return ChatClient(settings, http_client=httpx2.AsyncClient(transport=fake.transport))

    with patch.object(orchestrator, "_make_client", make):
        yield made


def _invoke(*args: str):
    return runner.invoke(cli.app, list(args))


LOCAL = ("--base-url", "http://localhost:8000/v1")


class TestRun:

    def test_generates_and_copies_the_dataset(self, doc, tmp_path):
        out = tmp_path / "out" / "data.jsonl"
        with _fake_server():
            res = _invoke("run", str(doc), *LOCAL, "-m", "my-model", "-n", "12", "--format", "sharegpt", "-o", str(out), "-q")
        assert res.exit_code == 0, res.output
        assert "12 pairs" in res.output and "cost $0.0000" in res.output
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 12 and "conversations" in json.loads(lines[0])

    def test_config_file_with_cli_overrides(self, doc, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("teacher_model: from-file\ndataset_size: 30\nquality_mode: fast\n"
                       "base_url: http://localhost:8000/v1\n", encoding="utf-8")
        with _fake_server() as made:
            res = _invoke("run", str(doc), "-c", str(cfg), "-n", "11", "-q")
        assert res.exit_code == 0, res.output
        assert {s.model for s in made} == {"from-file"}
        run_id = next(line.split()[1] for line in res.output.splitlines() if line.startswith("Run "))
        manifest = open_run(run_id).read_manifest()
        assert manifest["config"]["dataset_size"] == 11 and manifest["config"]["quality_mode"] == "fast"
        assert open_run(run_id).log.read_text(encoding="utf-8").count('"run_id"') >= 3  # quiet console, full run log
        assert len(read_records(open_run(run_id).records)) == 11

    def test_key_comes_from_the_environment_only(self, doc, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        with _fake_server() as made:
            res = _invoke("run", str(doc), "-n", "10", "-q")
        assert res.exit_code == 0, res.output
        assert all(s.api_key == "sk-from-env" for s in made)
        assert "sk-from-env" not in res.output

    def test_secrets_in_config_are_refused(self, doc, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("teacher_model: m\napi_key: sk-oops\n", encoding="utf-8")
        res = _invoke("run", str(doc), "-c", str(cfg))
        assert res.exit_code == 2 and "Put secrets in the environment" in res.output

    def test_invalid_settings_are_explained(self, doc):
        res = _invoke("run", str(doc), *LOCAL, "-n", "5")
        assert res.exit_code == 2 and "dataset_size" in res.output

    def test_openai_without_key_is_explained(self, doc):
        res = _invoke("run", str(doc))
        assert res.exit_code == 2 and "An API key is required" in res.output

    def test_bad_config_file(self, doc, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("- just\n- a list\n", encoding="utf-8")
        assert _invoke("run", str(doc), "-c", str(cfg)).exit_code == 2

    def test_failed_run_exits_1(self, doc):
        with _fake_server(FakeOpenAI(auth_error=True)):
            res = _invoke("run", str(doc), *LOCAL, "-n", "10", "-q")
        assert res.exit_code == 1 and "failed" in res.output

    def test_cancelled_run_exits_130(self, doc):
        with patch.object(orchestrator, "run_distillation", side_effect=RunCancelled()):
            res = _invoke("run", str(doc), *LOCAL, "-q")
        assert res.exit_code == 130 and "Cancelled" in res.output

    def test_unreadable_document_is_a_warning(self, doc, tmp_path):
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"%PDF-1.7 not really")
        with _fake_server():
            res = _invoke("run", str(bad), str(doc), *LOCAL, "-n", "10", "-q")
        assert res.exit_code == 0 and "Could not parse 'broken.pdf'" in res.output

    def test_no_text_at_all(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("   ", encoding="utf-8")
        res = _invoke("run", str(empty), *LOCAL)
        assert res.exit_code == 2 and "No text" in res.output

    def test_progress_bar_mode(self, doc):
        with _fake_server():
            res = _invoke("run", str(doc), *LOCAL, "-n", "10")
        assert res.exit_code == 0, res.output


class TestRuns:

    def test_list_and_show(self, doc):
        with _fake_server():
            assert _invoke("run", str(doc), *LOCAL, "-n", "10", "-q").exit_code == 0
        listed = json.loads(_invoke("runs", "list", "--json").output)
        assert len(listed) == 1 and listed[0]["status"] == "succeeded" and listed[0]["pairs"] == 10
        table = _invoke("runs", "list").output
        assert listed[0]["run_id"] in table
        shown = json.loads(_invoke("runs", "show", listed[0]["run_id"]).output)
        assert shown["run_id"] == listed[0]["run_id"] and "api_key" not in json.dumps(shown)

    def test_show_rejects_bad_ids(self):
        assert _invoke("runs", "show", "../../etc").exit_code == 2
        assert _invoke("runs", "show", "20990101-000000-abcdef").exit_code == 2


def test_version():
    res = _invoke("version")
    assert res.exit_code == 0 and res.output.strip() == cli.__version__
