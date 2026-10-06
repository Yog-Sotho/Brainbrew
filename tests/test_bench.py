"""
tests/test_bench.py

The Phase 2 benchmark (bench/run_bench.py) end to end on the real fixture
corpus, with the model server replaced by FakeOpenAI. The nightly workflow runs
the same script against a real model.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest

from bench import run_bench
from engine import ChatClient, EndpointSettings
from pipeline.records import Record
from tests.fake_openai import FakeOpenAI


def _factory(fake: FakeOpenAI):
    made: list[EndpointSettings] = []

    def make(settings: EndpointSettings) -> ChatClient:
        made.append(settings)
        return ChatClient(settings, http_client=httpx2.AsyncClient(transport=fake.transport))

    return make, made


def _bench(tmp_path: Path, fake: FakeOpenAI, *extra: str):
    out = tmp_path / "report.json"
    make, made = _factory(fake)
    code = run_bench.main(["--model", "m", "--base-url", "http://fake/v1", "--scale", "0.4",
                           "--only", "elements_of_style.pdf", "--out", str(out), *extra], make_client=make)
    return code, json.loads(out.read_text(encoding="utf-8")), made


def test_fixture_corpus_is_present():
    assert sorted(p.name for p in run_bench.FIXTURES.glob("*.pdf")) == sorted(run_bench.TARGETS)


def test_passing_run(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    code, report, made = _bench(tmp_path, FakeOpenAI())
    (doc,) = report["documents"]
    assert code == 0 and report["passed"], doc["failures"]
    assert doc["records"] == doc["target"] == 10
    assert doc["faithfulness"] == 5.0 and doc["judged"] == 10
    assert doc["refusal_rate"] == 0 and doc["near_dup_rate"] < 0.02
    assert made[-1].temperature == 0.0  # the independent judge pass
    assert "PASS" in summary.read_text(encoding="utf-8")


def test_low_faithfulness_fails_the_gate(tmp_path):
    code, report, _ = _bench(tmp_path, FakeOpenAI(low_score_every=2), "--mode", "fast")
    (doc,) = report["documents"]
    assert code == 1 and not report["passed"]
    assert any("faithfulness" in f for f in doc["failures"])


def test_failed_run_is_reported(tmp_path):
    code, report, _ = _bench(tmp_path, FakeOpenAI(auth_error=True))
    (doc,) = report["documents"]
    assert code == 1 and doc["error"].startswith("AuthenticationError")
    assert doc["fatal"]


def test_fatal_error_stops_after_the_first_document(tmp_path, capsys):
    # A rejected key fails every document the same way: report it once.
    out = tmp_path / "report.json"
    make, _ = _factory(FakeOpenAI(auth_error=True))
    code = run_bench.main(["--model", "m", "--base-url", "http://fake/v1", "--scale", "0.4", "--out", str(out)],
                          make_client=make)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and len(report["documents"]) == 1 and report["documents"][0]["fatal"]
    assert "every other document would fail" in capsys.readouterr().err


def test_no_credits_stops_the_benchmark_at_once(tmp_path, capsys):
    # 429 "no credits" looks like a rate limit but waiting does not help.
    out = tmp_path / "report.json"
    fake = FakeOpenAI(quota_exhausted=True)
    make, _ = _factory(fake)
    code = run_bench.main(["--model", "m", "--base-url", "http://fake/v1", "--scale", "0.4", "--out", str(out)],
                          make_client=make)
    (doc,) = json.loads(out.read_text(encoding="utf-8"))["documents"]
    assert code == 1 and doc["fatal"] and "no credits" in doc["error"]
    assert fake.count("quota") < 10  # no retries, and the run stopped instead of trying every chunk


def test_ordinary_failures_do_not_stop_the_benchmark(tmp_path):
    out = tmp_path / "report.json"
    make, _ = _factory(FakeOpenAI(low_score_every=2))
    code = run_bench.main(["--model", "m", "--base-url", "http://fake/v1", "--scale", "0.4", "--mode", "fast",
                           "--only", "elements_of_style.pdf", "federalist_10.pdf", "--out", str(out)],
                          make_client=make)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and len(report["documents"]) == 2 and not any(d["fatal"] for d in report["documents"])


@pytest.mark.parametrize(("questions", "rate"), [
    (["What is a comma used for?", "What is a comma used for ?", "Why use active voice?"], 1 / 3),
    (["What is a comma used for?", "Why use active voice?"], 0.0),
])
def test_near_dup_rate(questions, rate):
    recs = [Record(instruction=q, output="An answer long enough to be a real answer.") for q in questions]
    assert run_bench.near_dup_rate(recs) == pytest.approx(rate)


def test_gate_checks():
    res = run_bench.DocResult(document="d", target=10, records=8, faithfulness=4.5, yield_ratio=0.8)
    assert run_bench.check_gate(res) == ["yield 8/10 outside ±10%"]
