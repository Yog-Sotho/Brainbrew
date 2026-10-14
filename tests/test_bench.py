"""
tests/test_bench.py

The Phase 2 benchmark (bench/run_bench.py) end to end on the real fixture
corpus, with the model server replaced by FakeOpenAI. The nightly workflow runs
the same script against a real model.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx2
import pytest

from bench import run_bench
from engine import ChatClient, EndpointSettings
from pipeline.records import Record
from tests.fake_openai import FakeOpenAI


def _factory(fake: FakeOpenAI, retries: int | None = None):
    made: list[EndpointSettings] = []

    def make(settings: EndpointSettings) -> ChatClient:
        if retries is not None:  # an outage would otherwise sit out the SDK's backoff
            settings = dataclasses.replace(settings, max_retries=retries)
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


def test_provider_outage_is_a_warning_and_stops_the_run(tmp_path, capsys, monkeypatch):
    # Gemini answered 500 to every request for a night: nothing to measure, nothing Brainbrew did.
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    out = tmp_path / "report.json"
    make, _ = _factory(FakeOpenAI(server_down=True), retries=0)
    code = run_bench.main(["--model", "m", "--base-url", "http://fake/v1", "--scale", "0.4",
                           "--only", "elements_of_style.pdf", "federalist_10.pdf", "--out", str(out)],
                          make_client=make)
    report = json.loads(out.read_text(encoding="utf-8"))
    (doc,) = report["documents"]  # stopped after the first document
    assert code == 0 and report["passed"] and report["provider_down"]
    assert doc["provider_down"] and not doc["failures"] and doc["provider_errors"] > 0
    assert doc["warnings"] == [f"no pairs: the provider failed {doc['provider_errors']} requests"]
    captured = capsys.readouterr()
    assert "Not a Brainbrew failure" in captured.err
    assert "::warning title=Provider errors (elements_of_style.pdf)::" in captured.out
    assert "NOT MEASURED" in run_bench.summary_markdown(report)


def test_no_pairs_without_provider_errors_still_fails(tmp_path):
    # Every answer refused: a quality problem, not an outage.
    code, report, _ = _bench(tmp_path, FakeOpenAI(refuse_every=1))
    (doc,) = report["documents"]
    assert code == 1 and not report["provider_down"]
    assert doc["failures"][0].startswith("run failed: NoPairsError") and not doc["warnings"]


@pytest.mark.parametrize(("stats", "count"), [
    ({"errors": 61, "last_error": "InternalServerError: Error code: 500 - [{'error': {'code': 500}}]"}, 61),
    ({"errors": 3, "last_error": "APITimeoutError: Request timed out."}, 3),
    ({"errors": 2, "last_error": "APIConnectionError: Connection error. <- ConnectError: refused"}, 2),
    ({"errors": 4, "last_error": "BadRequestError: Error code: 400 - bad request"}, 0),
    ({"errors": 0, "last_error": None}, 0),
    ({}, 0),
])
def test_provider_errors(stats, count):
    assert run_bench.provider_errors(stats) == count


def test_provider_errors_excuse_only_the_shortfall_they_explain():
    # 2026-10-13: 61 requests failed on Google's side and one document got 10 of 15 pairs.
    res = run_bench.DocResult(document="d", target=15, records=10, faithfulness=5.0, yield_ratio=10 / 15,
                              provider_errors=61)
    res.failures = run_bench.check_gate(res)
    run_bench.excuse_provider_shortfall(res)
    assert not res.failures and res.warnings == ["yield 10/15 outside ±10%: the provider failed 61 requests"]

    few = run_bench.DocResult(document="d", target=15, records=10, faithfulness=3.0, yield_ratio=10 / 15,
                              provider_errors=3)  # 3 failures can't explain 5 missing pairs
    few.failures = run_bench.check_gate(few)
    run_bench.excuse_provider_shortfall(few)
    assert len(few.failures) == 2 and not few.warnings

    quality = run_bench.DocResult(document="d", target=15, records=10, faithfulness=3.0, yield_ratio=10 / 15,
                                  provider_errors=61)  # the quality gates are never excused
    quality.failures = run_bench.check_gate(quality)
    run_bench.excuse_provider_shortfall(quality)
    assert quality.failures == ["faithfulness 3.00 < 4.0"] and quality.warnings


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


def test_reasoning_effort_reaches_every_request(tmp_path):
    fake = FakeOpenAI()
    code, _, made = _bench(tmp_path, fake, "--reasoning-effort", "none")
    assert code == 0
    assert fake.calls and all(c.get("reasoning_effort") == "none" for c in fake.calls)
    assert {s.reasoning_effort for s in made} == {"none"}
