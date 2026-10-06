"""
tests/test_downstream.py

The downstream evaluation harness (bench/downstream.py) end to end, with the
model server replaced by a small fake: a writer that produces exam questions, a
"base" student that does not know the documents, a "tuned" student that does,
and a judge that compares answers with the reference.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx2
import pytest

from bench import downstream
from engine import ChatClient, EndpointSettings
from pipeline.records import Record, write_records

DOC = "\n\n".join(
    f"Topic {i}: the {name} rule states that {name} values must be checked before use in stage {i}."
    for i, name in enumerate(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"])
) * 4

TRAIN_QUESTION = "What does the alpha rule state about alpha values?"


def _reference(question: str) -> str:
    return f"The reference answer to: {question}"


class FakeServer:
    """Routes chat requests by their system prompt."""

    def __init__(self, base_knows: bool = False, fail_model: str | None = None) -> None:
        self.base_knows = base_knows
        self.fail_model = fail_model
        self.exam_calls = 0
        self.keys: list[str | None] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.keys.append(request.headers.get("authorization"))
        body = json.loads(request.content)
        system, user = body["messages"][0]["content"], body["messages"][-1]["content"]
        if body["model"] == self.fail_model:
            return httpx2.Response(400, json={"error": {"message": "unknown model"}})
        if system == downstream.EXAM_SYSTEM:
            content = self._exam(user)
        elif system == downstream.STUDENT_SYSTEM:
            knows = body["model"] == "tuned" or self.base_knows
            content = _reference(user) if knows else "I do not know."
        elif system == downstream.GRADER_SYSTEM:
            reference = re.search(r"Reference answer: (.*)\n", user).group(1)
            answer = user.split("Student answer:\n<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
            content = json.dumps({"score": 5 if answer == reference else 1, "reason": "compared"})
        else:  # pragma: no cover - a prompt this fake does not know
            return httpx2.Response(500, json={"error": {"message": "unexpected prompt"}})
        return httpx2.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    def _exam(self, user: str) -> str:
        self.exam_calls += 1
        topic = re.search(r"Topic (\d+): the (\w+) rule", user)
        n, name = (topic.group(1), topic.group(2)) if topic else ("0", "alpha")
        fresh = [t.format(n=n, name=name) for t in TEMPLATES[2 * (self.exam_calls - 1):2 * self.exam_calls]]
        items = [{"question": q, "answer": _reference(q)} for q in fresh] + [
            # Three kinds the harness must drop: a training question, one citing
            # the source, and an empty answer.
            {"question": TRAIN_QUESTION, "answer": "They must be checked."},
            {"question": "According to the passage, what is the first rule?", "answer": "Alpha."},
            {"question": f"Which department audits stage {n} each quarter?", "answer": ""},
        ]
        return json.dumps({"items": items})


TEMPLATES = [
    "When must {name} values be checked in stage {n}?",
    "Why is validation required before using {name} data?",
    "Which pipeline step comes first for {name} inputs?",
    "Who signs off on unverified {name} entries?",
    "How often are {name} records re-examined during audits?",
    "What error is raised for malformed {name} payloads?",
    "Where are rejected {name} items stored afterwards?",
    "Name the tool used to inspect {name} batches.",
    "What threshold triggers a {name} alarm?",
    "Under which condition can {name} checks be skipped?",
    "Describe the escalation path for {name} failures.",
    "List the outputs produced after {name} review completes.",
]


def _factory(server: FakeServer):
    made: list[EndpointSettings] = []

    def make(settings: EndpointSettings) -> ChatClient:
        made.append(settings)
        return ChatClient(settings, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle)))

    return make, made


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    doc = tmp_path / "doc.txt"
    doc.write_text(DOC, encoding="utf-8")
    records = tmp_path / "records.jsonl"
    write_records(records, [Record(instruction=TRAIN_QUESTION, output="They must be checked before use.")])
    return doc, records


def _testset(tmp_path: Path, corpus, server: FakeServer, *extra: str) -> tuple[int, list[dict]]:
    doc, records = corpus
    out = tmp_path / "testset.jsonl"
    make, _ = _factory(server)
    code = downstream.main(["testset", str(doc), "--train", str(records), "--model", "writer",
                            "--base-url", "http://fake/v1", "--size", "6", "--out", str(out), *extra],
                           make_client=make)
    lines = out.read_text(encoding="utf-8").splitlines()
    return code, [json.loads(line) for line in lines]


class TestTestset:

    def test_held_out_questions_only(self, tmp_path, corpus):
        code, items = _testset(tmp_path, corpus, FakeServer())
        assert code == 0 and len(items) == 6
        questions = [i["question"] for i in items]
        assert TRAIN_QUESTION not in questions                       # overlaps the training set
        assert not any("passage" in q.lower() for q in questions)   # cites the source
        assert all(i["reference"] and i["passage"] for i in items)  # empty answers dropped
        assert len(set(questions)) == len(questions)
        assert [i["id"] for i in items] == list(range(6))

    def test_deterministic_for_a_seed(self, tmp_path, corpus):
        first = _testset(tmp_path, corpus, FakeServer(), "--seed", "3")[1]
        second = _testset(tmp_path, corpus, FakeServer(), "--seed", "3")[1]
        assert first == second

    def test_short_testset_exits_nonzero(self, tmp_path, corpus):
        code, items = _testset(tmp_path, corpus, FakeServer(), "--size", "500")
        assert code == 1 and 0 < len(items) < 500

    def test_no_text(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("   ", encoding="utf-8")
        assert downstream.main(["testset", str(empty), "--model", "m", "--base-url", "http://fake/v1",
                                "--out", str(tmp_path / "t.jsonl")]) == 2


class TestEvaluate:

    def _run(self, tmp_path, corpus, server: FakeServer, models=("base", "tuned"), *extra: str):
        _testset(tmp_path, corpus, FakeServer())
        out = tmp_path / "report.json"
        make, made = _factory(server)
        code = downstream.main(["evaluate", str(tmp_path / "testset.jsonl"), "--models", *models,
                                "--student-base-url", "http://student/v1", "--judge-model", "judge",
                                "--judge-base-url", "http://judge/v1", "--out", str(out), *extra],
                               make_client=make)
        return code, json.loads(out.read_text(encoding="utf-8")), made

    def test_tuned_model_beats_baseline(self, tmp_path, corpus):
        code, report, made = self._run(tmp_path, corpus, FakeServer())
        base, tuned = report["models"]
        assert code == 0 and report["baseline"] == "base"
        assert base["mean_score"] == 1.0 and base["accuracy"] == 0.0 and base["vs_baseline"] == {}
        assert tuned["mean_score"] == 5.0 and tuned["accuracy"] == 1.0
        vs = tuned["vs_baseline"]
        assert vs["mean_gain"] == 4.0 and vs["significant"] and vs["wins"] == 6 and vs["losses"] == 0
        assert vs["ci95"] == [4.0, 4.0]
        # Students answer closed book at temperature 0; the judge is deterministic too.
        assert {s.temperature for s in made} == {0.0}
        first = report["answers"][0]
        assert first["base"]["score"] == 1 and first["tuned"]["answer"] == first["reference"]
        assert first["tuned"]["reason"] == "compared"

    def test_no_difference_is_not_significant(self, tmp_path, corpus):
        _, report, _ = self._run(tmp_path, corpus, FakeServer(base_knows=True))
        vs = report["models"][1]["vs_baseline"]
        assert vs["mean_gain"] == 0.0 and not vs["significant"] and vs["ties"] == 6

    def test_failed_model_is_reported(self, tmp_path, corpus):
        code, report, _ = self._run(tmp_path, corpus, FakeServer(fail_model="tuned"), ("base", "tuned"),
                                    "--timeout", "5")
        tuned = report["models"][1]
        assert code == 1 and tuned["graded"] == 0 and tuned["errors"] == 6
        assert report["answers"][0]["tuned"]["answer"].startswith("[error]")

    def test_markdown_summary(self, tmp_path, corpus, capsys):
        self._run(tmp_path, corpus, FakeServer())
        out = capsys.readouterr().out
        assert "| `tuned` | 5.00 | 100% | +4.00 [+4.00, +4.00] * | 6 / 0 / 0 |" in out


class TestKeysAndStats:

    def test_openai_key_only_goes_to_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.delenv("ENDPOINT_API_KEY", raising=False)
        assert downstream.api_key_for(None) == "sk-openai"
        assert downstream.api_key_for("http://127.0.0.1:8000/v1") is None
        monkeypatch.setenv("ENDPOINT_API_KEY", "local-key")
        assert downstream.api_key_for("http://127.0.0.1:8000/v1") == "local-key"

    def test_local_endpoints_never_receive_the_openai_key(self, tmp_path, corpus, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.delenv("ENDPOINT_API_KEY", raising=False)
        server = FakeServer()
        _testset(tmp_path, corpus, server)
        assert server.keys and not any("sk-openai" in (k or "") for k in server.keys)

    def test_bootstrap_interval(self):
        low, high = downstream.paired_bootstrap([1, 0, 1, 1, 0, 2, 1, 0], seed=1)
        assert 0 < low < 0.75 < high < 1.5
        assert downstream.paired_bootstrap([], seed=1) == (0.0, 0.0)
        assert downstream.compare({0: 3, 1: 3}, {1: 5, 2: 1}, seed=1)["questions"] == 1


def test_train_wraps_the_trainer(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_train(records, base, out, rank, **kwargs):
        seen.update(records=records, base=base, out=out, rank=rank, **kwargs)
        return out

    monkeypatch.setattr("training.lora_trainer.train_lora", fake_train)
    assert downstream.main(["train", "r.jsonl", "--base-model", "Qwen/Qwen2.5-1.5B-Instruct",
                            "--out", str(tmp_path / "ad"), "--epochs", "2", "--rank", "8"]) == 0
    assert seen == {"records": Path("r.jsonl"), "base": "Qwen/Qwen2.5-1.5B-Instruct", "out": tmp_path / "ad",
                    "rank": 8, "num_train_epochs": 2.0, "max_length": 2048}
    assert "--max-lora-rank 8" in capsys.readouterr().out
