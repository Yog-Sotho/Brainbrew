"""
tests/test_orchestrator.py

Contract tests for the whole pipeline: real chunking, real async engine and
openai SDK, real filters/judge/dedup, real run directories — only the model
server is replaced by the in-memory FakeOpenAI.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import httpx2
import pytest

import orchestrator
from config import DistillationConfig, OutputFormat, QualityMode
from engine import ChatClient, EndpointSettings
from pipeline.records import read_records
from pipeline.runs import open_run
from tests.fake_openai import FakeOpenAI, fake_answer

TOPICS = [
    "Photosynthesis lets plants turn sunlight, water and carbon dioxide into glucose and oxygen.",
    "The French Revolution began in 1789 and ended the absolute monarchy of Louis XVI.",
    "TCP guarantees ordered delivery of bytes using sequence numbers and acknowledgements.",
    "Plate tectonics explains earthquakes, volcanoes and the slow drift of the continents.",
    "Compound interest grows savings because interest is earned on previous interest.",
]


@pytest.fixture()
def source(tmp_path: Path) -> Path:
    p = tmp_path / "source.txt"
    p.write_text("\n\n".join((t + " ") * 8 for t in TOPICS), encoding="utf-8")
    return p


def _cfg(**overrides) -> DistillationConfig:
    base = {"teacher_model": "gpt-4o-mini", "api_key": "test-key", "dataset_size": 15,
            "quality_mode": QualityMode.BALANCED, "concurrency": 4}
    return DistillationConfig(**{**base, **overrides})


def _run_capture(cfg: DistillationConfig, src: Path, fake: FakeOpenAI | None = None, **kwargs):
    """Run the pipeline against the fake server; also return the clients' settings."""
    fake = fake or FakeOpenAI()
    made: list[EndpointSettings] = []

    def make(settings: EndpointSettings) -> ChatClient:
        made.append(settings)
        return ChatClient(settings, http_client=httpx2.AsyncClient(transport=fake.transport))

    with patch.object(orchestrator, "_make_client", make):
        return orchestrator.run_distillation(cfg, src, **kwargs), made


def _run(cfg: DistillationConfig, src: Path, fake: FakeOpenAI | None = None, **kwargs):
    return _run_capture(cfg, src, fake, **kwargs)[0]


_LAYOUT_KEY = {
    OutputFormat.ALPACA: "instruction", OutputFormat.SHAREGPT: "conversations",
    OutputFormat.CHATML: "messages", OutputFormat.OPENAI: "messages",
}


class TestPipelineContract:

    def test_reaches_target_with_grounded_answers(self, source):
        result = _run(_cfg(), source)
        records = read_records(result.run.records)
        assert result.record_count == 15 == len(records)
        assert all(r.output == fake_answer(r.instruction) for r in records)
        assert all(r.meta["judge"]["faithfulness"] == 5 for r in records)

    def test_every_format_scores_the_same(self, source):
        reports = {}
        for fmt in OutputFormat:
            result = _run(_cfg(output_format=fmt, sanitize_dataset=True), source)
            lines = result.dataset_path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == result.record_count > 0
            assert _LAYOUT_KEY[fmt] in json.loads(lines[0])
            assert result.dataset_path.name == f"dataset.{fmt.value}.jsonl"
            reports[fmt] = (result.record_count, result.quality)
        assert len({json.dumps(r, sort_keys=True) for r in reports.values()}) == 1

    @pytest.mark.parametrize("mode,judged,evolved", [
        (QualityMode.FAST, False, False),
        (QualityMode.BALANCED, True, False),
        (QualityMode.RESEARCH, True, True),
    ])
    def test_quality_modes(self, mode, judged, evolved, source):
        result = _run(_cfg(quality_mode=mode), source)
        gen = result.run.read_manifest()["generation"]
        assert (gen["judged"] > 0) is judged
        assert (gen["evolved"] > 0) is evolved

    def test_endpoint_settings_reach_the_client(self, source):
        cfg = _cfg(base_url="http://localhost:8000/v1", api_key=None, temperature=0.2,
                   max_new_tokens=512, concurrency=3, request_timeout=60, judge_model="judge-m")
        _, (teacher, judge) = _run_capture(cfg, source)
        assert teacher.model == "gpt-4o-mini" and judge.model == "judge-m"
        assert teacher.base_url == "http://localhost:8000/v1" and teacher.api_key is None
        assert (teacher.temperature, teacher.max_tokens, teacher.concurrency, teacher.timeout_s) == (0.2, 512, 3, 60.0)
        assert judge.temperature == 0.0

    def test_embedding_model_enables_semantic_dedup(self, source):
        cfg = _cfg(embedding_model="embed-m", semantic_dedup_threshold=0.95, quality_mode=QualityMode.FAST)
        result, made = _run_capture(cfg, source)
        assert [s.model for s in made] == ["gpt-4o-mini", "embed-m"]
        m = result.run.read_manifest()
        assert m["usage"]["embeddings:embed-m"]["requests"] > 0
        assert m["generation"]["semantic_duplicates"] > 0

    def test_no_embedder_when_dedup_is_off(self, source):
        cfg = _cfg(embedding_model="embed-m", enable_dedup=False, quality_mode=QualityMode.FAST)
        _, made = _run_capture(cfg, source)
        assert [s.model for s in made] == ["gpt-4o-mini"]

    def test_ensemble_uses_every_teacher(self, source):
        result = _run(_cfg(teacher_model="model-a,model-b", quality_mode=QualityMode.FAST), source)
        models = {r.meta["model"] for r in read_records(result.run.records)}
        assert models == {"model-a", "model-b"}


class TestRunDirectory:

    def test_manifest(self, source, _isolated_runs_dir):
        result = _run(_cfg(sanitize_dataset=True), source)
        run = open_run(result.run.run_id)
        assert run.root.parent == _isolated_runs_dir
        m = run.read_manifest()
        assert m["status"] == "succeeded"
        assert m["counts"]["generated"] == 15 and m["counts"]["chunks"] >= 1
        assert m["generation"]["accepted"] == 15
        assert m["usage"]["teacher:gpt-4o-mini"]["requests"] > 0
        assert m["usage"]["judge:gpt-4o-mini"]["requests"] > 0
        assert m["quality"]["grade"] == result.quality["grade"]

    def test_secrets_never_written(self, source):
        secret = "sk-this-must-never-be-written-anywhere"
        result = _run(_cfg(api_key=secret), source)
        for path in result.run.root.rglob("*"):
            if path.is_file():
                assert secret not in path.read_text(encoding="utf-8", errors="ignore"), path

    def test_failed_run_is_marked(self, source, _isolated_runs_dir):
        from openai import AuthenticationError

        with pytest.raises(AuthenticationError):
            _run(_cfg(), source, FakeOpenAI(auth_error=True))
        (root,) = _isolated_runs_dir.iterdir()
        manifest = open_run(root.name).read_manifest()
        assert manifest["status"] == "failed" and "bad key" in manifest["error"]

    def test_existing_run_dir_is_used(self, source):
        run = orchestrator.create_run()
        assert _run(_cfg(), source, run=run).run.root == run.root


class TestFailures:

    def test_nothing_accepted_raises(self, source):
        with pytest.raises(RuntimeError, match="No usable question/answer pairs"):
            _run(_cfg(), source, FakeOpenAI(refuse_every=1))

    def test_sanitizer_removing_everything_raises(self, source):
        with patch("pipeline.sanitizer.check_quality", return_value="rejected"), \
             pytest.raises(RuntimeError, match="Sanitizing removed all"):
            _run(_cfg(sanitize_dataset=True), source)

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _run(_cfg(), tmp_path / "nope.txt")

    def test_empty_source_raises(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            _run(_cfg(), empty)

    def test_oversized_source_raises(self, tmp_path):
        huge = tmp_path / "huge.txt"
        huge.write_bytes(b"x" * (orchestrator.MAX_SOURCE_BYTES + 1))
        with pytest.raises(ValueError, match="exceeds"):
            _run(_cfg(), huge)


class TestProgressAndLogging:

    def test_progress_increases_to_100(self, source):
        values: list[int] = []
        _run(_cfg(), source, progress_callback=values.append)
        assert values == sorted(values) and values[-1] == 100
        assert all(0 <= v <= 100 for v in values)

    def test_api_key_not_logged(self, source):
        secret = "sk-this-must-never-appear-in-logs"
        logged: list[str] = []
        with patch.object(orchestrator, "logger") as log:
            log.info.side_effect = lambda msg, **kw: logged.append(f"{msg} {kw}")
            _run(_cfg(api_key=secret), source)
        assert logged and not any(secret in line for line in logged)

    def test_runs_from_a_worker_thread(self, source):
        """Streamlit runs scripts on worker threads; asyncio.run must work there."""
        outcome: dict[str, object] = {}

        def target() -> None:
            try:
                outcome["result"] = _run(_cfg(), source)
            except BaseException as exc:  # pragma: no cover - reported below
                outcome["error"] = exc

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=120)
        assert "error" not in outcome, outcome.get("error")
        assert outcome["result"].record_count == 15


class TestOptionalStages:

    def test_decontamination_removes_overlapping_records(self, source):
        # Runs are deterministic: make the "benchmark" contain one record of a plain run.
        plain = read_records(_run(_cfg(quality_mode=QualityMode.FAST), source).run.records)
        leaked = plain[3].instruction  # fake answers share a template; questions are unique

        with patch("pipeline.decontam.load_eval_texts", return_value=[leaked]) as load:
            result = _run(_cfg(decontaminate=["gsm8k"], quality_mode=QualityMode.FAST), source)
        load.assert_called_once_with("gsm8k")
        m = result.run.read_manifest()
        assert m["decontamination"] == {"gsm8k": 1}
        assert m["counts"]["after_decontamination"] == m["counts"]["generated"] - 1 == result.record_count
        assert leaked not in {r.instruction for r in read_records(result.run.records)}

    def test_decontamination_removing_everything_raises(self, source):
        with patch("pipeline.decontam.decontaminate", return_value=([], {"gsm8k": 15})), \
             pytest.raises(RuntimeError, match="overlap the selected benchmarks"):
            _run(_cfg(decontaminate=["gsm8k"]), source)

    def test_sanitizer_gets_the_pii_options(self, source):
        from pipeline import sanitizer

        seen = []
        real = sanitizer.sanitize_records

        def spy(records, cfg):
            seen.append(cfg)
            return real(records, cfg)

        with patch("pipeline.sanitizer.sanitize_records", side_effect=spy):
            _run(_cfg(sanitize_dataset=True, pii_url_policy="redact"), source)
        assert seen[0].url_policy == "redact" and seen[0].presidio is False

    def test_training_uses_canonical_records_and_zips_adapter(self, source):
        def fake_train(records_path, base_model, output_dir, lora_rank):
            assert read_records(records_path)
            output_dir.mkdir(parents=True)
            (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            return output_dir

        with patch("training.lora_trainer.train_lora", side_effect=fake_train) as train:
            result = _run(_cfg(train_model=True, lora_rank=32, output_format=OutputFormat.SHAREGPT), source)
        args = train.call_args.args
        assert args[0] == result.run.records and args[2] == result.run.adapter_dir and args[3] == 32
        assert result.adapter_zip is not None and result.adapter_zip.is_file()

    def test_publish_uploads_the_formatted_dataset(self, source):
        cfg = _cfg(publish_dataset=True, hf_repo="user/my-dataset", hf_token="hf_x")
        with patch("publish.hf_publisher.publish_dataset") as publish:
            result = _run(cfg, source)
        publish.assert_called_once_with(str(result.dataset_path), "user/my-dataset", "hf_x")
        assert result.published_repo == "user/my-dataset"

    def test_optional_stages_off_by_default(self, source):
        with patch("training.lora_trainer.train_lora") as train, \
             patch("publish.hf_publisher.publish_dataset") as publish:
            result = _run(_cfg(), source)
        train.assert_not_called()
        publish.assert_not_called()
        assert result.adapter_zip is None
