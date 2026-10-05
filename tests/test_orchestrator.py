"""
tests/test_orchestrator.py

Contract tests: the orchestrator builds and runs the *real* distilabel DAG
(EvolInstruct -> TextGeneration -> ToCanonical -> KeepColumns) on CPU, with
only the LLM replaced by the deterministic offline FakeLLM. Nothing in
distilabel is mocked, so these tests fail if the pipeline does not validate,
if pairs are misaligned, or if a stage drops every record.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import orchestrator
from config import DistillationConfig, OutputFormat, QualityMode
from pipeline.records import Record, read_records
from pipeline.runs import open_run
from tests.fake_llm import ANSWER_PREFIX, FakeLLM, answer, evolve

TOPICS = [
    "Photosynthesis lets plants turn sunlight, water and carbon dioxide into glucose and oxygen.",
    "The French Revolution began in 1789 and ended the absolute monarchy of Louis XVI.",
    "TCP guarantees ordered delivery of bytes using sequence numbers and acknowledgements.",
    "Plate tectonics explains earthquakes, volcanoes and the slow drift of the continents.",
    "Compound interest grows savings because interest is earned on previous interest.",
    "Vaccines train the immune system by exposing it to a harmless form of a pathogen.",
]


@pytest.fixture()
def varied_source(tmp_path: Path) -> Path:
    """A document whose chunks are all different, so dedup keeps them."""
    paragraphs = [(t + " ") * 6 for t in TOPICS]
    p = tmp_path / "source.txt"
    p.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return p


def _cfg(**overrides) -> DistillationConfig:
    base = {
        "teacher_model": "gpt-4o-mini",
        "use_vllm": False,
        "api_key": "test-key",
        "quality_mode": QualityMode.FAST,
        "dataset_size": 100,
        "batch_size": 4,
    }
    return DistillationConfig(**{**base, **overrides})


def _run(cfg: DistillationConfig, source: Path, **kwargs) -> orchestrator.RunResult:
    with patch.object(orchestrator, "_create_llm", lambda name, cfg: FakeLLM()):
        return orchestrator.run_distillation(cfg, source, **kwargs)


def _evolve_n(seed: str, n: int) -> str:
    for _ in range(n):
        seed = evolve(seed)
    return seed


# ── Real DAG contract ────────────────────────────────────────────────────────

class TestRealPipelineContract:

    @pytest.mark.parametrize("mode,evolutions", [
        (QualityMode.FAST, 1),
        (QualityMode.RESEARCH, 3),
    ])
    def test_pairs_are_aligned(self, mode, evolutions, varied_source):
        """The kept instruction is the evolved one, and the output answers it."""
        result = _run(_cfg(quality_mode=mode, enable_dedup=False), varied_source)
        raw = read_records(result.run.raw)
        assert raw, "pipeline produced no records"
        for rec in raw:
            assert rec.instruction == _evolve_n(rec.meta["seed"], evolutions)
            assert rec.output == answer(rec.instruction)
            assert rec.meta["model"] == "fake-llm"

    def test_seed_prompts_wrap_document_chunks(self, varied_source):
        result = _run(_cfg(enable_dedup=False), varied_source)
        seeds = [r.meta["seed"] for r in read_records(result.run.raw)]
        assert all(s.startswith("Explain the following concept from the document") for s in seeds)
        assert any("Photosynthesis" in s for s in seeds)

    def test_dataset_size_caps_prompts(self, tmp_path):
        source = tmp_path / "long.txt"
        source.write_text("\n\n".join(f"Fact {i}: " + "word " * 180 for i in range(150)), encoding="utf-8")
        result = _run(_cfg(dataset_size=100, enable_dedup=False), source)
        assert result.run.read_manifest()["counts"]["chunks"] == 100


_LAYOUT_KEY = {
    OutputFormat.ALPACA: "instruction",
    OutputFormat.SHAREGPT: "conversations",
    OutputFormat.CHATML: "messages",
    OutputFormat.OPENAI: "messages",
}


class TestFormatMatrix:
    """Generate -> dedup -> sanitize -> score -> export, for every format."""

    def test_every_format_keeps_records_and_scores_the_same(self, varied_source):
        reports = {}
        for fmt in OutputFormat:
            result = _run(_cfg(output_format=fmt, sanitize_dataset=True), varied_source)
            manifest = result.run.read_manifest()
            assert manifest["counts"]["after_sanitize"] > 0, fmt
            assert manifest["counts"]["exported"] == manifest["counts"]["after_sanitize"]
            lines = result.dataset_path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == result.record_count
            assert ANSWER_PREFIX in lines[0]  # the answer made it into the export
            assert result.dataset_path.name == f"dataset.{fmt.value}.jsonl"
            assert _LAYOUT_KEY[fmt] in json.loads(lines[0])
            reports[fmt] = (result.record_count, result.quality)
        assert len(set(json.dumps(r, sort_keys=True) for r in reports.values())) == 1


# ── Run directory + manifest ─────────────────────────────────────────────────

class TestRunDirectory:

    def test_files_and_manifest(self, varied_source, _isolated_runs_dir):
        result = _run(_cfg(sanitize_dataset=True), varied_source)
        run = open_run(result.run.run_id)
        assert run.root.parent == _isolated_runs_dir
        for path in (run.source, run.raw, run.records, result.dataset_path):
            assert path.is_file()
        manifest = run.read_manifest()
        assert manifest["status"] == "succeeded"
        assert manifest["config"]["teacher_model"] == "gpt-4o-mini"
        assert manifest["quality"]["grade"] == result.quality["grade"]
        assert manifest["sanitizer"]["kept"] == manifest["counts"]["after_sanitize"]
        assert manifest["dataset_file"] == result.dataset_path.name

    def test_secrets_never_written_to_run_dir(self, varied_source):
        secret = "sk-this-must-never-be-written-anywhere"
        result = _run(_cfg(api_key=secret), varied_source)
        for path in result.run.root.rglob("*"):
            if path.is_file():
                assert secret not in path.read_text(encoding="utf-8", errors="ignore"), path

    def test_serialized_pipeline_has_no_api_key(self, tmp_path):
        """distilabel writes the pipeline (with its LLM) into the run's cache folder."""
        from pipeline.generation import build_pipeline

        secret = "sk-real-openai-llm-secret-0123456789"
        llm = orchestrator._create_llm("gpt-4o-mini", _cfg(api_key=secret))
        pipeline = build_pipeline(["Explain photosynthesis."], llm, 1, 4, "secret-check", tmp_path)
        out = tmp_path / "pipeline.yaml"
        pipeline.save(str(out), format="yaml")
        assert out.is_file() and "OpenAILLM" in out.read_text(encoding="utf-8")
        assert secret not in out.read_text(encoding="utf-8")

    def test_failed_run_is_marked_in_manifest(self, varied_source, _isolated_runs_dir):
        with patch.object(orchestrator, "generate_rows", side_effect=RuntimeError("teacher down")), \
             pytest.raises(RuntimeError, match="teacher down"):
            _run(_cfg(), varied_source)
        (run_root,) = _isolated_runs_dir.iterdir()
        manifest = open_run(run_root.name).read_manifest()
        assert manifest["status"] == "failed"
        assert "teacher down" in manifest["error"]

    def test_existing_run_dir_is_used(self, varied_source):
        run = orchestrator.create_run()
        result = _run(_cfg(), varied_source, run=run)
        assert result.run.root == run.root


# ── Failure modes that used to be silent ────────────────────────────────────

class TestFailures:

    def test_sanitizer_removing_everything_raises(self, varied_source):
        with patch("pipeline.sanitizer.check_quality", return_value="rejected"), \
             pytest.raises(RuntimeError, match="Sanitizing removed all"):
            _run(_cfg(sanitize_dataset=True), varied_source)

    def test_no_usable_records_raises(self, varied_source):
        with patch.object(orchestrator, "generate_rows", return_value=[]), \
             pytest.raises(RuntimeError, match="no usable records"):
            _run(_cfg(), varied_source)

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


# ── LLM construction ─────────────────────────────────────────────────────────

class TestCreateLLM:

    def test_openai_gets_generation_kwargs(self):
        llm = orchestrator._create_llm("gpt-4o-mini", _cfg(temperature=0.3, max_new_tokens=512))
        assert type(llm).__name__ == "OpenAILLM"
        assert llm.generation_kwargs == {"max_new_tokens": 512, "temperature": 0.3}

    def test_vllm_gets_generation_kwargs(self):
        llm = orchestrator._create_llm("some/model", _cfg(use_vllm=True, max_new_tokens=256))
        assert type(llm).__name__ == "vLLM"
        assert llm.generation_kwargs["max_new_tokens"] == 256


class TestToCanonical:

    def test_filters_failed_and_short_rows(self):
        from pipeline.steps import ToCanonical

        step = ToCanonical(min_length=10)
        rows = [
            {"instruction": "seed", "evolved_instruction": "evolved", "generation": "long enough answer"},
            {"instruction": "seed", "evolved_instruction": None, "generation": "long enough answer"},
            {"instruction": "seed", "evolved_instruction": "evolved", "generation": "short"},
            {"instruction": "seed", "evolved_instruction": "evolved", "generation": None},
        ]
        (out,) = list(step.process(rows))
        assert out == [{**rows[0], "seed": "seed", "instruction": "evolved", "output": "long enough answer"}]


def test_split_prompts_gives_remainder_to_last_model():
    parts = orchestrator._split_prompts([str(i) for i in range(7)], 3)
    assert [len(p) for p in parts] == [2, 2, 3]
    assert sum(parts, []) == [str(i) for i in range(7)]


def test_multi_model_runs_one_pipeline_per_model(varied_source):
    names = []
    real_generate = orchestrator.generate_rows

    def spy(*args, **kwargs):
        names.append(args[4])
        return real_generate(*args, **kwargs)

    with patch.object(orchestrator, "generate_rows", side_effect=spy):
        result = _run(_cfg(teacher_model="model-a,model-b", enable_dedup=False), varied_source)
    assert names == [f"brainbrew-{result.run.run_id}-m0", f"brainbrew-{result.run.run_id}-m1"]
    assert result.run.read_manifest()["counts"]["generated"] == result.run.read_manifest()["counts"]["chunks"]


# ── Progress, logging, optional stages ───────────────────────────────────────

class TestProgressAndLogging:

    def test_progress_increases_to_100(self, varied_source):
        values: list[int] = []
        _run(_cfg(), varied_source, progress_callback=values.append)
        assert values == sorted(values)
        assert values[-1] == 100
        assert all(0 <= v <= 100 for v in values)

    def test_api_key_not_logged(self, varied_source):
        secret = "sk-this-must-never-appear-in-logs"
        logged: list[str] = []
        with patch.object(orchestrator, "logger") as log:
            log.info.side_effect = lambda msg, **kw: logged.append(f"{msg} {kw}")
            _run(_cfg(api_key=secret), varied_source)
        assert logged and not any(secret in line for line in logged)

    def test_root_logging_untouched_by_pipeline(self, varied_source):
        """distilabel replaces root handlers; that must stay inside the child process."""
        import logging

        root = logging.getLogger()
        before = list(root.handlers)
        _run(_cfg(), varied_source)
        assert root.handlers == before


class TestOptionalStages:

    def test_training_uses_canonical_records_and_zips_adapter(self, varied_source):
        def fake_train(records_path, base_model, output_dir, lora_rank):
            assert read_records(records_path)  # canonical, whatever the export format
            output_dir.mkdir(parents=True)
            (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            return output_dir

        with patch("training.lora_trainer.train_lora", side_effect=fake_train) as train:
            result = _run(_cfg(train_model=True, lora_rank=32, output_format=OutputFormat.SHAREGPT), varied_source)
        args = train.call_args.args
        assert args[0] == result.run.records and args[2] == result.run.adapter_dir and args[3] == 32
        assert result.adapter_zip is not None and result.adapter_zip.is_file()
        assert result.run.read_manifest()["adapter_file"] == "adapter.zip"

    def test_training_not_called_by_default(self, varied_source):
        with patch("training.lora_trainer.train_lora") as train:
            result = _run(_cfg(), varied_source)
        train.assert_not_called()
        assert result.adapter_zip is None

    def test_publish_uploads_the_formatted_dataset(self, varied_source):
        cfg = _cfg(publish_dataset=True, hf_repo="user/my-dataset", hf_token="hf_x")
        with patch("publish.hf_publisher.publish_dataset") as publish:
            result = _run(cfg, varied_source)
        publish.assert_called_once_with(str(result.dataset_path), "user/my-dataset", "hf_x")
        assert result.published_repo == "user/my-dataset"

    def test_publish_not_called_by_default(self, varied_source):
        with patch("publish.hf_publisher.publish_dataset") as publish:
            _run(_cfg(), varied_source)
        publish.assert_not_called()


def test_records_file_matches_exported_count(varied_source):
    result = _run(_cfg(sanitize_dataset=True), varied_source)
    assert len(read_records(result.run.records)) == result.record_count
    assert all(isinstance(r, Record) for r in read_records(result.run.records))


# ── Runs off the main thread (Streamlit script threads) ─────────────────────

def test_runs_from_a_worker_thread(varied_source):
    """Pipeline.run installs a SIGINT handler, which only works on a main thread."""
    import threading

    outcome: dict[str, object] = {}

    def target() -> None:
        try:
            outcome["result"] = _run(_cfg(), varied_source)
        except BaseException as exc:  # pragma: no cover - reported below
            outcome["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=300)
    assert "error" not in outcome, outcome.get("error")
    assert outcome["result"].record_count > 0


def _explode(message: str) -> None:
    raise ValueError(message)


class _FailsInChild:
    """Unpickling this raises, i.e. the failure happens inside the child process."""

    def __reduce__(self):
        return (_explode, ("boom from the child",))


def test_child_process_error_is_reported(tmp_path):
    from pipeline.generation import generate_rows

    with pytest.raises(RuntimeError, match="Generation failed: ValueError: boom from the child"):
        generate_rows(["prompt"], _FailsInChild(), 1, 4, "fails", tmp_path)
