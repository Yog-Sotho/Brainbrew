"""
tests/test_orchestrator.py

Tests for orchestrator.py — run_distillation().

All heavy dependencies are mocked. Tests verify orchestration logic,
not the behaviour of distilabel/OpenAI internally.

FIX C-04: patches now target the correct attribute names in the live
orchestrator.py module:
  - No more patching RenameColumns / FilterRows (don't exist)
  - train_lora and publish_dataset are lazy-imported inside functions,
    so we patch them at their source modules.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from config import DistillationConfig, QualityMode


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_distiset(tmp_path: Path) -> MagicMock:
    record = {"instruction": "Q?", "output": "Long enough answer to pass the length filter easily here."}

    def fake_to_json(path: str) -> None:
        Path(path).write_text(json.dumps(record) + "\n", encoding="utf-8")

    mock_split = MagicMock()
    mock_split.to_json.side_effect = fake_to_json
    mock_default = MagicMock()
    mock_default.__getitem__ = MagicMock(return_value=mock_split)
    distiset = MagicMock()
    distiset.__getitem__ = MagicMock(return_value=mock_default)
    return distiset


def _run_with_mocks(cfg, source_file, progress_callback=None, tmp_path=None):
    """Run orchestrator with all heavy deps mocked.

    FIX C-04: patches target correct module paths:
      - orchestrator.Pipeline, orchestrator.OpenAILLM, orchestrator.vLLM etc.
        are module-level imports and can be patched on orchestrator directly.
      - train_lora / publish_dataset are lazy-imported from their source modules,
        so they are patched at training.lora_trainer / publish.hf_publisher.
    """
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.return_value = _make_distiset(tmp_path or Path("."))
    # Pipeline is used as a context manager: with Pipeline(...) as pipeline:
    mock_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)
    mock_pipeline_cls.__enter__ = MagicMock(return_value=mock_pipeline_instance)
    mock_pipeline_cls.__exit__ = MagicMock(return_value=False)

    output_dir = tmp_path / "output" if tmp_path else Path(".") / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    with patch("orchestrator.Pipeline", return_value=mock_pipeline_instance) as MockPipeline, \
         patch("orchestrator.OpenAILLM", return_value=MagicMock()) as MockOpenAI, \
         patch("orchestrator.vLLM", return_value=MagicMock()) as MockVLLM, \
         patch("orchestrator.EvolInstruct", return_value=MagicMock()) as MockEvol, \
         patch("orchestrator.TextGeneration", return_value=MagicMock()), \
         patch("orchestrator.KeepColumns", return_value=MagicMock()), \
         patch("orchestrator.LoadDataFromDicts", return_value=MagicMock()), \
         patch("orchestrator.FilterAndRenameOutputs", return_value=MagicMock()), \
         patch("orchestrator.export_dataset") as mock_export, \
         patch("training.lora_trainer.train_lora") as mock_train, \
         patch("publish.hf_publisher.publish_dataset") as mock_publish:

        # Make export_dataset return a count
        mock_export.return_value = 10

        from orchestrator import run_distillation
        result = run_distillation(
            cfg, source_file, progress_callback, output_dir=output_dir
        )

        return result, {
            "pipeline": mock_pipeline_instance,
            "MockPipeline": MockPipeline,
            "MockOpenAI": MockOpenAI,
            "MockVLLM": MockVLLM,
            "MockEvol": MockEvol,
            "train_lora": mock_train,
            "publish_dataset": mock_publish,
            "export_dataset": mock_export,
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

def test_run_distillation_importable():
    from orchestrator import run_distillation
    assert callable(run_distillation)


# ── Return value ──────────────────────────────────────────────────────────────

class TestReturnValue:

    def test_returns_path(self, base_config, source_file, tmp_path):
        result, _ = _run_with_mocks(base_config, source_file, tmp_path=tmp_path)
        assert isinstance(result, Path)

    def test_returns_dataset_filename(self, base_config, source_file, tmp_path):
        result, _ = _run_with_mocks(base_config, source_file, tmp_path=tmp_path)
        assert "dataset.jsonl" in result.name


# ── Security: api_key never logged ───────────────────────────────────────────

class TestApiKeyNeverLogged:

    def test_api_key_not_in_any_log_call(self, source_file, tmp_path):
        cfg = DistillationConfig(
            teacher_model="gpt-4o", use_vllm=False,
            quality_mode=QualityMode.FAST, dataset_size=100,
            api_key="sk-this-must-never-appear-in-logs",
        )
        logged_args = []
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("orchestrator.Pipeline", return_value=MagicMock(
                run=MagicMock(return_value=_make_distiset(tmp_path)))), \
             patch("orchestrator.OpenAILLM", return_value=MagicMock()), \
             patch("orchestrator.vLLM", return_value=MagicMock()), \
             patch("orchestrator.EvolInstruct", return_value=MagicMock()), \
             patch("orchestrator.TextGeneration", return_value=MagicMock()), \
             patch("orchestrator.KeepColumns", return_value=MagicMock()), \
             patch("orchestrator.LoadDataFromDicts", return_value=MagicMock()), \
             patch("orchestrator.FilterAndRenameOutputs", return_value=MagicMock()), \
             patch("orchestrator.export_dataset", return_value=10), \
             patch("training.lora_trainer.train_lora"), \
             patch("publish.hf_publisher.publish_dataset"), \
             patch("orchestrator.logger") as mock_logger:

            mock_logger.info.side_effect = lambda msg, **kw: logged_args.append((msg, kw))

            from orchestrator import run_distillation
            run_distillation(cfg, source_file, output_dir=output_dir)

        for msg, kwargs in logged_args:
            assert "sk-this-must-never-appear-in-logs" not in str(msg) + str(kwargs)


# ── Progress callbacks ───────────────────────────────────────────────────────

class TestRealProgressCallbacks:

    def test_callback_called_multiple_times(self, base_config, source_file, tmp_path):
        calls = []
        _run_with_mocks(base_config, source_file,
                        progress_callback=lambda p: calls.append(p), tmp_path=tmp_path)
        assert len(calls) > 1, (
            f"Progress callback must be called multiple times (got {len(calls)})."
        )

    def test_callback_called_with_100_on_success(self, base_config, source_file, tmp_path):
        calls = []
        _run_with_mocks(base_config, source_file,
                        progress_callback=lambda p: calls.append(p), tmp_path=tmp_path)
        assert 100 in calls

    def test_progress_values_are_increasing(self, base_config, source_file, tmp_path):
        calls = []
        _run_with_mocks(base_config, source_file,
                        progress_callback=lambda p: calls.append(p), tmp_path=tmp_path)
        for i in range(1, len(calls)):
            assert calls[i] >= calls[i - 1], (
                f"Progress went backwards: {calls[i - 1]} -> {calls[i]}"
            )

    def test_progress_values_between_0_and_100(self, base_config, source_file, tmp_path):
        calls = []
        _run_with_mocks(base_config, source_file,
                        progress_callback=lambda p: calls.append(p), tmp_path=tmp_path)
        for p in calls:
            assert 0 <= p <= 100, f"Progress value out of range: {p}"

    def test_none_callback_does_not_raise(self, base_config, source_file, tmp_path):
        _run_with_mocks(base_config, source_file, progress_callback=None, tmp_path=tmp_path)


# ── LLM backend selection ────────────────────────────────────────────────────

class TestLLMBackendSelection:

    def test_openai_llm_used_when_use_vllm_false(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["MockOpenAI"].assert_called()
        mocks["MockVLLM"].assert_not_called()

    def test_vllm_used_when_use_vllm_true(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                                  use_vllm=True, quality_mode=QualityMode.FAST, dataset_size=100)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["MockVLLM"].assert_called()
        mocks["MockOpenAI"].assert_not_called()

    def test_first_model_from_comma_separated_list(self, source_file, tmp_path):
        """Enhancement 4: multi-model splits prompts across models."""
        cfg = DistillationConfig(teacher_model="gpt-4o,gpt-3.5-turbo", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        # Both models should be created
        assert mocks["MockOpenAI"].call_count == 2


# ── Conditional steps ────────────────────────────────────────────────────────

class TestConditionalSteps:

    def test_train_lora_called_when_train_model_true(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100,
                                  train_model=True)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["train_lora"].assert_called_once()

    def test_train_lora_not_called_when_train_model_false(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100,
                                  train_model=False)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["train_lora"].assert_not_called()

    def test_publish_called_when_enabled_with_repo(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100,
                                  publish_dataset=True, hf_repo="user/repo")
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["publish_dataset"].assert_called_once()

    def test_publish_not_called_when_disabled(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100,
                                  publish_dataset=False, hf_repo="user/repo")
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["publish_dataset"].assert_not_called()

    def test_publish_not_called_when_hf_repo_none(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100,
                                  publish_dataset=True, hf_repo=None)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["publish_dataset"].assert_not_called()


# ── Pipeline execution ───────────────────────────────────────────────────────

class TestPipelineExecution:

    def test_pipeline_run_called(self, base_config, source_file, tmp_path):
        _, mocks = _run_with_mocks(base_config, source_file, tmp_path=tmp_path)
        mocks["pipeline"].run.assert_called()

    def test_export_called_once(self, base_config, source_file, tmp_path):
        _, mocks = _run_with_mocks(base_config, source_file, tmp_path=tmp_path)
        mocks["export_dataset"].assert_called_once()


# ── Error conditions ─────────────────────────────────────────────────────────

class TestErrorConditions:

    def test_missing_source_file_raises(self, base_config, tmp_path):
        from orchestrator import run_distillation
        with pytest.raises(Exception):
            run_distillation(base_config, Path("/nonexistent/path/source.txt"),
                             output_dir=tmp_path)

    def test_empty_source_file_raises(self, base_config, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(Exception):
            _run_with_mocks(base_config, empty, tmp_path=tmp_path)

    def test_large_source_file_raises_valueerror(self, base_config, tmp_path):
        from orchestrator import MAX_SOURCE_BYTES
        huge = tmp_path / "huge.txt"
        huge.write_bytes(b"x" * (MAX_SOURCE_BYTES + 1))
        with pytest.raises(ValueError, match="exceeds"):
            _run_with_mocks(base_config, huge, tmp_path=tmp_path)


# ── Quality mode -> num_evolutions ───────────────────────────────────────────

@pytest.mark.parametrize("mode,expected_evolutions", [
    (QualityMode.FAST, 1),
    (QualityMode.BALANCED, 2),
    (QualityMode.RESEARCH, 3),
])
def test_quality_mode_controls_num_evolutions(mode, expected_evolutions, source_file, tmp_path):
    cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                              quality_mode=mode, dataset_size=100)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("orchestrator.Pipeline", return_value=MagicMock(
            run=MagicMock(return_value=_make_distiset(tmp_path)))), \
         patch("orchestrator.OpenAILLM", return_value=MagicMock()), \
         patch("orchestrator.vLLM", return_value=MagicMock()), \
         patch("orchestrator.EvolInstruct", return_value=MagicMock()) as MockEvol, \
         patch("orchestrator.TextGeneration", return_value=MagicMock()), \
         patch("orchestrator.KeepColumns", return_value=MagicMock()), \
         patch("orchestrator.LoadDataFromDicts", return_value=MagicMock()), \
         patch("orchestrator.FilterAndRenameOutputs", return_value=MagicMock()), \
         patch("orchestrator.export_dataset", return_value=10), \
         patch("training.lora_trainer.train_lora"), \
         patch("publish.hf_publisher.publish_dataset"):

        from orchestrator import run_distillation
        run_distillation(cfg, source_file, output_dir=output_dir)

    _, kwargs = MockEvol.call_args
    assert kwargs.get("num_evolutions") == expected_evolutions


# ── Enhancement 10: quality scoring ──────────────────────────────────────────

class TestQualityScoring:

    def test_score_dataset_importable(self):
        from orchestrator import score_dataset
        assert callable(score_dataset)

    def test_empty_file_returns_disaster(self, tmp_path):
        from orchestrator import score_dataset
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        result = score_dataset(p)
        assert result["grade"] == "DISASTER"

    def test_nonexistent_file_returns_disaster(self, tmp_path):
        from orchestrator import score_dataset
        result = score_dataset(tmp_path / "nonexistent.jsonl")
        assert result["grade"] == "DISASTER"

    def test_good_dataset_scores_above_bad(self, tmp_path):
        from orchestrator import score_dataset
        p = tmp_path / "good.jsonl"
        records = [
            {"instruction": f"Unique question number {i}?", "output": "A " * 100}
            for i in range(200)
        ]
        p.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )
        result = score_dataset(p)
        assert result["grade"] in ("SUPER", "GOOD", "NORMAL")
        assert result["record_count"] == 200
