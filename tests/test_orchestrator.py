"""
tests/test_orchestrator.py

Tests for orchestrator.py — run_distillation().

All heavy dependencies are mocked. Tests verify orchestration logic,
not the behaviour of distilabel/OpenAI internally.

Covers (original + new):
  - Import smoke test (no NameError — regression guard for duplicate function)
  - Returns a Path pointing to alpaca_dataset.jsonl
  - api_key NEVER appears in any log call (S-01 security regression)
  - FIX: progress_callback called at multiple real stages, not just 100
  - progress_callback called with 100 on success
  - progress_callback not required (None is safe)
  - OpenAILLM instantiated when use_vllm=False
  - vLLM instantiated when use_vllm=True
  - First teacher_model used when comma-separated list provided
  - train_lora called when train_model=True
  - train_lora NOT called when train_model=False
  - publish_dataset called when publish_dataset=True and hf_repo set
  - publish_dataset NOT called when publish_dataset=False
  - publish_dataset NOT called when hf_repo is None
  - pipeline.run() called exactly once
  - export_alpaca called once
  - FIX: large source file raises ValueError
  - Empty source file raises exception
  - QualityMode.FAST uses 1 evolution, BALANCED uses 2, RESEARCH uses 3
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import DistillationConfig, QualityMode


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_distiset(tmp_path: Path) -> MagicMock:
    record = {"instruction": "Q?", "output": "Long enough answer to pass the length filter."}

    def fake_to_json(path: str) -> None:
        Path(path).write_text(json.dumps(record) + "\n", encoding="utf-8")

    mock_split = MagicMock()
    mock_split.to_json.side_effect = fake_to_json
    distiset = MagicMock()
    distiset.__getitem__ = MagicMock(return_value=mock_split)
    return distiset


def _run_with_mocks(cfg, source_file, progress_callback=None, tmp_path=None):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = _make_distiset(tmp_path or Path("."))

    with patch("orchestrator.Pipeline", return_value=mock_pipeline), \
         patch("orchestrator.OpenAILLM", return_value=MagicMock()) as MockOpenAI, \
         patch("orchestrator.vLLM", return_value=MagicMock()) as MockVLLM, \
         patch("orchestrator.EvolInstruct", return_value=MagicMock()), \
         patch("orchestrator.TextGeneration", return_value=MagicMock()), \
         patch("orchestrator.RenameColumns", return_value=MagicMock()), \
         patch("orchestrator.KeepColumns", return_value=MagicMock()), \
         patch("orchestrator.FilterRows", return_value=MagicMock()), \
         patch("orchestrator.LoadDataFromDicts", return_value=MagicMock()), \
         patch("orchestrator.train_lora") as mock_train, \
         patch("orchestrator.publish_dataset") as mock_publish, \
         patch("orchestrator.export_alpaca") as mock_export:

        from orchestrator import run_distillation
        result = run_distillation(cfg, source_file, progress_callback)

        return result, {
            "pipeline": mock_pipeline,
            "MockOpenAI": MockOpenAI,
            "MockVLLM": MockVLLM,
            "train_lora": mock_train,
            "publish_dataset": mock_publish,
            "export_alpaca": mock_export,
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

    def test_returns_alpaca_dataset_filename(self, base_config, source_file, tmp_path):
        result, _ = _run_with_mocks(base_config, source_file, tmp_path=tmp_path)
        assert result.name == "alpaca_dataset.jsonl"


# ── Security: api_key never logged ───────────────────────────────────────────

class TestApiKeyNeverLogged:

    def test_api_key_not_in_any_log_call(self, source_file, tmp_path):
        cfg = DistillationConfig(
            teacher_model="gpt-4o", use_vllm=False,
            quality_mode=QualityMode.FAST, dataset_size=100,
            api_key="sk-this-must-never-appear-in-logs",
        )
        logged_args = []

        with patch("orchestrator.Pipeline", return_value=MagicMock(
                run=MagicMock(return_value=_make_distiset(tmp_path)))), \
             patch("orchestrator.OpenAILLM", return_value=MagicMock()), \
             patch("orchestrator.vLLM", return_value=MagicMock()), \
             patch("orchestrator.EvolInstruct", return_value=MagicMock()), \
             patch("orchestrator.TextGeneration", return_value=MagicMock()), \
             patch("orchestrator.RenameColumns", return_value=MagicMock()), \
             patch("orchestrator.KeepColumns", return_value=MagicMock()), \
             patch("orchestrator.FilterRows", return_value=MagicMock()), \
             patch("orchestrator.LoadDataFromDicts", return_value=MagicMock()), \
             patch("orchestrator.train_lora"), \
             patch("orchestrator.publish_dataset"), \
             patch("orchestrator.export_alpaca"), \
             patch("orchestrator.logger") as mock_logger:

            mock_logger.info.side_effect = lambda msg, **kw: logged_args.append((msg, kw))

            from orchestrator import run_distillation
            run_distillation(cfg, source_file)

        for msg, kwargs in logged_args:
            assert "sk-this-must-never-appear-in-logs" not in str(msg) + str(kwargs)


# ── FIX: Real progress callbacks ─────────────────────────────────────────────

class TestRealProgressCallbacks:

    def test_callback_called_multiple_times(self, base_config, source_file, tmp_path):
        """Progress bar must be called at real intermediate stages, not just at 100."""
        calls = []
        _run_with_mocks(base_config, source_file, progress_callback=lambda p: calls.append(p),
                        tmp_path=tmp_path)
        assert len(calls) > 1, (
            f"Progress callback must be called multiple times (got {len(calls)}). "
            "It was called only at the end — the 'fake progress bar' bug."
        )

    def test_callback_called_with_100_on_success(self, base_config, source_file, tmp_path):
        calls = []
        _run_with_mocks(base_config, source_file, progress_callback=lambda p: calls.append(p),
                        tmp_path=tmp_path)
        assert 100 in calls

    def test_progress_values_are_increasing(self, base_config, source_file, tmp_path):
        """Progress values must never go backwards."""
        calls = []
        _run_with_mocks(base_config, source_file, progress_callback=lambda p: calls.append(p),
                        tmp_path=tmp_path)
        for i in range(1, len(calls)):
            assert calls[i] >= calls[i - 1], (
                f"Progress went backwards: {calls[i - 1]} → {calls[i]}"
            )

    def test_progress_values_between_0_and_100(self, base_config, source_file, tmp_path):
        calls = []
        _run_with_mocks(base_config, source_file, progress_callback=lambda p: calls.append(p),
                        tmp_path=tmp_path)
        for p in calls:
            assert 0 <= p <= 100, f"Progress value out of range: {p}"

    def test_none_callback_does_not_raise(self, base_config, source_file, tmp_path):
        _run_with_mocks(base_config, source_file, progress_callback=None, tmp_path=tmp_path)


# ── LLM backend selection ─────────────────────────────────────────────────────

class TestLLMBackendSelection:

    def test_openai_llm_used_when_use_vllm_false(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["MockOpenAI"].assert_called_once()
        mocks["MockVLLM"].assert_not_called()

    def test_vllm_used_when_use_vllm_true(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                                  use_vllm=True, quality_mode=QualityMode.FAST, dataset_size=100)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["MockVLLM"].assert_called_once()
        mocks["MockOpenAI"].assert_not_called()

    def test_first_model_from_comma_separated_list(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o,gpt-3.5-turbo", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        _, kwargs = mocks["MockOpenAI"].call_args
        assert kwargs.get("model") == "gpt-4o"


# ── Conditional steps ─────────────────────────────────────────────────────────

class TestConditionalSteps:

    def test_train_lora_called_when_train_model_true(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100, train_model=True)
        _, mocks = _run_with_mocks(cfg, source_file, tmp_path=tmp_path)
        mocks["train_lora"].assert_called_once()

    def test_train_lora_not_called_when_train_model_false(self, source_file, tmp_path):
        cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                                  quality_mode=QualityMode.FAST, dataset_size=100, train_model=False)
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


# ── Pipeline execution ────────────────────────────────────────────────────────

class TestPipelineExecution:

    def test_pipeline_run_called_exactly_once(self, base_config, source_file, tmp_path):
        _, mocks = _run_with_mocks(base_config, source_file, tmp_path=tmp_path)
        mocks["pipeline"].run.assert_called_once()

    def test_export_alpaca_called_once(self, base_config, source_file, tmp_path):
        _, mocks = _run_with_mocks(base_config, source_file, tmp_path=tmp_path)
        mocks["export_alpaca"].assert_called_once()


# ── Error conditions ──────────────────────────────────────────────────────────

class TestErrorConditions:

    def test_missing_source_file_raises(self, base_config):
        from orchestrator import run_distillation
        with pytest.raises(Exception):
            run_distillation(base_config, Path("/nonexistent/path/source.txt"))

    def test_empty_source_file_raises(self, base_config, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(Exception):
            _run_with_mocks(base_config, empty, tmp_path=tmp_path)

    def test_large_source_file_raises_valueerror(self, base_config, tmp_path):
        """FIX: files over 100 MB must be rejected with a clear error, not a silent slowdown."""
        from orchestrator import MAX_SOURCE_BYTES
        huge = tmp_path / "huge.txt"
        # write a file just over the limit
        huge.write_bytes(b"x" * (MAX_SOURCE_BYTES + 1))
        with pytest.raises(ValueError, match="exceeds"):
            _run_with_mocks(base_config, huge, tmp_path=tmp_path)


# ── Quality mode → num_evolutions ────────────────────────────────────────────

@pytest.mark.parametrize("mode,expected_evolutions", [
    (QualityMode.FAST, 1),
    (QualityMode.BALANCED, 2),
    (QualityMode.RESEARCH, 3),
])
def test_quality_mode_controls_num_evolutions(mode, expected_evolutions, source_file, tmp_path):
    cfg = DistillationConfig(teacher_model="gpt-4o", use_vllm=False,
                              quality_mode=mode, dataset_size=100)

    with patch("orchestrator.Pipeline", return_value=MagicMock(
            run=MagicMock(return_value=_make_distiset(tmp_path)))), \
         patch("orchestrator.OpenAILLM", return_value=MagicMock()), \
         patch("orchestrator.vLLM", return_value=MagicMock()), \
         patch("orchestrator.EvolInstruct", return_value=MagicMock()) as MockEvol, \
         patch("orchestrator.TextGeneration", return_value=MagicMock()), \
         patch("orchestrator.RenameColumns", return_value=MagicMock()), \
         patch("orchestrator.KeepColumns", return_value=MagicMock()), \
         patch("orchestrator.FilterRows", return_value=MagicMock()), \
         patch("orchestrator.LoadDataFromDicts", return_value=MagicMock()), \
         patch("orchestrator.train_lora"), \
         patch("orchestrator.publish_dataset"), \
         patch("orchestrator.export_alpaca"):

        from orchestrator import run_distillation
        run_distillation(cfg, source_file)

    _, kwargs = MockEvol.call_args
    assert kwargs.get("num_evolutions") == expected_evolutions
