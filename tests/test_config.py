"""
tests/test_config.py

Tests for config.py — DistillationConfig, QualityMode, OutputFormat,
QUALITY_MODE_LABELS, OUTPUT_FORMAT_LABELS, safe_dict(), __repr__.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from config import (
    DistillationConfig,
    OutputFormat,
    QualityMode,
    QUALITY_MODE_LABELS,
    OUTPUT_FORMAT_LABELS,
)


# ── QualityMode ───────────────────────────────────────────────────────────────

class TestQualityMode:

    def test_all_three_values_exist(self):
        assert QualityMode.FAST.value == "fast"
        assert QualityMode.BALANCED.value == "balanced"
        assert QualityMode.RESEARCH.value == "research"

    def test_string_coercion(self):
        assert QualityMode("balanced") is QualityMode.BALANCED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            QualityMode("turbo")


# ── OutputFormat ──────────────────────────────────────────────────────────────

class TestOutputFormat:

    def test_all_four_values_exist(self):
        assert OutputFormat.ALPACA.value == "alpaca"
        assert OutputFormat.SHAREGPT.value == "sharegpt"
        assert OutputFormat.CHATML.value == "chatml"
        assert OutputFormat.OPENAI.value == "openai"

    def test_string_coercion(self):
        assert OutputFormat("sharegpt") is OutputFormat.SHAREGPT

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            OutputFormat("invalid_format")


# ── QUALITY_MODE_LABELS ──────────────────────────────────────────────────────

class TestQualityModeLabels:

    def test_all_three_modes_have_labels(self):
        assert QualityMode.FAST in QUALITY_MODE_LABELS
        assert QualityMode.BALANCED in QUALITY_MODE_LABELS
        assert QualityMode.RESEARCH in QUALITY_MODE_LABELS

    def test_labels_are_non_empty_strings(self):
        for mode, label in QUALITY_MODE_LABELS.items():
            assert isinstance(label, str) and label.strip(), (
                f"Label for {mode} must be a non-empty string"
            )

    def test_labels_are_unique(self):
        labels = list(QUALITY_MODE_LABELS.values())
        assert len(labels) == len(set(labels)), "All labels must be unique"

    def test_labels_map_back_to_modes(self):
        for mode, label in QUALITY_MODE_LABELS.items():
            found = next((k for k, v in QUALITY_MODE_LABELS.items() if v == label), None)
            assert found == mode


# ── OUTPUT_FORMAT_LABELS ─────────────────────────────────────────────────────

class TestOutputFormatLabels:

    def test_all_four_formats_have_labels(self):
        for fmt in OutputFormat:
            assert fmt in OUTPUT_FORMAT_LABELS

    def test_labels_are_unique(self):
        labels = list(OUTPUT_FORMAT_LABELS.values())
        assert len(labels) == len(set(labels))


# ── Valid construction ────────────────────────────────────────────────────────

class TestDistillationConfigValid:

    def test_minimal_construction(self):
        cfg = DistillationConfig(teacher_model="gpt-4o")
        assert cfg.teacher_model == "gpt-4o"
        assert cfg.quality_mode == QualityMode.BALANCED
        assert cfg.dataset_size == 2000

    def test_all_fields_explicit(self):
        cfg = DistillationConfig(
            teacher_model="gpt-4o-mini",
            dataset_size=500,
            quality_mode=QualityMode.RESEARCH,
            output_format=OutputFormat.SHAREGPT,
            use_vllm=False,
            train_model=True,
            publish_dataset=True,
            hf_repo="user/repo",
            temperature=1.0,
            max_new_tokens=512,
            batch_size=32,
            lora_rank=32,
            api_key="sk-test",
            use_semantic_chunking=True,
            enable_dedup=False,
        )
        assert cfg.teacher_model == "gpt-4o-mini"
        assert cfg.quality_mode == QualityMode.RESEARCH
        assert cfg.output_format == OutputFormat.SHAREGPT

    def test_teacher_model_whitespace_stripped(self):
        cfg = DistillationConfig(teacher_model="  gpt-4o  ")
        assert cfg.teacher_model == "gpt-4o"

    def test_comma_separated_teacher_model_accepted(self):
        cfg = DistillationConfig(teacher_model="gpt-4o,gpt-3.5-turbo")
        assert "gpt-4o" in cfg.teacher_model

    def test_judge_model_field_exists(self):
        cfg = DistillationConfig(teacher_model="gpt-4o")
        assert hasattr(cfg, "judge_model")
        assert cfg.judge_model == "gpt-4o-mini"

    def test_default_api_key_is_none(self):
        cfg = DistillationConfig(teacher_model="gpt-4o")
        assert cfg.api_key is None

    def test_default_output_format_is_alpaca(self):
        cfg = DistillationConfig(teacher_model="gpt-4o")
        assert cfg.output_format == OutputFormat.ALPACA

    def test_default_enable_dedup_is_true(self):
        cfg = DistillationConfig(teacher_model="gpt-4o")
        assert cfg.enable_dedup is True


# ── Validation errors ─────────────────────────────────────────────────────────

class TestDistillationConfigInvalid:

    def test_blank_teacher_model_raises(self):
        with pytest.raises(ValidationError, match="Teacher model is required"):
            DistillationConfig(teacher_model="")

    def test_whitespace_teacher_model_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="   ")

    def test_dataset_size_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="gpt-4o", dataset_size=99)

    def test_dataset_size_above_maximum_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="gpt-4o", dataset_size=50_001)

    def test_dataset_size_boundaries_accepted(self):
        DistillationConfig(teacher_model="gpt-4o", dataset_size=100)
        DistillationConfig(teacher_model="gpt-4o", dataset_size=50_000)

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="gpt-4o", temperature=-0.1)

    def test_temperature_above_two_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="gpt-4o", temperature=2.1)

    def test_temperature_boundaries_accepted(self):
        DistillationConfig(teacher_model="gpt-4o", temperature=0.0)
        DistillationConfig(teacher_model="gpt-4o", temperature=2.0)

    def test_max_new_tokens_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="gpt-4o", max_new_tokens=127)

    def test_lora_rank_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="gpt-4o", lora_rank=7)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="gpt-4o", batch_size=0)


# ── FIX C-01: safe_dict() ────────────────────────────────────────────────────

class TestSafeDict:

    def test_safe_dict_redacts_api_key(self):
        cfg = DistillationConfig(teacher_model="gpt-4o", api_key="sk-secret-123")
        safe = cfg.safe_dict()
        assert safe["api_key"] == "***REDACTED***"
        assert "sk-secret-123" not in str(safe)

    def test_safe_dict_preserves_other_fields(self):
        cfg = DistillationConfig(teacher_model="gpt-4o", api_key="sk-secret")
        safe = cfg.safe_dict()
        assert safe["teacher_model"] == "gpt-4o"
        assert "dataset_size" in safe

    def test_safe_dict_without_api_key_omits_field(self):
        cfg = DistillationConfig(teacher_model="gpt-4o")
        safe = cfg.safe_dict()
        assert "api_key" not in safe

    def test_safe_dict_is_json_serialisable(self):
        cfg = DistillationConfig(teacher_model="gpt-4o", api_key="sk-secret")
        safe = cfg.safe_dict()
        try:
            json.dumps(safe)
        except (TypeError, ValueError) as e:
            pytest.fail(f"safe_dict() is not JSON-serialisable: {e}")


# ── FIX C-02: API key never leaks via repr/str ──────────────────────────────

class TestApiKeyNeverLeaks:

    @pytest.fixture()
    def cfg_with_key(self):
        return DistillationConfig(teacher_model="gpt-4o", api_key="sk-supersecret-key-12345")

    def test_repr_does_not_contain_api_key(self, cfg_with_key):
        assert "sk-supersecret-key-12345" not in repr(cfg_with_key)

    def test_str_does_not_contain_api_key(self, cfg_with_key):
        assert "sk-supersecret-key-12345" not in str(cfg_with_key)

    def test_safe_dict_redacts_api_key(self, cfg_with_key):
        safe = cfg_with_key.safe_dict()
        assert "sk-supersecret-key-12345" not in str(safe)
        assert safe["api_key"] == "***REDACTED***"

    def test_model_dump_via_safe_dict_never_leaks(self, cfg_with_key):
        safe = cfg_with_key.safe_dict()
        for value in safe.values():
            assert "sk-supersecret-key-12345" not in str(value)


# ── Parametrized quality mode round-trips ────────────────────────────────────

@pytest.mark.parametrize("mode_str,expected", [
    ("fast", QualityMode.FAST),
    ("balanced", QualityMode.BALANCED),
    ("research", QualityMode.RESEARCH),
])
def test_quality_mode_round_trip(mode_str, expected):
    cfg = DistillationConfig(teacher_model="gpt-4o", quality_mode=mode_str)
    assert cfg.quality_mode == expected
