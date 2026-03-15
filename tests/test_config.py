"""
tests/test_config.py

Tests for config.py — DistillationConfig and QualityMode.

Covers:
  - Valid construction (all defaults, all fields explicit)
  - Field boundary validation (ge/le/ge enforcement)
  - teacher_model validator (blank, whitespace-only, comma-separated)
  - api_key never leaks via repr() or safe_dict()
  - safe_dict() redacts only api_key, preserves all other fields
  - QualityMode enum exhaustiveness
  - Pydantic type coercion and rejection
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import DistillationConfig, QualityMode


# ---------------------------------------------------------------------------
# QualityMode
# ---------------------------------------------------------------------------

class TestQualityMode:

    def test_all_three_values_exist(self):
        assert QualityMode.FAST.value == "fast"
        assert QualityMode.BALANCED.value == "balanced"
        assert QualityMode.RESEARCH.value == "research"

    def test_string_coercion(self):
        """QualityMode is a str-enum; "balanced" must round-trip."""
        assert QualityMode("balanced") is QualityMode.BALANCED

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            QualityMode("turbo")


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------

class TestDistillationConfigValid:

    def test_minimal_construction(self):
        cfg = DistillationConfig(teacher_model="gpt-4o")
        assert cfg.teacher_model == "gpt-4o"
        assert cfg.quality_mode == QualityMode.BALANCED
        assert cfg.dataset_size == 2000
        assert cfg.use_vllm is True

    def test_all_fields_explicit(self):
        cfg = DistillationConfig(
            teacher_model="gpt-4o-mini",
            judge_model="gpt-4o",
            dataset_size=500,
            quality_mode=QualityMode.RESEARCH,
            use_vllm=False,
            train_model=True,
            base_model="unsloth/llama-3-8b-bnb-4bit",
            publish_dataset=True,
            hf_repo="user/repo",
            temperature=1.0,
            max_new_tokens=512,
            batch_size=32,
            lora_rank=32,
            api_key="sk-test",
        )
        assert cfg.teacher_model == "gpt-4o-mini"
        assert cfg.quality_mode == QualityMode.RESEARCH
        assert cfg.dataset_size == 500
        assert cfg.temperature == 1.0

    def test_teacher_model_whitespace_stripped(self):
        cfg = DistillationConfig(teacher_model="  gpt-4o  ")
        assert cfg.teacher_model == "gpt-4o"

    def test_comma_separated_teacher_model_accepted(self):
        cfg = DistillationConfig(teacher_model="gpt-4o,gpt-3.5-turbo")
        assert "gpt-4o" in cfg.teacher_model

    def test_default_api_key_is_none(self):
        cfg = DistillationConfig(teacher_model="gpt-4o")
        assert cfg.api_key is None


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

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

    def test_dataset_size_at_minimum_boundary_accepted(self):
        cfg = DistillationConfig(teacher_model="gpt-4o", dataset_size=100)
        assert cfg.dataset_size == 100

    def test_dataset_size_at_maximum_boundary_accepted(self):
        cfg = DistillationConfig(teacher_model="gpt-4o", dataset_size=50_000)
        assert cfg.dataset_size == 50_000

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="gpt-4o", temperature=-0.1)

    def test_temperature_above_two_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="gpt-4o", temperature=2.1)

    def test_temperature_at_boundaries_accepted(self):
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


# ---------------------------------------------------------------------------
# Security: api_key must never leak
# ---------------------------------------------------------------------------

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

    def test_safe_dict_preserves_all_other_fields(self, cfg_with_key):
        safe = cfg_with_key.safe_dict()
        assert safe["teacher_model"] == "gpt-4o"
        assert "dataset_size" in safe
        assert "quality_mode" in safe

    def test_safe_dict_without_api_key_has_no_api_key_field(self):
        cfg = DistillationConfig(teacher_model="gpt-4o")
        safe = cfg.safe_dict()
        # api_key=None is excluded by exclude_none=True
        assert "api_key" not in safe

    def test_model_dump_excludes_api_key_from_safe_dict(self, cfg_with_key):
        """safe_dict must always redact regardless of how model_dump is called."""
        safe = cfg_with_key.safe_dict()
        for value in safe.values():
            assert "sk-supersecret-key-12345" not in str(value)


# ---------------------------------------------------------------------------
# Parametrized quality mode round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode_str,expected", [
    ("fast", QualityMode.FAST),
    ("balanced", QualityMode.BALANCED),
    ("research", QualityMode.RESEARCH),
])
def test_quality_mode_round_trip(mode_str, expected):
    cfg = DistillationConfig(teacher_model="gpt-4o", quality_mode=mode_str)
    assert cfg.quality_mode == expected
