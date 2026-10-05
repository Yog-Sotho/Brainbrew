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
    OUTPUT_FORMAT_LABELS,
    QUALITY_MODE_LABELS,
    DistillationConfig,
    OutputFormat,
    QualityMode,
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

# A local OpenAI-compatible server needs no API key, so field tests stay focused.
LOCAL_URL = "http://localhost:8000/v1"


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
        cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o")
        assert cfg.teacher_model == "gpt-4o"
        assert cfg.quality_mode == QualityMode.BALANCED
        assert cfg.dataset_size == 500
        assert cfg.base_url == LOCAL_URL

    def test_all_fields_explicit(self):
        cfg = DistillationConfig(
            teacher_model="gpt-4o-mini",
            dataset_size=500,
            quality_mode=QualityMode.RESEARCH,
            output_format=OutputFormat.SHAREGPT,
            train_model=True,
            publish_dataset=True,
            hf_repo="user/repo",
            hf_token="hf_test",
            temperature=1.0,
            max_new_tokens=512,
            concurrency=16,
            request_timeout=300,
            judge_model="gpt-4o",
            judge_threshold=3,
            lora_rank=32,
            api_key="sk-test",
            use_semantic_chunking=True,
            enable_dedup=False,
        )
        assert cfg.teacher_model == "gpt-4o-mini"
        assert cfg.quality_mode == QualityMode.RESEARCH
        assert cfg.output_format == OutputFormat.SHAREGPT

    def test_teacher_model_whitespace_stripped(self):
        cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model="  gpt-4o  ")
        assert cfg.teacher_model == "gpt-4o"

    def test_comma_separated_teacher_model_accepted(self):
        cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o,gpt-3.5-turbo")
        assert "gpt-4o" in cfg.teacher_model

    def test_removed_fields(self):
        # In-process vLLM and distilabel batching are gone (Phase 2); resume lost data (Phase 1).
        cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o")
        for gone in ("use_vllm", "batch_size", "checkpoint_dir"):
            assert not hasattr(cfg, gone)

    def test_quality_mode_controls_judge_and_evolve(self):
        def mk(mode):
            return DistillationConfig(base_url=LOCAL_URL, teacher_model="m", quality_mode=mode)
        assert (mk("fast").uses_judge, mk("fast").evolves) == (False, False)
        assert (mk("balanced").uses_judge, mk("balanced").evolves) == (True, False)
        assert (mk("research").uses_judge, mk("research").evolves) == (True, True)

    def test_teacher_models_list(self):
        cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model=" a , b ,")
        assert cfg.teacher_models == ["a", "b"]

    def test_default_base_model_has_chat_template_family(self):
        from config import DEFAULT_BASE_MODEL
        assert DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o").base_model == DEFAULT_BASE_MODEL

    def test_public_dict_has_no_secrets(self):
        cfg = DistillationConfig(teacher_model="gpt-4o", api_key="sk-x", hf_token="hf_x")
        public = cfg.public_dict()
        assert "api_key" not in public and "hf_token" not in public
        assert public["output_format"] == "alpaca"  # JSON-safe values

    def test_default_api_key_is_none(self):
        cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o")
        assert cfg.api_key is None

    def test_default_output_format_is_alpaca(self):
        cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o")
        assert cfg.output_format == OutputFormat.ALPACA

    def test_default_enable_dedup_is_true(self):
        cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o")
        assert cfg.enable_dedup is True


# ── Validation errors ─────────────────────────────────────────────────────────

class TestDistillationConfigInvalid:

    def test_blank_teacher_model_raises(self):
        with pytest.raises(ValidationError, match="Teacher model is required"):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="")

    def test_whitespace_teacher_model_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="   ")

    def test_dataset_size_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", dataset_size=9)

    def test_dataset_size_above_maximum_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", dataset_size=50_001)

    def test_dataset_size_boundaries_accepted(self):
        DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", dataset_size=10)
        DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", dataset_size=50_000)

    def test_temperature_below_zero_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", temperature=-0.1)

    def test_temperature_above_two_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", temperature=2.1)

    def test_temperature_boundaries_accepted(self):
        DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", temperature=0.0)
        DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", temperature=2.0)

    def test_max_new_tokens_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", max_new_tokens=127)

    def test_lora_rank_below_minimum_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", lora_rank=3)

    def test_concurrency_zero_raises(self):
        with pytest.raises(ValidationError):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", concurrency=0)


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
        cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o")
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
    cfg = DistillationConfig(base_url=LOCAL_URL, teacher_model="gpt-4o", quality_mode=mode_str)
    assert cfg.quality_mode == expected


class TestCrossFieldRules:
    """Rules the UI used to duplicate; now only DistillationConfig enforces them."""

    def test_api_key_required_for_openai(self):
        with pytest.raises(ValidationError, match="API key is required for the OpenAI API"):
            DistillationConfig(teacher_model="gpt-4o")

    def test_blank_api_key_counts_as_missing(self):
        with pytest.raises(ValidationError, match="API key is required"):
            DistillationConfig(teacher_model="gpt-4o", api_key="   ")

    def test_no_api_key_needed_for_a_local_server(self):
        assert DistillationConfig(teacher_model="m", base_url=LOCAL_URL).api_key is None

    def test_all_cross_field_problems_reported_together(self):
        with pytest.raises(ValidationError) as exc:
            DistillationConfig(teacher_model="m", publish_dataset=True)
        msg = str(exc.value)
        assert "API key is required" in msg and "hf_repo is required" in msg and "token is required" in msg

    def test_publish_requires_token(self):
        with pytest.raises(ValidationError, match="Hugging Face token is required"):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="m", publish_dataset=True, hf_repo="user/repo")

    def test_publish_requires_repo(self):
        with pytest.raises(ValidationError, match="hf_repo is required"):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="m", publish_dataset=True, hf_token="hf_x")

    @pytest.mark.parametrize("field,value", [
        ("max_new_tokens", 32769), ("concurrency", 65), ("lora_rank", 257),
        ("request_timeout", 1801), ("judge_threshold", 6), ("dataset_size", 50_001),
    ])
    def test_upper_bounds(self, field, value):
        with pytest.raises(ValidationError):
            DistillationConfig(base_url=LOCAL_URL, teacher_model="m", **{field: value})


def test_check_hf_repo_name_shared_rule():
    from config import check_hf_repo_name
    assert check_hf_repo_name("  user/repo ") == "user/repo"
    for bad in ["user", "user/../x", "a/b/c", "user/my repo", ""]:
        with pytest.raises(ValueError, match="Invalid Hugging Face repository name"):
            check_hf_repo_name(bad)


class TestEndpointUrl:

    @pytest.mark.parametrize("url,expected", [
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
        ("https://api.example.com/v1/", "https://api.example.com/v1"),
        ("  http://vllm:8000/v1  ", "http://vllm:8000/v1"),
    ])
    def test_valid(self, url, expected):
        assert DistillationConfig(teacher_model="m", base_url=url).base_url == expected

    @pytest.mark.parametrize("url", [
        "ftp://host/v1", "localhost:8000", "http://", "https://user:pw@host/v1",
        "http://host/v1?key=1", "http://host/v1#x", "http://host/v 1", "javascript:alert(1)",
    ])
    def test_invalid(self, url):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="m", base_url=url)

    def test_blank_means_openai(self):
        cfg = DistillationConfig(teacher_model="m", base_url="  ", api_key="sk-x")
        assert cfg.base_url is None


class TestDataCleaningOptions:

    def _cfg(self, **kw) -> DistillationConfig:
        return DistillationConfig(teacher_model="m", base_url=LOCAL_URL, **kw)

    def test_defaults(self):
        cfg = self._cfg()
        assert cfg.pii_url_policy == "domain" and cfg.pii_presidio is False and cfg.decontaminate == []

    def test_benchmarks_validated_and_deduplicated(self):
        assert self._cfg(decontaminate=["gsm8k", "mmlu", "gsm8k"]).decontaminate == ["gsm8k", "mmlu"]
        with pytest.raises(ValidationError, match="Unknown benchmark"):
            self._cfg(decontaminate=["not-a-benchmark"])

    def test_url_policy_validated(self):
        assert self._cfg(pii_url_policy="keep").pii_url_policy == "keep"
        with pytest.raises(ValidationError):
            self._cfg(pii_url_policy="sometimes")


class TestEmbeddingModel:

    def test_blank_is_off(self):
        cfg = DistillationConfig(teacher_model="m", base_url=LOCAL_URL, embedding_model="  ")
        assert cfg.embedding_model is None

    def test_validated_like_other_model_names(self):
        with pytest.raises(ValidationError, match="path traversal"):
            DistillationConfig(teacher_model="m", base_url=LOCAL_URL, embedding_model="../x")

    def test_threshold_range(self):
        with pytest.raises(ValidationError):
            DistillationConfig(teacher_model="m", base_url=LOCAL_URL, semantic_dedup_threshold=1.0)
