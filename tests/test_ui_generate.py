"""
tests/test_ui_generate.py

The Generate page's widget-free logic: endpoint choices, estimates, readable
validation errors and upload checks.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from config import DistillationConfig
from ui.generate import (
    CUSTOM,
    MAX_HARD_BYTES,
    endpoint_options,
    estimate,
    friendly_errors,
    server_endpoint_label,
    upload_errors,
)


class TestEndpoints:

    def test_locked_offers_only_the_server_endpoint(self):
        assert endpoint_options("http://vllm:8000/v1", allow_custom=False) == {"Server default": "http://vllm:8000/v1"}
        assert endpoint_options(None, allow_custom=False) == {"OpenAI API": None}

    def test_custom_choices_come_after_the_server_endpoint(self):
        options = endpoint_options(None, allow_custom=True)
        assert next(iter(options)) == server_endpoint_label(None) == "OpenAI API"
        assert options["Custom URL…"] == CUSTOM
        assert "http://localhost:11434/v1" in options.values()


class TestEstimate:

    def test_local_is_free(self):
        assert estimate("llama3.1:8b", 200, "balanced", local=True) == ("Free (your server)", "~9 min")

    def test_hosted_uses_the_first_model_price(self):
        cost, minutes = estimate("gpt-4o-mini, gpt-4.1", 1000, "fast", local=False)
        assert cost.startswith("~$") and minutes == "~25 min"

    def test_minimum_one_minute_and_unknown_mode(self):
        assert estimate("m", 1, "unknown", local=True)[1] == "~1 min"


class TestFriendlyErrors:

    def test_fields_get_ui_labels(self):
        with pytest.raises(ValidationError) as info:
            DistillationConfig(teacher_model="m; rm -rf /", base_url="http://169.254.169.254/v1")
        messages = friendly_errors(info.value)
        assert any(m.startswith("Teacher model: ") for m in messages)
        assert any(m.startswith("Endpoint URL: ") and "metadata" in m for m in messages)
        assert not any("Value error," in m for m in messages)

    def test_model_level_errors_have_no_label(self):
        with pytest.raises(ValidationError) as info:
            DistillationConfig(teacher_model="m", base_url="http://localhost:8000/v1", publish_dataset=True)
        assert any(m.startswith("hf_repo is required") for m in friendly_errors(info.value))


class TestUploadErrors:

    def _file(self, name: str, size: int):
        return SimpleNamespace(name=name, size=size)

    def test_nothing_uploaded(self):
        assert upload_errors([], "", []) == ["Upload at least one document (PDF/TXT) to begin."]
        assert upload_errors(None, "", []) == ["Upload at least one document (PDF/TXT) to begin."]

    def test_oversized_files_are_named(self):
        errors = upload_errors([self._file("big.pdf", MAX_HARD_BYTES + 1), self._file("ok.txt", 10)], "text", [])
        assert len(errors) == 1 and "big.pdf" in errors[0]

    def test_no_text_extracted(self):
        assert upload_errors([self._file("a.pdf", 10)], "  ", []) == [
            "No text could be extracted from the uploaded documents."
        ]

    def test_read_errors_are_reported_elsewhere(self):
        # When some files failed to read, the page shows those warnings instead.
        assert upload_errors([self._file("a.pdf", 10)], "", ["a.pdf: unreadable"]) == []
