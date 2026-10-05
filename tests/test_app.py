"""
tests/test_app.py

Runs the real app.py through Streamlit's AppTest harness: validation comes from
DistillationConfig, and a full upload -> Generate -> results flow runs the real
pipeline (in its child process) with only the LLM replaced by FakeLLM.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import orchestrator
from tests.fake_llm import FakeLLM

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
GENERATE = "🚀 Generate Dataset"
DOC = "\n\n".join(
    f"Section {i}. " + (f"Topic {i} covers distinct material about subject number {i}. ") * 12
    for i in range(4)
).encode()


@pytest.fixture()
def at(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    for var in ("OPENAI_API_KEY", "HF_TOKEN", "BRAINBREW_REQUIRE_LOGIN", "HF_USERNAME"):
        monkeypatch.delenv(var, raising=False)
    app = AppTest.from_file(APP_PATH, default_timeout=180)
    app.run()
    assert not app.exception, app.exception
    return app


def _by_label(widgets, prefix: str):
    return next(w for w in widgets if w.label.startswith(prefix))


def _errors(app: AppTest) -> str:
    return "\n".join(e.value for e in app.error)


def _generate_button(app: AppTest):
    return _by_label(app.button, GENERATE)


def _ready_for_openai(app: AppTest, key: str = "sk-test") -> AppTest:
    _by_label(app.sidebar.checkbox, "Use vLLM").uncheck()
    _by_label(app.sidebar.text_input, "OpenAI API Key").input(key)
    app.file_uploader[0].set_value(("notes.txt", DOC, "text/plain"))
    return app.run()


class TestValidation:

    def test_no_upload_blocks_generation(self, at):
        assert "Upload at least one document" in _errors(at)
        assert _generate_button(at).disabled

    def test_api_key_rule_comes_from_config(self, at):
        _by_label(at.sidebar.checkbox, "Use vLLM").uncheck()
        at.run()
        assert "An API key is required when not using vLLM" in _errors(at)

    def test_teacher_model_rule_comes_from_config(self, at):
        _by_label(at.text_input, "Teacher Model").input("../../etc/passwd")
        at.run()
        assert "Teacher model: Model name cannot contain path traversal" in _errors(at)

    def test_publish_rules_come_from_config(self, at):
        _by_label(at.checkbox, "Publish to Hugging Face").check()
        at.run()
        _by_label(at.text_input, "Hugging Face Repo").input("not-a-repo")
        at.run()
        assert "Hugging Face repo: Invalid Hugging Face repository name" in _errors(at)

    def test_publish_needs_token(self, at):
        _by_label(at.checkbox, "Publish to Hugging Face").check()
        at.run()
        _by_label(at.text_input, "Hugging Face Repo").input("user/repo")
        at.run()
        assert "A Hugging Face token is required" in _errors(at)

    def test_valid_input_enables_generation(self, at):
        _ready_for_openai(at)
        assert not at.error
        assert not _generate_button(at).disabled

    def test_any_filename_is_accepted(self, at):
        _by_label(at.sidebar.checkbox, "Use vLLM").uncheck()
        _by_label(at.sidebar.text_input, "OpenAI API Key").input("sk-test")
        at.file_uploader[0].set_value(("Report (1) — café.txt", DOC, "text/plain"))
        at.run()
        assert not at.error


class TestSettings:

    def test_generation_settings_exposed(self, at):
        labels = {w.label for w in [*at.sidebar.slider, *at.sidebar.number_input]}
        assert {"Temperature", "Max answer length (tokens)", "Batch size"} <= labels

    def test_lora_settings_only_when_training(self, at):
        assert not any(w.label.startswith("Base model") for w in at.text_input)
        _by_label(at.checkbox, "Auto-train LoRA adapter").check()
        at.run()
        assert _by_label(at.text_input, "Base model").value == "Qwen/Qwen3-4B-Instruct-2507"
        assert _by_label(at.select_slider, "LoRA rank").value == 16

    def test_no_judge_model_or_resume_controls(self, at):
        labels = " ".join(w.label for w in [*at.text_input, *at.sidebar.text_input, *at.checkbox])
        assert "judge" not in labels.lower() and "resume" not in labels.lower()


class TestGenerationFlow:

    def test_generate_shows_results_that_survive_reruns(self, at):
        _ready_for_openai(at)
        with patch.object(orchestrator, "_create_llm", lambda name, cfg: FakeLLM()):
            _generate_button(at).click()
            at.run()
        assert not at.exception, at.exception
        assert any("Dataset generated" in s.value for s in at.success), _errors(at)
        run_id = at.session_state["run_id"]

        # Any widget interaction reruns the script; results must still be there.
        _by_label(at.sidebar.checkbox, "Semantic chunking").check()
        at.run()
        assert at.session_state["run_id"] == run_id
        assert any("Dataset Quality" in m.value for m in at.markdown)
        download = _by_label(at.get("download_button"), "📥 Download dataset")
        assert download is not None
        assert {m.label for m in at.metric} >= {"Records", "Avg. Output Length", "Uniqueness"}

    def test_failure_is_reported(self, at):
        _ready_for_openai(at)
        with patch.object(orchestrator, "generate_rows", side_effect=RuntimeError("teacher unreachable")):
            _generate_button(at).click()
            at.run()
        assert "Generation failed: teacher unreachable" in _errors(at)
        assert "run_id" not in at.session_state

    def test_unreadable_document(self, at):
        _ready_for_openai(at)
        at.file_uploader[0].set_value(("broken.pdf", b"%PDF-1.7 this is not really a pdf", "application/pdf"))
        at.run()
        _generate_button(at).click()
        at.run()
        assert "No text could be extracted" in _errors(at) or any(
            "Could not parse" in w.value for w in at.warning
        )
