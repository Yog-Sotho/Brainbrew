"""
tests/test_app.py

Runs the real app.py through Streamlit's AppTest harness: validation comes from
DistillationConfig, and a full upload -> Generate -> results flow runs the real
pipeline with only the model server replaced by the in-memory FakeOpenAI.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx2
import pytest

import orchestrator
from engine import ChatClient, EndpointSettings
from tests.fake_openai import FakeOpenAI

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
GENERATE = "🚀 Generate Dataset"
DOC = "\n\n".join(
    f"Section {i}. " + (f"Topic {i} covers distinct material about subject number {i}. ") * 12
    for i in range(4)
).encode()
ENV_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "HF_TOKEN", "BRAINBREW_REQUIRE_LOGIN",
            "HF_USERNAME", "BRAINBREW_ALLOW_CUSTOM_ENDPOINTS", "BRAINBREW_DEFAULT_MODEL")


def _app(monkeypatch: pytest.MonkeyPatch, **env: str) -> AppTest:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    app = AppTest.from_file(APP_PATH, default_timeout=180)
    app.run()
    assert not app.exception, app.exception
    return app


@pytest.fixture()
def at(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    return _app(monkeypatch)


def _by_label(widgets, prefix: str):
    return next(w for w in widgets if w.label.startswith(prefix))


def _errors(app: AppTest) -> str:
    return "\n".join(e.value for e in app.error)


def _generate_button(app: AppTest):
    return _by_label(app.button, GENERATE)


def _upload(app: AppTest, name: str = "notes.txt", data: bytes = DOC, mime: str = "text/plain") -> AppTest:
    app.file_uploader[0].set_value((name, data, mime))
    return app.run()


def _ready(app: AppTest, key: str = "sk-test") -> AppTest:
    _by_label(app.sidebar.text_input, "API Key").input(key)
    _by_label(app.slider, "Target Dataset Size").set_value(12)
    return _upload(app)


@contextmanager
def _fake_server(fake: FakeOpenAI | None = None):
    """Route every client the pipeline creates to the fake; yield their settings."""
    fake = fake or FakeOpenAI()
    made: list[EndpointSettings] = []

    def make(settings: EndpointSettings) -> ChatClient:
        made.append(settings)
        return ChatClient(settings, http_client=httpx2.AsyncClient(transport=fake.transport))

    with patch.object(orchestrator, "_make_client", make):
        yield made


class TestValidation:

    def test_no_upload_blocks_generation(self, at):
        assert "Upload at least one document" in _errors(at)
        assert _generate_button(at).disabled

    def test_openai_needs_a_key(self, at):
        assert "An API key is required for the OpenAI API" in _errors(at)

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
        # Cross-field rules run once the fields themselves are valid.
        _by_label(at.text_input, "Hugging Face Repo").input("user/repo")
        at.run()
        assert "A Hugging Face token is required" in _errors(at)

    def test_valid_input_enables_generation(self, at):
        _ready(at)
        assert not at.error
        assert not _generate_button(at).disabled

    def test_any_filename_is_accepted(self, at):
        _by_label(at.sidebar.text_input, "API Key").input("sk-test")
        _upload(at, "Report (1) — café.txt")
        assert not at.error

    def test_unreadable_pdf_is_reported(self, at):
        _by_label(at.sidebar.text_input, "API Key").input("sk-test")
        _upload(at, "broken.pdf", b"%PDF-1.7 this is not really a pdf", "application/pdf")
        _generate_button(at).click()
        at.run()
        assert any("Could not parse" in w.value for w in at.warning) or "No text" in _errors(at)


class TestEndpoints:

    def test_local_endpoints_need_no_key(self, at):
        _by_label(at.sidebar.selectbox, "Model endpoint").set_value("Local vLLM server (localhost:8000)")
        at.run()
        _upload(at)
        assert not at.error

    def test_custom_url_is_validated(self, at):
        _by_label(at.sidebar.selectbox, "Model endpoint").set_value("Custom URL…")
        at.run()
        assert "Enter the endpoint URL" in _errors(at)
        _by_label(at.sidebar.text_input, "Endpoint URL").input("ftp://user:pw@host/v1")
        at.run()
        assert "Endpoint URL:" in _errors(at)

    def test_operator_can_lock_the_endpoint(self, monkeypatch):
        app = _app(monkeypatch, BRAINBREW_ALLOW_CUSTOM_ENDPOINTS="0", OPENAI_BASE_URL="http://vllm:8000/v1")
        assert _by_label(app.sidebar.selectbox, "Model endpoint").options == ["Server default"]

    def test_operator_sets_the_default_model(self, monkeypatch):
        app = _app(monkeypatch, OPENAI_BASE_URL="http://vllm:8000/v1", BRAINBREW_DEFAULT_MODEL="Qwen/Qwen3-4B")
        assert _by_label(app.text_input, "Teacher Model").value == "Qwen/Qwen3-4B"
        assert not app.error or "Upload" in _errors(app)

    def test_server_key_never_sent_to_other_endpoints(self, monkeypatch):
        app = _app(monkeypatch, OPENAI_API_KEY="sk-server-secret")
        _by_label(app.sidebar.selectbox, "Model endpoint").set_value("Custom URL…")
        app.run()
        _by_label(app.sidebar.text_input, "Endpoint URL").input("https://attacker.example/v1")
        _by_label(app.slider, "Target Dataset Size").set_value(12)
        _upload(app)
        with _fake_server() as made:
            _generate_button(app).click()
            app.run()
        assert made and all(s.api_key is None for s in made)
        assert all(s.base_url == "https://attacker.example/v1" for s in made)

    def test_server_key_used_for_server_endpoint(self, monkeypatch):
        app = _app(monkeypatch, OPENAI_API_KEY="sk-server-secret")
        _by_label(app.slider, "Target Dataset Size").set_value(12)
        _upload(app)
        with _fake_server() as made:
            _generate_button(app).click()
            app.run()
        assert made and all(s.api_key == "sk-server-secret" and s.base_url is None for s in made)


class TestSettings:

    def test_generation_settings_exposed(self, at):
        labels = {w.label for w in [*at.sidebar.slider, *at.sidebar.number_input, *at.sidebar.text_input,
                                    *at.sidebar.select_slider]}
        assert {"Temperature", "Max answer length (tokens)", "Parallel requests",
                "Request timeout (seconds)", "Judge model", "Minimum judge score"} <= labels

    def test_lora_settings_only_when_training(self, at):
        assert not any(w.label.startswith("Base model") for w in at.text_input)
        _by_label(at.checkbox, "Auto-train LoRA adapter").check()
        at.run()
        assert _by_label(at.text_input, "Base model").value == "Qwen/Qwen3-4B-Instruct-2507"

    def test_yield_estimate_after_upload(self, at):
        _ready(at)
        info = " ".join(i.value for i in at.info)
        assert "chunks" in info and "pairs possible" in info

    def test_target_beyond_capacity_warns(self, at):
        _ready(at)
        _by_label(at.slider, "Target Dataset Size").set_value(5000)
        at.run()
        assert any("likely support about" in w.value for w in at.warning)


class TestGenerationFlow:

    def test_generate_shows_results_that_survive_reruns(self, at):
        _ready(at)
        with _fake_server():
            _generate_button(at).click()
            at.run()
        assert not at.exception, at.exception
        assert any("Dataset generated" in s.value for s in at.success), _errors(at)
        run_id = at.session_state["run_id"]

        _by_label(at.sidebar.checkbox, "Semantic chunking").check()
        at.run()
        assert at.session_state["run_id"] == run_id
        assert any("Dataset Quality" in m.value for m in at.markdown)
        assert _by_label(at.get("download_button"), "📥 Download dataset") is not None
        assert {m.label for m in at.metric} >= {"Records", "Avg. Output Length", "Uniqueness"}

    def test_failure_is_reported(self, at):
        _ready(at)
        with _fake_server(FakeOpenAI(auth_error=True)):
            _generate_button(at).click()
            at.run()
        assert "Generation failed" in _errors(at)
        assert "run_id" not in at.session_state
