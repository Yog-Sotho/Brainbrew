"""
tests/test_app_secrets.py

Regression tests: server-side secrets must never reach the browser.

Runs the real app.py through Streamlit's AppTest harness and inspects every
element proto the server would send to a client.
"""
from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
SERVER_OPENAI_KEY = "sk-server-secret-must-not-leak-0123456789"
SERVER_HF_TOKEN = "hf_server_secret_must_not_leak_0123456789"


@pytest.fixture()
def app_with_server_secrets(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("OPENAI_API_KEY", SERVER_OPENAI_KEY)
    monkeypatch.setenv("HF_TOKEN", SERVER_HF_TOKEN)
    monkeypatch.delenv("BRAINBREW_REQUIRE_LOGIN", raising=False)
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _all_protos(at: AppTest) -> str:
    """Serialize every rendered element (main + sidebar) the way the frontend receives it."""
    chunks: list[str] = []
    for node in [*at.main, *at.sidebar]:  # Block iteration walks all descendants
        proto = getattr(node, "proto", None)
        if proto is not None:
            chunks.append(str(proto))
    return "\n".join(chunks)


class TestServerSecretsStayServerSide:

    def test_password_inputs_are_not_prefilled(self, app_with_server_secrets: AppTest):
        for widget in app_with_server_secrets.text_input:
            assert widget.value not in (SERVER_OPENAI_KEY, SERVER_HF_TOKEN), (
                f"Widget {widget.label!r} is pre-filled with a server secret"
            )

    def test_secrets_absent_from_every_rendered_element(self, app_with_server_secrets: AppTest):
        payload = _all_protos(app_with_server_secrets)
        assert payload, "No elements rendered; the test would be vacuous"
        assert SERVER_OPENAI_KEY not in payload
        assert SERVER_HF_TOKEN not in payload

    def test_user_is_told_server_key_is_configured(self, app_with_server_secrets: AppTest):
        captions = " ".join(c.value for c in app_with_server_secrets.sidebar.caption)
        assert "Server API key configured" in captions

    def test_missing_key_validation_satisfied_by_server_key(self, app_with_server_secrets: AppTest):
        at = app_with_server_secrets
        at.sidebar.checkbox[0].uncheck().run()  # switch to OpenAI mode
        errors = " ".join(e.value for e in at.error)
        assert "OpenAI API Key is required" not in errors


class TestLoginGate:

    def test_login_required_blocks_app(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BRAINBREW_REQUIRE_LOGIN", "1")
        monkeypatch.setenv("OPENAI_API_KEY", SERVER_OPENAI_KEY)
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        assert not at.exception, at.exception
        # No [auth] config in the test env: the gate must fail closed.
        assert any("auth" in e.value for e in at.error)
        assert not at.text_input, "Settings must not render before login"
        assert not at.file_uploader, "Uploader must not render before login"
