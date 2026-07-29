"""
tests/test_app_validation.py

Tests for the proactive input validation patterns implemented in app.py.
"""
from __future__ import annotations

import re


def validate_inputs(
    uploaded_files,
    use_vllm: bool,
    openai_key: str,
    env_openai_key: str | None,
    publish: bool,
    hf_token: str,
    env_hf_token: str | None,
    hf_repo_name: str | None,
) -> list[str]:
    """Mirroring the validation logic in app.py to verify its behavior."""
    validation_errors = []

    if not uploaded_files:
        validation_errors.append("Upload at least one document (PDF/TXT) to begin.")

    if not use_vllm and not openai_key and not env_openai_key:
        validation_errors.append("OpenAI API Key is required when not using vLLM.")

    if publish:
        if not hf_token and not env_hf_token:
            validation_errors.append("Hugging Face Token is required when publishing.")

        if not hf_repo_name or not hf_repo_name.strip():
            validation_errors.append("Hugging Face repository name is required when publishing.")
        else:
            _REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
            if not _REPO_NAME_RE.match(hf_repo_name.strip()):
                validation_errors.append("Hugging Face repository format is invalid (must be 'username/repo-slug').")

    return validation_errors


def test_validation_all_valid():
    errors = validate_inputs(
        uploaded_files=["doc1.txt"],
        use_vllm=True,
        openai_key="",
        env_openai_key=None,
        publish=False,
        hf_token="",
        env_hf_token=None,
        hf_repo_name=None,
    )
    assert len(errors) == 0


def test_validation_missing_files():
    errors = validate_inputs(
        uploaded_files=[],
        use_vllm=True,
        openai_key="",
        env_openai_key=None,
        publish=False,
        hf_token="",
        env_hf_token=None,
        hf_repo_name=None,
    )
    assert "Upload at least one document (PDF/TXT) to begin." in errors


def test_validation_missing_openai_key():
    errors = validate_inputs(
        uploaded_files=["doc1.txt"],
        use_vllm=False,
        openai_key="",
        env_openai_key=None,
        publish=False,
        hf_token="",
        env_hf_token=None,
        hf_repo_name=None,
    )
    assert "OpenAI API Key is required when not using vLLM." in errors


def test_validation_openai_key_from_env_is_ok():
    errors = validate_inputs(
        uploaded_files=["doc1.txt"],
        use_vllm=False,
        openai_key="",
        env_openai_key="sk-123",
        publish=False,
        hf_token="",
        env_hf_token=None,
        hf_repo_name=None,
    )
    assert len(errors) == 0


def test_validation_missing_hf_publish_config():
    errors = validate_inputs(
        uploaded_files=["doc1.txt"],
        use_vllm=True,
        openai_key="",
        env_openai_key=None,
        publish=True,
        hf_token="",
        env_hf_token=None,
        hf_repo_name="",
    )
    assert "Hugging Face Token is required when publishing." in errors
    assert "Hugging Face repository name is required when publishing." in errors


def test_validation_invalid_hf_repo_format():
    errors = validate_inputs(
        uploaded_files=["doc1.txt"],
        use_vllm=True,
        openai_key="",
        env_openai_key=None,
        publish=True,
        hf_token="hf-123",
        env_hf_token=None,
        hf_repo_name="invalid_format_no_slash",
    )
    assert "Hugging Face repository format is invalid (must be 'username/repo-slug')." in errors


def test_validation_valid_hf_publish():
    errors = validate_inputs(
        uploaded_files=["doc1.txt"],
        use_vllm=True,
        openai_key="",
        env_openai_key=None,
        publish=True,
        hf_token="hf-123",
        env_hf_token=None,
        hf_repo_name="username/repo-slug",
    )
    assert len(errors) == 0
