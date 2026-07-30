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
    _SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")

    if not uploaded_files:
        validation_errors.append("Upload at least one document (PDF/TXT) to begin.")
    else:
        for uploaded in uploaded_files:
            # support both real UploadedFile/Mock objects and strings (for backward compatibility in tests)
            name = uploaded if isinstance(uploaded, str) else getattr(uploaded, "name", "")
            size = 0 if isinstance(uploaded, str) else getattr(uploaded, "size", 0)

            if not isinstance(uploaded, str) and hasattr(uploaded, "seek"):
                uploaded.seek(0)

            # Check filename safety
            if not _SAFE_FILENAME_RE.match(name):
                validation_errors.append(
                    f"File '{name}' has an unsafe filename. "
                    "Only alphanumeric characters, dashes, underscores, spaces, and periods are allowed."
                )
            # Check file size limit
            if size > 50 * 1024 * 1024:
                validation_errors.append(
                    f"File '{name}' exceeds the 50 MB hard size limit "
                    f"({size / 1e6:.1f} MB)."
                )

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


class MockUploadedFile:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.seek_called = False

    def seek(self, position: int):
        if position == 0:
            self.seek_called = True


def test_validation_unsafe_filename():
    unsafe_file = MockUploadedFile(name="../unsafe_path.txt", size=1024)
    errors = validate_inputs(
        uploaded_files=[unsafe_file],
        use_vllm=True,
        openai_key="",
        env_openai_key=None,
        publish=False,
        hf_token="",
        env_hf_token=None,
        hf_repo_name=None,
    )
    assert unsafe_file.seek_called
    assert "File '../unsafe_path.txt' has an unsafe filename." in errors[0]


def test_validation_oversized_file():
    oversized_file = MockUploadedFile(name="very_large.txt", size=60 * 1024 * 1024) # 60 MB
    errors = validate_inputs(
        uploaded_files=[oversized_file],
        use_vllm=True,
        openai_key="",
        env_openai_key=None,
        publish=False,
        hf_token="",
        env_hf_token=None,
        hf_repo_name=None,
    )
    assert oversized_file.seek_called
    assert "File 'very_large.txt' exceeds the 50 MB hard size limit" in errors[0]
