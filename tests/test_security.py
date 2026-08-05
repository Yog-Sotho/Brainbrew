"""
tests/test_security.py

Cross-cutting security tests covering:
  S-01: API key must never appear in logs, repr, str, or serialised config.
  S-02: Filename sanitisation must block path-traversal and shell-injection patterns.
  M-10: HF repo name validation.
"""
from __future__ import annotations

import re

import pytest

# ---------------------------------------------------------------------------
# S-01 — API key containment
# ---------------------------------------------------------------------------

class TestApiKeyContainment:

    SECRET = "sk-prod-key-abc123xyz789"

    @pytest.fixture()
    def cfg(self):
        from config import DistillationConfig
        return DistillationConfig(teacher_model="gpt-4o", api_key=self.SECRET)

    def test_repr_does_not_contain_key(self, cfg):
        assert self.SECRET not in repr(cfg)

    def test_str_does_not_contain_key(self, cfg):
        assert self.SECRET not in str(cfg)

    def test_safe_dict_redacts_key(self, cfg):
        safe = cfg.safe_dict()
        assert self.SECRET not in str(safe)

    def test_safe_dict_value_is_redacted_string(self, cfg):
        safe = cfg.safe_dict()
        assert safe["api_key"] == "***REDACTED***"

    def test_safe_dict_never_leaks(self, cfg):
        dumped = cfg.safe_dict()
        for v in dumped.values():
            assert self.SECRET not in str(v)

    def test_config_without_api_key_has_no_api_key_in_safe_dict(self):
        from config import DistillationConfig
        cfg = DistillationConfig(teacher_model="gpt-4o")
        safe = cfg.safe_dict()
        assert "api_key" not in safe

    @pytest.mark.parametrize("secret", [
        "sk-abc123",
        "hf_ABCDEFGHIJKLMNOP",
        "Bearer eyJhbGciOiJIUzI1NiJ9",
        "password123",
        "token_xyzABC",
    ])
    def test_various_secret_formats_redacted(self, secret):
        from config import DistillationConfig
        cfg = DistillationConfig(teacher_model="gpt-4o", api_key=secret)
        safe = cfg.safe_dict()
        assert secret not in repr(cfg)
        assert secret not in str(safe)


# ---------------------------------------------------------------------------
# S-02 — Filename sanitisation
# ---------------------------------------------------------------------------

_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")


class TestFilenameSanitisation:

    @pytest.mark.parametrize("filename", [
        "document.txt",
        "my-file.pdf",
        "report_2024.txt",
        "My Document v2.pdf",
        "data.PDF",
        "file123.txt",
        "some_long_file_name_with_underscores.pdf",
        "file with spaces.txt",
        "README.md",
    ])
    def test_safe_filename_accepted(self, filename):
        assert _SAFE_FILENAME_RE.match(filename), (
            f"'{filename}' should be accepted as safe but was rejected"
        )

    @pytest.mark.parametrize("filename", [
        "../etc/passwd",
        "../../secret.txt",
        "/etc/passwd",
        "file\x00name.txt",
        "file;rm -rf /.txt",
        "file`whoami`.txt",
        "file$(id).txt",
        "file|cat /etc/passwd.txt",
        "file>output.txt",
        "file<input.txt",
        "file&background.txt",
        r"C:\Windows\System32\cmd",
        "file\ninjection.txt",
        "file\tname.txt",
        "file'name.txt",
        'file"name.txt',
    ])
    def test_unsafe_filename_rejected(self, filename):
        assert not _SAFE_FILENAME_RE.match(filename), (
            f"'{filename}' should be REJECTED as unsafe but was accepted"
        )

    def test_empty_filename_rejected(self):
        assert not _SAFE_FILENAME_RE.match("")

    def test_regex_is_anchored(self):
        dangerous = "safe_prefix/../../../etc/passwd"
        assert not _SAFE_FILENAME_RE.match(dangerous)


# ---------------------------------------------------------------------------
# M-10 — HF repo name validation
# ---------------------------------------------------------------------------

class TestHfRepoNameValidation:

    def test_valid_repo_name_format(self):
        from publish.hf_publisher import _REPO_NAME_RE
        assert _REPO_NAME_RE.match("user/dataset")
        assert _REPO_NAME_RE.match("my-org/my_dataset-v2")
        assert _REPO_NAME_RE.match("user123/repo.name")

    def test_invalid_repo_name_no_slash(self):
        from publish.hf_publisher import _REPO_NAME_RE
        assert not _REPO_NAME_RE.match("just-a-name")

    def test_invalid_repo_name_double_slash(self):
        from publish.hf_publisher import _REPO_NAME_RE
        assert not _REPO_NAME_RE.match("user/sub/repo")

    def test_invalid_repo_name_empty(self):
        from publish.hf_publisher import _REPO_NAME_RE
        assert not _REPO_NAME_RE.match("")

    def test_invalid_repo_name_spaces(self):
        from publish.hf_publisher import _REPO_NAME_RE
        assert not _REPO_NAME_RE.match("user/my dataset")


# ---------------------------------------------------------------------------
# Combined invariants
# ---------------------------------------------------------------------------

class TestCombinedSecurityInvariants:

    def test_config_with_both_api_key_and_hf_repo_never_leaks_key(self):
        from config import DistillationConfig
        cfg = DistillationConfig(
            teacher_model="gpt-4o",
            api_key="sk-never-log-this",
            hf_repo="user/dataset",
            publish_dataset=True,
        )
        representation = repr(cfg) + str(cfg.safe_dict())
        assert "sk-never-log-this" not in representation

    def test_safe_dict_is_json_serialisable(self):
        import json

        from config import DistillationConfig
        cfg = DistillationConfig(teacher_model="gpt-4o", api_key="sk-secret")
        safe = cfg.safe_dict()
        try:
            json.dumps(safe)
        except (TypeError, ValueError) as e:
            pytest.fail(f"safe_dict() is not JSON-serialisable: {e}")


# ---------------------------------------------------------------------------
# S-03 — HF Token and Repo Name security checks
# ---------------------------------------------------------------------------

class TestHfTokenAndRepoSecurity:

    def test_hf_token_not_in_repr_or_str(self):
        from config import DistillationConfig
        token = "hf_secret_token_12345"
        cfg = DistillationConfig(teacher_model="gpt-4o", hf_token=token)
        assert token not in repr(cfg)
        assert token not in str(cfg)

    def test_hf_token_redacted_in_safe_dict(self):
        from config import DistillationConfig
        token = "hf_secret_token_12345"
        cfg = DistillationConfig(teacher_model="gpt-4o", hf_token=token)
        safe = cfg.safe_dict()
        assert safe["hf_token"] == "***REDACTED***"

    def test_hf_repo_validation_fails_on_invalid_format(self):
        from pydantic import ValidationError

        from config import DistillationConfig
        with pytest.raises(ValidationError, match="Invalid Hugging Face repository name format"):
            DistillationConfig(teacher_model="gpt-4o", hf_repo="invalid-repo-format-no-slash")

    def test_publish_dataset_requires_hf_repo(self):
        from pydantic import ValidationError

        from config import DistillationConfig
        with pytest.raises(ValidationError, match="hf_repo is required when publish_dataset is enabled"):
            DistillationConfig(teacher_model="gpt-4o", publish_dataset=True, hf_repo=None)

    @pytest.mark.parametrize("path_traversal_repo", [
        "user/..",
        "user/../repo",
        "../repo",
        "user/..repo",
    ])
    def test_hf_repo_validation_rejects_path_traversal(self, path_traversal_repo):
        from pydantic import ValidationError

        from config import DistillationConfig
        with pytest.raises(ValidationError, match="Cannot contain path traversal sequences"):
            DistillationConfig(teacher_model="gpt-4o", hf_repo=path_traversal_repo)

    @pytest.mark.parametrize("path_traversal_repo", [
        "user/..",
        "user/../repo",
        "../repo",
        "user/..repo",
    ])
    def test_publisher_rejects_path_traversal(self, path_traversal_repo, monkeypatch):
        from publish.hf_publisher import publish_dataset
        # Monkeypatch env var to avoid missing token error
        monkeypatch.setenv("HF_TOKEN", "mock_token")
        with pytest.raises(ValueError, match="Cannot contain path traversal sequences"):
            publish_dataset(dataset_path="mock.jsonl", repo_name=path_traversal_repo)


# ---------------------------------------------------------------------------
# S-04 — Secrets and Model Names Validation Security Checks
# ---------------------------------------------------------------------------

class TestInputValidationSecurity:

    def test_secrets_max_length(self):
        from pydantic import ValidationError

        from config import DistillationConfig
        long_key = "a" * 513
        with pytest.raises(ValidationError, match="Secret key/token exceeds maximum allowed length"):
            DistillationConfig(teacher_model="gpt-4o", api_key=long_key)
        with pytest.raises(ValidationError, match="Secret key/token exceeds maximum allowed length"):
            DistillationConfig(teacher_model="gpt-4o", hf_token=long_key)

    def test_secrets_control_chars(self):
        from pydantic import ValidationError

        from config import DistillationConfig
        bad_key = "sk-key\nwithnewlines"
        with pytest.raises(ValidationError, match="Secret key/token contains invalid or control characters"):
            DistillationConfig(teacher_model="gpt-4o", api_key=bad_key)

    def test_model_names_max_length(self):
        from pydantic import ValidationError

        from config import DistillationConfig
        long_model = "a" * 256
        with pytest.raises(ValidationError, match="Model name exceeds maximum allowed length"):
            DistillationConfig(teacher_model=long_model)
        with pytest.raises(ValidationError, match="Model name exceeds maximum allowed length"):
            DistillationConfig(teacher_model="gpt-4o", base_model=long_model)

    @pytest.mark.parametrize("bad_name", [
        "../../etc/passwd",
        "/absolute/path",
        "\\windows\\path",
        "some_model/../other",
        "C:/absolute/path",
        "D:\\windows\\path",
        "E:/some_model/../other",
    ])
    def test_model_names_path_traversal(self, bad_name):
        from pydantic import ValidationError

        from config import DistillationConfig
        with pytest.raises(ValidationError, match="Model name cannot contain path traversal or absolute local paths"):
            DistillationConfig(teacher_model=bad_name)
        with pytest.raises(ValidationError, match="Model name cannot contain path traversal or absolute local paths"):
            DistillationConfig(teacher_model="gpt-4o", base_model=bad_name)

    @pytest.mark.parametrize("invalid_name", [
        "model; rm -rf /",
        "model`whoami`",
        "model$(id)",
        "model|cat",
    ])
    def test_model_names_invalid_chars(self, invalid_name):
        from pydantic import ValidationError

        from config import DistillationConfig
        with pytest.raises(ValidationError, match="Model name contains invalid characters"):
            DistillationConfig(teacher_model=invalid_name)

    def test_judge_model_validation(self):
        from pydantic import ValidationError

        from config import DistillationConfig
        # Verify valid judge model is accepted
        cfg = DistillationConfig(teacher_model="gpt-4o", judge_model="gpt-4o-mini")
        assert cfg.judge_model == "gpt-4o-mini"

        # Verify invalid paths or traversals are rejected
        with pytest.raises(ValidationError, match="Model name cannot contain path traversal or absolute local paths"):
            DistillationConfig(teacher_model="gpt-4o", judge_model="../../etc/passwd")

        # Verify long names are rejected
        with pytest.raises(ValidationError, match="Model name exceeds maximum allowed length"):
            DistillationConfig(teacher_model="gpt-4o", judge_model="a" * 256)

        # Verify invalid characters are rejected
        with pytest.raises(ValidationError, match="Model name contains invalid characters"):
            DistillationConfig(teacher_model="gpt-4o", judge_model="model; rm -rf /")

    def test_checkpoint_dir_validation(self):
        from pydantic import ValidationError

        from config import DistillationConfig
        # Verify valid directory path is accepted
        cfg = DistillationConfig(teacher_model="gpt-4o", checkpoint_dir="checkpoints/run1")
        assert cfg.checkpoint_dir == "checkpoints/run1"

        # Verify path traversal is rejected
        with pytest.raises(ValidationError, match="Checkpoint directory path cannot contain path traversal sequences"):
            DistillationConfig(teacher_model="gpt-4o", checkpoint_dir="checkpoints/../secret")

        # Verify excessive length is rejected
        with pytest.raises(ValidationError, match="Checkpoint directory path exceeds maximum allowed length"):
            DistillationConfig(teacher_model="gpt-4o", checkpoint_dir="a" * 513)

        # Verify control characters are rejected
        with pytest.raises(ValidationError, match="Checkpoint directory path contains invalid or control characters"):
            DistillationConfig(teacher_model="gpt-4o", checkpoint_dir="checkpoints\nrun")
