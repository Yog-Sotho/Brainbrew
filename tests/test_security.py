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
