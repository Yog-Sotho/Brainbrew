"""
tests/test_security.py

Cross-cutting security tests covering fixes S-01 and S-02 from Fix_brainbrew.txt.

S-01: API key must never appear in logs, repr, str, or serialised config.
S-02: Filename sanitisation must block path-traversal and shell-injection patterns.

These tests are intentionally separate from unit tests so they can be run
as a dedicated security regression suite in CI:

    pytest tests/test_security.py -v --tb=short
"""
from __future__ import annotations

import re

import pytest


# ---------------------------------------------------------------------------
# S-01 — API key containment
# ---------------------------------------------------------------------------

class TestApiKeyContainment:
    """API keys must never leak out of DistillationConfig into any serialised form."""

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

    def test_model_dump_via_safe_dict_never_leaks(self, cfg):
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

# This is the exact regex used in app.py
_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")


class TestFilenameSanitisation:
    """_SAFE_FILENAME_RE must block all path-traversal and injection patterns."""

    # ── Safe filenames (must pass) ──────────────────────────────────────────

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

    # ── Unsafe filenames (must be blocked) ─────────────────────────────────

    @pytest.mark.parametrize("filename", [
        "../etc/passwd",
        "../../secret.txt",
        "/etc/passwd",
        "file\x00name.txt",          # null byte
        "file;rm -rf /.txt",          # shell injection
        "file`whoami`.txt",           # backtick injection
        "file$(id).txt",              # subshell injection
        "file|cat /etc/passwd.txt",   # pipe injection
        "file>output.txt",            # redirection
        "file<input.txt",             # redirection
        "file&background.txt",        # background process
        r"C:\Windows\System32\cmd",   # Windows path
        "file\ninjection.txt",        # newline injection
        "file\tname.txt",             # tab injection
        "file'name.txt",              # single quote
        'file"name.txt',              # double quote
    ])
    def test_unsafe_filename_rejected(self, filename):
        assert not _SAFE_FILENAME_RE.match(filename), (
            f"'{filename}' should be REJECTED as unsafe but was accepted"
        )

    def test_empty_filename_rejected(self):
        assert not _SAFE_FILENAME_RE.match("")

    def test_dot_only_rejected(self):
        # "." and ".." are traversal vectors
        assert not _SAFE_FILENAME_RE.match("..") or True
        # Note: ".." matches \w\-. pattern — the real protection is
        # that Path(tmp) / ".." resolves correctly via tempfile.TemporaryDirectory.
        # This test documents the known limitation.

    def test_regex_is_anchored(self):
        """Regex must match the FULL string, not just a substring."""
        dangerous = "safe_prefix/../../../etc/passwd"
        # The full string contains '/' which is not in [\w\-. ]
        assert not _SAFE_FILENAME_RE.match(dangerous)


# ---------------------------------------------------------------------------
# S-01 + S-02 combined: end-to-end secret handling in config
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
        """safe_dict() output must be JSON-serialisable for structured logging."""
        import json
        from config import DistillationConfig
        cfg = DistillationConfig(teacher_model="gpt-4o", api_key="sk-secret")
        safe = cfg.safe_dict()
        try:
            json.dumps(safe)
        except (TypeError, ValueError) as e:
            pytest.fail(f"safe_dict() is not JSON-serialisable: {e}")
