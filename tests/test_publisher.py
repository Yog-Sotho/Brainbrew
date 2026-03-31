"""
tests/test_publisher.py

Tests for publish/hf_publisher.py — publish_dataset().

All HuggingFace Hub calls are mocked — no network traffic, no token required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_dataset_path(tmp_path: Path) -> str:
    p = tmp_path / "alpaca.jsonl"
    p.write_text('{"instruction": "Q?", "input": "", "output": "A."}\n', encoding="utf-8")
    return str(p)


@pytest.fixture()
def mock_hf_api():
    api = MagicMock(name="HfApi_instance")
    api.repo_exists.return_value = True
    return api


@pytest.fixture()
def mock_dataset():
    return MagicMock(name="DatasetDict")


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------

class TestTokenValidation:

    def test_no_token_arg_no_env_raises_valueerror(self, fake_dataset_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        from publish.hf_publisher import publish_dataset
        with pytest.raises(ValueError, match="token"):
            publish_dataset(fake_dataset_path, "user/repo", token="")

    def test_none_token_no_env_raises_valueerror(self, fake_dataset_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        from publish.hf_publisher import publish_dataset
        with pytest.raises(ValueError, match="token"):
            publish_dataset(fake_dataset_path, "user/repo", token=None)

    def test_env_token_used_when_arg_is_none(self, fake_dataset_path, monkeypatch, mock_hf_api, mock_dataset):
        monkeypatch.setenv("HF_TOKEN", "hf_env_token")
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/repo", token=None)
        mock_hf_api.repo_exists.assert_called_once()

    def test_explicit_token_takes_precedence_over_env(self, fake_dataset_path, monkeypatch, mock_hf_api, mock_dataset):
        monkeypatch.setenv("HF_TOKEN", "hf_env_token")
        with patch("publish.hf_publisher.HfApi") as MockHfApi, \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            MockHfApi.return_value = mock_hf_api
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/repo", token="hf_explicit_token")
        MockHfApi.assert_called_once_with(token="hf_explicit_token")


# ---------------------------------------------------------------------------
# M-10: Repo name validation
# ---------------------------------------------------------------------------

class TestRepoNameValidation:

    def test_invalid_repo_name_raises_valueerror(self, fake_dataset_path):
        from publish.hf_publisher import publish_dataset
        with pytest.raises(ValueError, match="Invalid repo name"):
            publish_dataset(fake_dataset_path, "no-slash-here", token="hf_test")

    def test_repo_name_with_spaces_raises(self, fake_dataset_path):
        from publish.hf_publisher import publish_dataset
        with pytest.raises(ValueError, match="Invalid repo name"):
            publish_dataset(fake_dataset_path, "user/my dataset", token="hf_test")

    def test_empty_repo_name_raises(self, fake_dataset_path):
        from publish.hf_publisher import publish_dataset
        with pytest.raises(ValueError, match="Invalid repo name"):
            publish_dataset(fake_dataset_path, "", token="hf_test")

    def test_valid_repo_name_accepted(self, fake_dataset_path, mock_hf_api, mock_dataset):
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            # Should not raise
            publish_dataset(fake_dataset_path, "user/my-dataset-v2", token="hf_test")


# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------

class TestRepoManagement:

    def test_repo_exists_no_create_called(self, fake_dataset_path, mock_hf_api, mock_dataset):
        mock_hf_api.repo_exists.return_value = True
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/repo", token="hf_test")
        mock_hf_api.create_repo.assert_not_called()

    def test_repo_missing_create_repo_called(self, fake_dataset_path, mock_hf_api, mock_dataset):
        mock_hf_api.repo_exists.return_value = False
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/repo", token="hf_test")
        mock_hf_api.create_repo.assert_called_once()

    def test_new_repo_created_as_private_by_default(self, fake_dataset_path, mock_hf_api, mock_dataset):
        mock_hf_api.repo_exists.return_value = False
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/repo", token="hf_test")
        _, kwargs = mock_hf_api.create_repo.call_args
        assert kwargs.get("private") is True

    def test_private_false_respected(self, fake_dataset_path, mock_hf_api, mock_dataset):
        mock_hf_api.repo_exists.return_value = False
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/repo", token="hf_test", private=False)
        mock_dataset.push_to_hub.assert_called_once()
        _, kwargs = mock_dataset.push_to_hub.call_args
        assert kwargs.get("private") is False


# ---------------------------------------------------------------------------
# push_to_hub behaviour
# ---------------------------------------------------------------------------

class TestPushToHub:

    def test_push_to_hub_called_exactly_once(self, fake_dataset_path, mock_hf_api, mock_dataset):
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/my-dataset", token="hf_test")
        mock_dataset.push_to_hub.assert_called_once()

    def test_push_to_hub_receives_correct_repo_name(self, fake_dataset_path, mock_hf_api, mock_dataset):
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/my-dataset", token="hf_test")
        args, _ = mock_dataset.push_to_hub.call_args
        assert args[0] == "user/my-dataset"

    def test_load_dataset_called_with_correct_path(self, fake_dataset_path, mock_hf_api, mock_dataset):
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset) as mock_load:
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/repo", token="hf_test")
        mock_load.assert_called_once_with("json", data_files=fake_dataset_path)
