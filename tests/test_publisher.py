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

@pytest.fixture
def fake_dataset_path(tmp_path: Path) -> str:
    p = tmp_path / "alpaca.jsonl"
    p.write_text('{"instruction": "Q?", "input": "", "output": "A."}\n', encoding="utf-8")
    return str(p)


@pytest.fixture
def mock_hf_api():
    api = MagicMock(name="HfApi_instance")
    api.repo_exists.return_value = True
    return api


@pytest.fixture
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
# Repo name validation
# ---------------------------------------------------------------------------

class TestRepoNameValidation:

    def test_invalid_repo_name_raises_valueerror(self, fake_dataset_path):
        from publish.hf_publisher import publish_dataset
        with pytest.raises(ValueError, match="Invalid Hugging Face repository name"):
            publish_dataset(fake_dataset_path, "no-slash-here", token="hf_test")

    def test_repo_name_with_spaces_raises(self, fake_dataset_path):
        from publish.hf_publisher import publish_dataset
        with pytest.raises(ValueError, match="Invalid Hugging Face repository name"):
            publish_dataset(fake_dataset_path, "user/my dataset", token="hf_test")

    def test_empty_repo_name_raises(self, fake_dataset_path):
        from publish.hf_publisher import publish_dataset
        with pytest.raises(ValueError, match="Invalid Hugging Face repository name"):
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


# ---------------------------------------------------------------------------
# Phase 3.3: dataset card, adapter upload
# ---------------------------------------------------------------------------

class TestDatasetCard:

    def test_card_uploaded_before_the_data(self, fake_dataset_path, mock_hf_api, mock_dataset):
        order = []
        mock_hf_api.upload_file.side_effect = lambda **kw: order.append(("card", kw["path_in_repo"]))
        mock_dataset.push_to_hub.side_effect = lambda *a, **kw: order.append(("data", a[0]))
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/repo", token="hf_test", card="---\nlicense: mit\n---\n# x")
        assert order == [("card", "README.md"), ("data", "user/repo")]
        assert mock_hf_api.upload_file.call_args.kwargs["path_or_fileobj"] == b"---\nlicense: mit\n---\n# x"

    def test_no_card_no_upload(self, fake_dataset_path, mock_hf_api, mock_dataset):
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api), \
             patch("publish.hf_publisher.load_dataset", return_value=mock_dataset):
            from publish.hf_publisher import publish_dataset
            publish_dataset(fake_dataset_path, "user/repo", token="hf_test")
        mock_hf_api.upload_file.assert_not_called()

    def test_card_contents(self, tmp_path, fake_dataset_path):
        import yaml

        from publish.dataset_card import dataset_card

        source = tmp_path / "source.txt"
        source.write_text("secret-plan.pdf contents", encoding="utf-8")
        manifest = {
            "run_id": "20261005-120000-abcdef", "brainbrew_version": "2.0.0", "seed": 7,
            "config": {"output_format": "sharegpt", "quality_mode": "balanced", "judge_threshold": 4,
                       "temperature": 0.7},
            "models": {"teacher": ["gpt-4o-mini"], "judge": "gpt-4o", "embedding": None},
            "counts": {"chunks": 9, "exported": 1500},
            "generation": {"questions_generated": 2000, "questions_dropped": 50, "answers": 1900,
                           "answers_filtered": {"refusal": 3}, "judge_rejected": 300, "duplicates": 47},
            "decontamination": {"gsm8k": 2},
            "sanitizer": {"total": 1502, "kept": 1500, "pii_redacted": 12},
            "quality": {"grade": "GOOD", "avg_output_length": 640, "unique_ratio": 0.99},
        }
        card = dataset_card("user/my-data", manifest, Path(fake_dataset_path), source, "cc-by-4.0")
        front = yaml.safe_load(card.split("---\n")[1])
        assert front["license"] == "cc-by-4.0" and front["size_categories"] == ["1K<n<10K"]
        assert "sharegpt" in front["tags"] and front["pretty_name"] == "my-data"
        for expected in ("| Records | 1500 |", "gpt-4o (kept pairs scoring at least 4 / 5",
                         "| Answers removed: refusal | 3 |", "| Overlap with gsm8k removed | 2 |",
                         "| Records with PII redacted | 12 |", "| Seed | 7 |", '"instruction": "Q?"'):
            assert expected in card, expected
        import hashlib
        assert hashlib.sha256(source.read_bytes()).hexdigest() in card
        assert "secret-plan" not in card  # source text and names are never published

    def test_fast_mode_card_has_no_judge(self, fake_dataset_path):
        from publish.dataset_card import dataset_card

        card = dataset_card("user/x", {"models": {"teacher": ["m"]}, "counts": {"exported": 5}},
                            Path(fake_dataset_path))
        assert "none (Fast mode" in card and "Rejected by the judge" not in card

    @pytest.mark.parametrize(("n", "label"), [(5, "n<1K"), (999, "n<1K"), (1000, "1K<n<10K"), (50_000, "10K<n<100K"),
                                         (200_000, "100K<n<1M")])
    def test_size_category(self, n, label):
        from publish.dataset_card import size_category
        assert size_category(n) == label


class TestPublishAdapter:

    def test_uploads_folder_and_card(self, tmp_path, mock_hf_api):
        (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        with patch("publish.hf_publisher.HfApi", return_value=mock_hf_api):
            from publish.hf_publisher import ADAPTER_IGNORE, publish_adapter
            publish_adapter(tmp_path, "user/my-lora", token="hf_test", card="# card")
        mock_hf_api.create_repo.assert_called_once_with(repo_id="user/my-lora", repo_type="model",
                                                        private=True, exist_ok=True)
        assert mock_hf_api.upload_folder.call_args.kwargs["ignore_patterns"] == ADAPTER_IGNORE
        assert mock_hf_api.upload_file.call_args.kwargs["path_in_repo"] == "README.md"

    def test_missing_adapter(self, tmp_path):
        from publish.hf_publisher import publish_adapter
        with pytest.raises(ValueError, match="No LoRA adapter"):
            publish_adapter(tmp_path, "user/my-lora", token="hf_test")

    def test_model_card(self):
        import yaml

        from publish.dataset_card import model_card

        card = model_card("user/my-lora", {"config": {"base_model": "Qwen/Qwen3-4B", "lora_rank": 32},
                                           "counts": {"exported": 120}}, "user/my-data", "mit")
        front = yaml.safe_load(card.split("---\n")[1])
        assert front == {"base_model": "Qwen/Qwen3-4B", "library_name": "peft", "license": "mit",
                         "tags": ["lora", "peft", "trl", "sft", "brainbrew"], "datasets": ["user/my-data"]}
        assert 'from_pretrained("user/my-lora")' in card and "rank 32" in card
