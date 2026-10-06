"""
Brainbrew HF publisher — push generated datasets (and LoRA adapters) to the Hugging Face Hub.

Repos are created private unless the caller asks otherwise. A dataset card
is uploaded before the data: `datasets.push_to_hub` keeps an existing
README.md and only merges its own metadata (configs, dataset_info) into it.
"""
from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from config import check_hf_repo_name

# Training leaves optimizer state and checkpoints next to the adapter; only the adapter is published.
ADAPTER_IGNORE = ["checkpoint-*", "checkpoint-*/**", "runs/**", "*.pt", "*.bin.index.json", "training_args.bin"]


def _token(token: str | None) -> str:
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "A Hugging Face token is required. Set HF_TOKEN in your environment."
        )
    return token


def publish_dataset(
    dataset_path: str,
    repo_name: str,
    token: str | None = None,
    private: bool = True,
    card: str | None = None,
) -> None:
    """Upload a JSONL dataset to the Hugging Face Hub.

    Args:
        dataset_path: Path to the JSONL file on disk.
        repo_name: HF repo in 'username/repo-slug' format.
        token: HF API token (falls back to HF_TOKEN env var).
        private: If True, create as a private dataset repo.
        card: README.md text for the repo (see publish/dataset_card.py).

    Raises:
        ValueError: If token is missing or repo_name format is invalid.
    """
    token = _token(token)
    # Same repo-name rule as the UI and DistillationConfig.
    repo_name = check_hf_repo_name(repo_name)

    api = HfApi(token=token)

    # Create repo if it doesn't exist; always private by default
    if not api.repo_exists(repo_id=repo_name, repo_type="dataset"):
        api.create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    if card is not None:
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            commit_message="Brainbrew: dataset card",
        )

    dataset = load_dataset("json", data_files=dataset_path)
    dataset.push_to_hub(repo_name, token=token, private=private)


def publish_adapter(
    adapter_dir: Path,
    repo_name: str,
    token: str | None = None,
    private: bool = True,
    card: str | None = None,
) -> None:
    """Upload a trained LoRA adapter folder as a model repo."""
    token = _token(token)
    repo_name = check_hf_repo_name(repo_name)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError(f"No LoRA adapter found in {adapter_dir}")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_name, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_name,
        repo_type="model",
        ignore_patterns=ADAPTER_IGNORE,
        commit_message="Brainbrew: LoRA adapter",
    )
    if card is not None:
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Brainbrew: model card",
        )
