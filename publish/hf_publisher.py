"""
Brainbrew HF publisher — push generated datasets to the Hugging Face Hub.
"""
from __future__ import annotations

import os

from datasets import load_dataset
from huggingface_hub import HfApi

from config import check_hf_repo_name


def publish_dataset(
    dataset_path: str,
    repo_name: str,
    token: str | None = None,
    private: bool = True,
) -> None:
    """Upload a JSONL dataset to the Hugging Face Hub.

    Args:
        dataset_path: Path to the JSONL file on disk.
        repo_name: HF repo in 'username/repo-slug' format.
        token: HF API token (falls back to HF_TOKEN env var).
        private: If True, create as a private dataset repo.

    Raises:
        ValueError: If token is missing or repo_name format is invalid.
    """
    # Validate token
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "A Hugging Face token is required. Set HF_TOKEN in your environment."
        )

    # FIX M-10: same repo-name rule as the UI and DistillationConfig
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

    dataset = load_dataset("json", data_files=dataset_path)
    dataset.push_to_hub(repo_name, token=token, private=private)
