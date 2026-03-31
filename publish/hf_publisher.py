"""
Brainbrew HF publisher — push generated datasets to the Hugging Face Hub.
"""
from __future__ import annotations

import os
import re

from datasets import load_dataset
from huggingface_hub import HfApi

# FIX M-10: repo name must be username/slug format
_REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


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

    # FIX M-10: validate repo name format
    if not _REPO_NAME_RE.match(repo_name):
        raise ValueError(
            f"Invalid repo name: {repo_name!r}. "
            "Must be in 'username/repo-slug' format (letters, numbers, hyphens, dots, underscores)."
        )

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
