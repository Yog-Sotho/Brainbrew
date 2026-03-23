"""
Hugging Face dataset publisher module.
Handles publishing datasets to Hugging Face Hub.
"""
from huggingface_hub import HfApi
from datasets import load_dataset
import os
import structlog
from typing import Optional

logger = structlog.get_logger(__name__)


def validate_token(token: Optional[str]) -> str:
    """
    Validate that a Hugging Face token is provided.

    Args:
        token: The Hugging Face token to validate

    Returns:
        The validated token

    Raises:
        ValueError: If token is missing or invalid
    """
    if not token:
        raise ValueError(
            "Hugging Face token is required for publishing. "
            "Please provide HF_TOKEN in .env file or as parameter."
        )

    # Basic token format validation (HF tokens start with 'hf_')
    if not token.startswith("hf_"):
        logger.warning("Token does not appear to be a valid Hugging Face token format")

    return token


def validate_repo_name(repo_name: str) -> str:
    """
    Validate the repository name format to prevent injection.

    Args:
        repo_name: The repository name to validate

    Returns:
        The validated repository name

    Raises:
        ValueError: If repository name is invalid
    """
    if not repo_name:
        raise ValueError("Repository name is required")

    # Must contain exactly one slash for username/repo format
    if "/" not in repo_name:
        raise ValueError(
            "Invalid repository name format. Expected: 'username/repo-name'. "
            f"Got: '{repo_name}'"
        )

    parts = repo_name.split("/")
    if len(parts) != 2:
        raise ValueError(
            "Invalid repository name format. Expected: 'username/repo-name'. "
            f"Got: '{repo_name}'"
        )

    username, reponame = parts

    # Validate username and repo name characters
    if not username or not reponame:
        raise ValueError("Username and repository name cannot be empty")

    # Hugging Face repo names should be alphanumeric with hyphens/underscores
    if not reponame.replace("-", "").replace("_", "").isalnum():
        raise ValueError(
            f"Invalid repository name '{reponame}'. "
            "Use only alphanumeric characters, hyphens, and underscores."
        )

    return repo_name


def validate_dataset_path(dataset_path: str) -> str:
    """
    Validate that the dataset file exists and is readable.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        The validated path

    Raises:
        ValueError: If path is invalid or file doesn't exist
    """
    if not dataset_path:
        raise ValueError("Dataset path is required")

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset file not found: {dataset_path}")

    if not os.path.isfile(dataset_path):
        raise ValueError(f"Dataset path is not a file: {dataset_path}")

    file_size = os.path.getsize(dataset_path)
    if file_size == 0:
        raise ValueError("Dataset file is empty")

    return dataset_path


def publish_dataset(dataset_path: str, repo_name: str, token: Optional[str] = None) -> str:
    """
    Publish a dataset to Hugging Face Hub.

    Args:
        dataset_path: Path to the local dataset file (JSONL format)
        repo_name: The target repository name (format: 'username/repo-name')
        token: Optional Hugging Face token (will use HF_TOKEN env var if not provided)

    Returns:
        The URL of the published dataset

    Raises:
        ValueError: If validation fails
        RuntimeError: If publishing fails
    """
    logger.info("Starting dataset publication", repo=repo_name, path=dataset_path)

    # Validate inputs
    token = validate_token(token or os.getenv("HF_TOKEN"))
    repo_name = validate_repo_name(repo_name)
    dataset_path = validate_dataset_path(dataset_path)

    try:
        # Load dataset from JSON file
        dataset = load_dataset("json", data_files=dataset_path)

        # Check dataset is not empty
        if hasattr(dataset, 'num_rows') and dataset.num_rows == 0:
            raise ValueError("Dataset is empty")

        logger.info("Dataset loaded successfully", num_rows=dataset.num_rows)

    except ValueError:
        raise
    except Exception as e:
        logger.exception("Failed to load dataset")
        raise RuntimeError(f"Failed to load dataset: {e}")

    try:
        # Push to Hub
        dataset.push_to_hub(repo_name, token=token, commit_message="Upload dataset via Brainbrew")

        # Construct the dataset URL
        dataset_url = f"https://huggingface.co/datasets/{repo_name}"

        logger.info("Dataset published successfully", url=dataset_url)
        return dataset_url

    except Exception as e:
        logger.exception("Failed to publish dataset")
        raise RuntimeError(f"Failed to publish dataset to {repo_name}: {e}")
