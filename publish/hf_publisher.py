from huggingface_hub import HfApi
from datasets import load_dataset
import os

def publish_dataset(dataset_path: str, repo_name: str, token: str | None = None, private: bool = True) -> None:
    # FIX: validate token is present before attempting any Hub operations
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("A Hugging Face token is required. Set HF_TOKEN in your environment.")

    api = HfApi(token=token)

    # FIX: create repo if it doesn't exist; always private by default
    if not api.repo_exists(repo_id=repo_name, repo_type="dataset"):
        api.create_repo(repo_id=repo_name, repo_type="dataset", private=private, exist_ok=True)

    dataset = load_dataset("json", data_files=dataset_path)
    dataset.push_to_hub(repo_name, token=token, private=private)
