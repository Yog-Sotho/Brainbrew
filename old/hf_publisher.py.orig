from huggingface_hub import HfApi
from datasets import load_dataset
import os

def publish_dataset(dataset_path: str, repo_name: str, token: str | None = None) -> None:
    token = token or os.getenv("HF_TOKEN")
    dataset = load_dataset("json", data_files=dataset_path)
    dataset.push_to_hub(repo_name, token=token)
