from huggingface_hub import HfApi
from datasets import load_dataset


def publish_dataset(dataset_path, repo_name):

    dataset = load_dataset(
        "json",
        data_files=dataset_path
    )

    dataset.push_to_hub(repo_name)
