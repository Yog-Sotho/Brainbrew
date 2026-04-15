"""Backward-compatible setup.py — delegates to pyproject.toml via setuptools."""
from setuptools import setup, find_packages

setup(
    name="brainbrew",
    version="1.2.0",
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    python_requires=">=3.12",
    install_requires=[
        "streamlit==1.42.0",
        "pydantic==2.10.6",
        "python-dotenv==1.0.1",
        "structlog==25.1.0",
        "pdfminer.six==20240706",
        "langchain-text-splitters==0.3.6",
        "tiktoken==0.8.0",
        "openai>=1.50.0,<2.0",
        "distilabel[openai,vllm]==1.5.2",
        "datasets==3.2.0",
        "huggingface_hub==0.28.1",
        "pandas==2.2.3",
        "numpy==2.2.1",
        "vllm==0.19.0",
        "unsloth==2024.12.7",
        "trl==0.12.2",
        "transformers==4.48.0",
        "accelerate==1.2.1",
        "bitsandbytes==0.45.0",
        "sentencepiece==0.2.0",
        "tenacity==9.0.0",
        "tqdm==4.67.1",
        "rich>=10.14.0,<14",
    ],
    entry_points={
        "console_scripts": [
            "brainbrew=app:main",
        ],
    },
)
