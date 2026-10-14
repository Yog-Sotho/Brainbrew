# Quick start

Brainbrew needs Python 3.12 or 3.13. A GPU is only needed for a local model
server or LoRA training.

## Install

```bash
git clone https://github.com/Yog-Sotho/Brainbrew.git
cd Brainbrew
uv sync                         # or: bash install.sh
cp .env.sample .env             # then add your keys
```

Optional extras:

| Extra | Adds |
|---|---|
| `uv sync --extra train` | LoRA training with TRL + PEFT |
| `uv sync --extra pii` | Presidio name detection (then `python -m spacy download en_core_web_lg`) |

Without uv: `pip install --require-hashes -r requirements.txt` (or
`requirements-train.txt` for core + LoRA training).

A local GPU model server is not an extra: install vLLM in its own environment with
`uv tool install --python 3.12 vllm==0.31.0` (Linux, NVIDIA), let `install.sh` do it,
or run the official image with `docker compose` (see [Docker](docker.md)).

## Run the app

```bash
uv run streamlit run app.py
```

Open <http://localhost:8501>, upload a document, pick an endpoint and a model,
choose a target size and click **Generate Dataset**.

No API key? Start a local model server and pick **Local vLLM server** in the
sidebar:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507    # or: bash run_vllm.sh
```

## Run from the command line

```bash
export OPENAI_API_KEY=sk-...
uv run brainbrew run notes.pdf --size 100 --format sharegpt -o dataset.jsonl
```

See [Command line](cli.md) for config files and run inspection.
