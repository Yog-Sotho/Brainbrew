# Docker

## Images

| Image | Build | For |
|---|---|---|
| CPU (`api` target) | `docker build --target api -t brainbrew-api .` | OpenAI or any OpenAI-compatible endpoint; the CLI |
| GPU (default) | `docker build -t brainbrew .` | `vllm serve` and LoRA training (NVIDIA driver R580+, CUDA 13.0) |
| GPU without training | `docker build --build-arg GPU_EXTRAS=vllm -t brainbrew-vllm .` | `vllm serve` only |

Images run as a non-root user (uid 10001), log JSON lines, and never contain
`.env` or `.streamlit/secrets.toml` (`.dockerignore` is an allow-list).

```bash
docker run -p 127.0.0.1:8501:8501 --env-file .env -v brainbrew-runs:/app/runs brainbrew-api
```

Mount a volume at `/app/runs` to keep runs across restarts.

## Brainbrew with a local vLLM server

`compose.yaml` runs the CPU app next to the official vLLM server image:

```bash
cp .env.sample .env       # HF_TOKEN only for gated models or publishing
docker compose up -d      # model: VLLM_MODEL (default Qwen/Qwen3-4B-Instruct-2507)
```

The app can only reach the vLLM container (`BRAINBREW_ALLOW_CUSTOM_ENDPOINTS=0`),
and vLLM's port is not published on the host. The first start downloads the
model, so the health check allows ten minutes.

## Headless runs

```bash
docker run --rm -v "$PWD:/data" -v brainbrew-runs:/app/runs -e OPENAI_API_KEY brainbrew-api \
    python cli.py run /data/notes.pdf -o /data/dataset.jsonl
```
