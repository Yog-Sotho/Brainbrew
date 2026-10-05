# syntax=docker/dockerfile:1
#
# Brainbrew images, built from the locked dependency set (uv.lock).
#
#   docker build -t brainbrew .                         # GPU image (`vllm serve` + LoRA training), default
#   docker build --target api -t brainbrew-api .        # CPU image: any OpenAI-compatible endpoint
#   docker build --build-arg GPU_EXTRAS=vllm -t brainbrew-vllm .    # GPU image without training
#
# Generation talks to an OpenAI-compatible server. compose.yaml runs the CPU
# image next to the official vLLM server image, which is the simplest local
# setup; the GPU image is only needed for LoRA training inside the container.
#
#   docker run --gpus all -p 127.0.0.1:8501:8501 --env-file .env \
#       -v brainbrew-runs:/app/runs brainbrew
#
# The GPU image needs an NVIDIA driver with CUDA 13.0 support (R580+) and the
# NVIDIA Container Toolkit. Generated datasets and adapters are written to
# /app/runs; mount a volume there to keep them across container restarts.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21

FROM ${UV_IMAGE} AS uv

# ── API (CPU) image ──────────────────────────────────────────────────────────

FROM ${PYTHON_IMAGE} AS api-builder
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

FROM ${PYTHON_IMAGE} AS api
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
COPY --from=api-builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app app.py config.py orchestrator.py ./
COPY --chown=app:app engine/ engine/
COPY --chown=app:app ui/ ui/
COPY --chown=app:app pages/ pages/
COPY --chown=app:app pipeline/ pipeline/
COPY --chown=app:app publish/ publish/
COPY --chown=app:app training/ training/
COPY --chown=app:app .streamlit/config.toml .streamlit/config.toml
RUN install -d -o app -g app /app/runs
USER app
# Listen on all interfaces *inside* the container only; publish the port on
# the host as 127.0.0.1:8501 unless the app is behind auth (see README).
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    BRAINBREW_LOG_FORMAT=json \
    BRAINBREW_RUNS_DIR=/app/runs
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"]
CMD ["streamlit", "run", "app.py"]

# ── GPU image (default target) ───────────────────────────────────────────────

FROM ${CUDA_IMAGE} AS gpu-base
# Ubuntu 24.04 ships Python 3.12. gcc + headers are runtime requirements:
# Triton (used by vLLM / torch.compile) JIT-compiles a C launcher stub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

FROM gpu-base AS gpu-builder
ARG GPU_EXTRAS="vllm train"
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PYTHON=python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project $(printf -- '--extra %s ' ${GPU_EXTRAS})

FROM gpu-base AS gpu
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
COPY --from=gpu-builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app app.py config.py orchestrator.py ./
COPY --chown=app:app engine/ engine/
COPY --chown=app:app ui/ ui/
COPY --chown=app:app pages/ pages/
COPY --chown=app:app pipeline/ pipeline/
COPY --chown=app:app publish/ publish/
COPY --chown=app:app training/ training/
COPY --chown=app:app .streamlit/config.toml .streamlit/config.toml
RUN install -d -o app -g app /app/runs
USER app
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    BRAINBREW_LOG_FORMAT=json \
    BRAINBREW_RUNS_DIR=/app/runs
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"]
CMD ["streamlit", "run", "app.py"]
