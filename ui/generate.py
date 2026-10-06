"""
Logic behind the Generate page that needs no widgets: the endpoint choices,
cost and time estimates, readable validation messages and upload checks.
Kept apart from app.py so it can be unit-tested without AppTest.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from pipeline.pricing import estimate_cost

CUSTOM = "custom"
MAX_WARN_BYTES: int = 10 * 1024 * 1024   # warn at 10 MB
MAX_HARD_BYTES: int = 50 * 1024 * 1024   # hard limit at 50 MB per file

URL_POLICY_LABELS = {
    "domain": "Keep the site, drop the path",
    "redact": "Remove links",
    "keep": "Keep links",
}


def server_endpoint_label(server_base_url: str | None) -> str:
    return "Server default" if server_base_url else "OpenAI API"


def endpoint_options(server_base_url: str | None, allow_custom: bool) -> dict[str, str | None]:
    """Sidebar endpoint label -> base URL (None: OpenAI; CUSTOM: ask for a URL).

    The server's own endpoint comes first; the others only when custom
    endpoints are allowed (see ui.common.custom_endpoints_allowed).
    """
    options: dict[str, str | None] = {server_endpoint_label(server_base_url): server_base_url}
    if allow_custom:
        options.update({
            "Local vLLM server (localhost:8000)": "http://localhost:8000/v1",
            "Ollama (localhost:11434)": "http://localhost:11434/v1",
            "Custom URL…": CUSTOM,
        })
    return options


# Tokens per *accepted* pair, including over-generation, question writing,
# answering, and (Balanced/Research) judging and evolving.
TOKENS_PER_PAIR: dict[str, int] = {"fast": 1500, "balanced": 2600, "research": 3800}


def estimate(model: str, size: int, mode: str, local: bool) -> tuple[str, str]:
    """(cost, time) estimates for the info bar."""
    total_tokens = size * TOKENS_PER_PAIR.get(mode, 2600)
    minutes = max(1, round(total_tokens / 60_000))  # ~1k tokens/s across parallel requests
    if local:
        return "Free (your server)", f"~{minutes} min"
    cost = estimate_cost(model.split(",")[0].strip(), total_tokens, local) or 0.0
    return f"~${cost:.2f}", f"~{minutes} min"


FIELD_LABELS: dict[str, str] = {
    "teacher_model": "Teacher model",
    "judge_model": "Judge model",
    "embedding_model": "Embedding model",
    "base_model": "Base model",
    "base_url": "Endpoint URL",
    "hf_repo": "Hugging Face repo",
    "hf_model_repo": "Adapter repo",
    "api_key": "API key",
    "hf_token": "Hugging Face token",
    "dataset_size": "Dataset size",
    "temperature": "Temperature",
    "max_new_tokens": "Max answer length",
    "concurrency": "Parallel requests",
    "request_timeout": "Request timeout",
    "lora_rank": "LoRA rank",
    "decontaminate": "Benchmarks",
}


def friendly_errors(exc: ValidationError) -> list[str]:
    """One readable line per pydantic error, prefixed with the field's UI label."""
    messages = []
    for err in exc.errors():
        field = str(err["loc"][0]) if err["loc"] else ""
        label = FIELD_LABELS.get(field)
        for msg in str(err["msg"]).removeprefix("Value error, ").splitlines():
            messages.append(f"{label}: {msg}" if label else msg)
    return messages


def upload_errors(files: Sequence[Any] | None, source_text: str, read_errors: Sequence[str]) -> list[str]:
    """Problems with the uploaded documents that block a run."""
    if not files:
        return ["Upload at least one document (PDF/TXT) to begin."]
    errors = [
        f"File '{f.name}' exceeds the 50 MB hard size limit ({f.size / 1e6:.1f} MB)."
        for f in files
        if (getattr(f, "size", 0) or 0) > MAX_HARD_BYTES
    ]
    if not read_errors and not source_text.strip():
        errors.append("No text could be extracted from the uploaded documents.")
    return errors
