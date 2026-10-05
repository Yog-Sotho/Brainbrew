"""
Brainbrew configuration — Pydantic-validated settings for the distillation pipeline.

Provides DistillationConfig (all pipeline parameters), QualityMode (fast/balanced/research),
QUALITY_MODE_LABELS (friendly display names for the Streamlit UI), and OutputFormat
(alpaca/sharegpt/chatml/openai).
"""
from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class QualityMode(StrEnum):
    """Controls the depth of Evol-Instruct evolution passes."""
    FAST = "fast"
    BALANCED = "balanced"
    RESEARCH = "research"


class OutputFormat(StrEnum):
    """Supported dataset export formats."""
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CHATML = "chatml"
    OPENAI = "openai"


# FIX C-07: app.py imports this dict for the selectbox display labels.
QUALITY_MODE_LABELS: dict[QualityMode, str] = {
    QualityMode.FAST:     "Fast ⚡ (quick & cheap)",
    QualityMode.BALANCED: "Balanced 🎯 (sweet spot)",
    QualityMode.RESEARCH: "Research 🔬 (maximum quality)",
}

# Hugging Face repo ids: "username/repo-slug". Shared with publish/hf_publisher.py.
HF_REPO_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def check_hf_repo_name(name: str) -> str:
    """Return the stripped repo id, or raise ValueError if it is not 'username/repo-slug'."""
    name = name.strip()
    if ".." in name:
        raise ValueError(
            f"Invalid Hugging Face repository name: {name!r}. "
            "Cannot contain path traversal sequences ('..')."
        )
    if not HF_REPO_RE.fullmatch(name):
        raise ValueError(
            f"Invalid Hugging Face repository name: {name!r}. "
            "Must be in 'username/repo-slug' format (letters, numbers, '-', '_', '.')."
        )
    return name


OUTPUT_FORMAT_LABELS: dict[OutputFormat, str] = {
    OutputFormat.ALPACA:  "Alpaca (instruction / input / output)",
    OutputFormat.SHAREGPT: "ShareGPT (conversations)",
    OutputFormat.CHATML:  "ChatML (messages array)",
    OutputFormat.OPENAI:  "OpenAI fine-tuning (messages JSONL)",
}


class DistillationConfig(BaseModel):
    """Type-safe, validated pipeline configuration."""

    teacher_model: str = Field(..., description="Model name or comma-separated list for multi-model ensemble")
    dataset_size: int = Field(2000, ge=100, le=50000)
    quality_mode: QualityMode = QualityMode.BALANCED
    output_format: OutputFormat = OutputFormat.ALPACA
    use_vllm: bool = True
    train_model: bool = False
    base_model: str = DEFAULT_BASE_MODEL
    publish_dataset: bool = False
    hf_repo: str | None = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(2048, ge=128, le=32768)
    batch_size: int = Field(64, ge=1, le=1024)
    lora_rank: int = Field(16, ge=4, le=256)
    api_key: str | None = None
    hf_token: str | None = None
    use_semantic_chunking: bool = False
    enable_dedup: bool = True
    sanitize_dataset: bool = False

    @field_validator("api_key", "hf_token")
    @classmethod
    def validate_secrets(cls, v: str | None) -> str | None:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                return None
            if len(v_stripped) > 512:
                raise ValueError("Secret key/token exceeds maximum allowed length of 512 characters.")
            if any(ord(c) < 32 or ord(c) > 126 for c in v_stripped):
                raise ValueError("Secret key/token contains invalid or control characters.")
            return v_stripped
        return v

    @field_validator("teacher_model", "base_model")
    @classmethod
    def validate_model_names(cls, v: str | None, info: ValidationInfo) -> str | None:
        if v is None:
            if info.field_name == "teacher_model":
                raise ValueError("Teacher model is required")
            return None
        v_stripped = v.strip()
        if not v_stripped:
            if info.field_name == "teacher_model":
                raise ValueError("Teacher model is required")
            raise ValueError("Model name is required")
        if len(v_stripped) > 255:
            raise ValueError("Model name exceeds maximum allowed length of 255 characters.")

        # Split and validate individual model names (e.g., for multi-model ensembles in teacher_model)
        names = [n.strip() for n in v_stripped.split(",")] if info.field_name == "teacher_model" else [v_stripped]
        for name in names:
            if not name:
                continue
            if ".." in name or name.startswith("/") or name.startswith("\\") or re.match(r"^[a-zA-Z]:", name):
                raise ValueError("Model name cannot contain path traversal or absolute local paths.")

        if not re.match(r"^[a-zA-Z0-9_\-. /@,:]+$", v_stripped):
            raise ValueError("Model name contains invalid characters.")
        return v_stripped

    @field_validator("hf_repo")
    @classmethod
    def validate_hf_repo(cls, v: str | None) -> str | None:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                return None
            return check_hf_repo_name(v_stripped)
        return v

    @model_validator(mode="after")
    def validate_cross_field(self) -> DistillationConfig:
        if not self.use_vllm and not self.api_key:
            raise ValueError(
                "An API key is required when not using vLLM "
                "(any non-empty value for a local OpenAI-compatible server)."
            )
        if self.publish_dataset:
            if not self.hf_repo:
                raise ValueError("hf_repo is required when publish_dataset is enabled")
            if not self.hf_token:
                raise ValueError("A Hugging Face token is required when publish_dataset is enabled")
        return self

    # ── FIX C-01: safe serialisation that never leaks secrets ────────────
    def safe_dict(self) -> dict:
        """Return model_dump with api_key and hf_token redacted. Safe for logging / display."""
        d = self.model_dump(exclude_none=True)
        if "api_key" in d:
            d["api_key"] = "***REDACTED***"
        if "hf_token" in d:
            d["hf_token"] = "***REDACTED***"
        return d

    def public_dict(self) -> dict[str, Any]:
        """JSON-safe settings with secrets removed entirely (for run manifests)."""
        return self.model_dump(mode="json", exclude={"api_key", "hf_token"})

    # ── FIX C-02: prevent API key from leaking in repr / str ─────────────
    def __repr__(self) -> str:
        safe = self.safe_dict()
        fields = ", ".join(f"{k}={v!r}" for k, v in safe.items())
        return f"DistillationConfig({fields})"

    def __str__(self) -> str:
        return self.__repr__()
