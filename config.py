"""
Brainbrew configuration — Pydantic-validated settings for the distillation pipeline.

Provides DistillationConfig (all pipeline parameters), QualityMode (fast/balanced/research),
QUALITY_MODE_LABELS (friendly display names for the Streamlit UI), and OutputFormat
(alpaca/sharegpt/chatml/openai).
"""
from __future__ import annotations

from enum import Enum
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo


class QualityMode(str, Enum):
    """Controls the depth of Evol-Instruct evolution passes."""
    FAST = "fast"
    BALANCED = "balanced"
    RESEARCH = "research"


class OutputFormat(str, Enum):
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

OUTPUT_FORMAT_LABELS: dict[OutputFormat, str] = {
    OutputFormat.ALPACA:  "Alpaca (instruction / input / output)",
    OutputFormat.SHAREGPT: "ShareGPT (conversations)",
    OutputFormat.CHATML:  "ChatML (messages array)",
    OutputFormat.OPENAI:  "OpenAI fine-tuning (messages JSONL)",
}


class DistillationConfig(BaseModel):
    """Type-safe, validated pipeline configuration."""

    teacher_model: str = Field(..., description="Model name or comma-separated list for multi-model ensemble")
    judge_model: Optional[str] = "gpt-4o-mini"
    dataset_size: int = Field(2000, ge=100, le=50000)
    quality_mode: QualityMode = QualityMode.BALANCED
    output_format: OutputFormat = OutputFormat.ALPACA
    use_vllm: bool = True
    train_model: bool = False
    base_model: str = "unsloth/mistral-7b-bnb-4bit"
    publish_dataset: bool = False
    hf_repo: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(2048, ge=128)
    batch_size: int = Field(64, ge=1)
    lora_rank: int = Field(16, ge=8)
    api_key: Optional[str] = None
    hf_token: Optional[str] = None
    use_semantic_chunking: bool = False
    enable_dedup: bool = True
    checkpoint_dir: Optional[str] = None
    sanitize_dataset: bool = False

    @field_validator("api_key", "hf_token")
    @classmethod
    def validate_secrets(cls, v: Optional[str]) -> Optional[str]:
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

    @field_validator("teacher_model", "base_model", "judge_model")
    @classmethod
    def validate_model_names(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
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
        if ".." in v_stripped or v_stripped.startswith("/") or v_stripped.startswith("\\"):
            raise ValueError("Model name cannot contain path traversal or absolute local paths.")
        if not re.match(r"^[a-zA-Z0-9_\-. /@,:]+$", v_stripped):
            raise ValueError("Model name contains invalid characters.")
        return v_stripped

    @field_validator("checkpoint_dir")
    @classmethod
    def validate_checkpoint_dir(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                return None
            if len(v_stripped) > 512:
                raise ValueError("Checkpoint directory path exceeds maximum allowed length of 512 characters.")
            if ".." in v_stripped:
                raise ValueError("Checkpoint directory path cannot contain path traversal sequences ('..').")
            if any(ord(c) < 32 or ord(c) > 126 for c in v_stripped):
                raise ValueError("Checkpoint directory path contains invalid or control characters.")
            return v_stripped
        return v

    @field_validator("hf_repo")
    @classmethod
    def validate_hf_repo(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_stripped = v.strip()
            if not v_stripped:
                return None
            if ".." in v_stripped:
                raise ValueError(
                    f"Invalid Hugging Face repository name format: {v_stripped!r}. "
                    "Cannot contain path traversal sequences ('..')."
                )
            _REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
            if not _REPO_NAME_RE.match(v_stripped):
                raise ValueError(
                    f"Invalid Hugging Face repository name format: {v_stripped!r}. "
                    "Must be in 'username/repo-slug' format."
                )
            return v_stripped
        return v

    @model_validator(mode="after")
    def validate_publish_config(self) -> DistillationConfig:
        if self.publish_dataset:
            if not self.hf_repo or not self.hf_repo.strip():
                raise ValueError("hf_repo is required when publish_dataset is enabled")
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

    # ── FIX C-02: prevent API key from leaking in repr / str ─────────────
    def __repr__(self) -> str:
        safe = self.safe_dict()
        fields = ", ".join(f"{k}={v!r}" for k, v in safe.items())
        return f"DistillationConfig({fields})"

    def __str__(self) -> str:
        return self.__repr__()
