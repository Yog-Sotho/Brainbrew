"""
Brainbrew configuration — Pydantic-validated settings for the distillation pipeline.

Provides DistillationConfig (all pipeline parameters), QualityMode (fast/balanced/research),
QUALITY_MODE_LABELS (friendly display names for the Streamlit UI), and OutputFormat
(alpaca/sharegpt/chatml/openai).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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
    use_semantic_chunking: bool = False
    enable_dedup: bool = True
    checkpoint_dir: Optional[str] = None

    @field_validator("teacher_model")
    @classmethod
    def validate_teacher(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Teacher model is required")
        return v.strip()

    # ── FIX C-01: safe serialisation that never leaks secrets ────────────
    def safe_dict(self) -> dict:
        """Return model_dump with api_key redacted. Safe for logging / display."""
        d = self.model_dump(exclude_none=True)
        if "api_key" in d:
            d["api_key"] = "***REDACTED***"
        return d

    # ── FIX C-02: prevent API key from leaking in repr / str ─────────────
    def __repr__(self) -> str:
        safe = self.safe_dict()
        fields = ", ".join(f"{k}={v!r}" for k, v in safe.items())
        return f"DistillationConfig({fields})"

    def __str__(self) -> str:
        return self.__repr__()
