from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class QualityMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    RESEARCH = "research"


# FIX: app.py imports this dict for the selectbox display labels.
# Was missing entirely from config.py — caused ImportError at startup.
QUALITY_MODE_LABELS: dict[QualityMode, str] = {
    QualityMode.FAST:     "Fast ⚡ (quick & cheap)",
    QualityMode.BALANCED: "Balanced 🎯 (sweet spot)",
    QualityMode.RESEARCH: "Research 🔬 (maximum quality)",
}


class DistillationConfig(BaseModel):
    teacher_model: str = Field(..., description="Model name or comma-separated list")
    judge_model: Optional[str] = "gpt-4o-mini"
    dataset_size: int = Field(2000, ge=100, le=50000)
    quality_mode: QualityMode = QualityMode.BALANCED
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

    @field_validator("teacher_model")
    @classmethod
    def validate_teacher(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Teacher model is required")
        return v.strip()
