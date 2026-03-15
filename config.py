from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class QualityMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    RESEARCH = "research"

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
    # FIX S-01: repr=False prevents api_key appearing in __repr__ / logs
    api_key: Optional[str] = Field(None, repr=False)

    @field_validator("teacher_model")
    @classmethod
    def validate_teacher(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Teacher model is required")
        return v.strip()

    # FIX S-01: safe serialisation helper — always call this for logging
    def safe_dict(self) -> dict:
        data = self.model_dump(exclude_none=True)
        if "api_key" in data:
            data["api_key"] = "***REDACTED***"
        return data
