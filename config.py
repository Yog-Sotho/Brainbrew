from enum import Enum
from pydantic import BaseModel


class QualityMode(str, Enum):
    FAST="fast"
    BALANCED="balanced"
    RESEARCH="research"


class DistillationConfig(BaseModel):

    teacher_model:str
    judge_model:str|None=None

    dataset_size:int=2000
    quality_mode:QualityMode=QualityMode.BALANCED

    source_file:str="input.txt"

    raw_dataset:str="dataset_raw.jsonl"
    clean_dataset:str="dataset_clean.jsonl"
    scored_dataset:str="dataset_scored.jsonl"
    final_dataset:str="alpaca_dataset.jsonl"

    use_vllm:bool=True

    train_model:bool=False
    base_model:str="unsloth/mistral-7b-bnb-4bit"

    publish_dataset:bool=False
    hf_repo:str|None=None
