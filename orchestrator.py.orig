import tempfile
from pathlib import Path
from typing import Callable, Optional
import os
import structlog
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import EvolInstruct, TextGeneration
from distilabel.steps import LoadDataFromDicts, FilterStep, KeepColumns
from distilabel.llms import OpenAILLM, vLLM

from config import DistillationConfig, QualityMode
from pipeline.document_loader import semantic_chunk
from pipeline.exporter import export_alpaca
from training.lora_trainer import train_lora
from publish.hf_publisher import publish_dataset

logger = structlog.get_logger(__name__)

def run_distillation(cfg: DistillationConfig, source_file: Path, progress_callback: Optional[Callable[[int], None]] = None) -> Path:
    logger.info("Starting distillation", config=cfg.model_dump(exclude_none=True))

    text = source_file.read_text(encoding="utf-8")
    chunks = semantic_chunk(text)
    prompts = [f"Explain the following concept from the document clearly and completely:\n\n{c}" for c in chunks][:cfg.dataset_size]

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.jsonl"
        final_path = Path("alpaca_dataset.jsonl")

        if cfg.use_vllm:
            llm = vLLM(model=cfg.teacher_model.split(",")[0], max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)
        else:
            llm = OpenAILLM(model=cfg.teacher_model.split(",")[0], api_key=cfg.api_key, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)

        pipeline = Pipeline(steps=[
            LoadDataFromDicts(data=[{"instruction": p} for p in prompts]),
            EvolInstruct(llm=llm, num_evolutions=3 if cfg.quality_mode == QualityMode.RESEARCH else 2 if cfg.quality_mode == QualityMode.BALANCED else 1),
            TextGeneration(llm=llm),
            KeepColumns(columns=["instruction", "output"]),
            FilterStep(condition="len(output) > 100"),
        ])

        distiset = pipeline.run()
        distiset["default"].to_json(str(raw_path))
        export_alpaca(str(raw_path), str(final_path))

        if cfg.train_model:
            train_lora(str(final_path), cfg.base_model, "trained_adapter", cfg.lora_rank)

        if cfg.publish_dataset and cfg.hf_repo:
            publish_dataset(str(final_path), cfg.hf_repo, os.getenv("HF_TOKEN"))

        if progress_callback:
            progress_callback(100)

        logger.info("Finished", path=str(final_path))
        return final_path
def run_distillation(cfg: DistillationConfig):

    text = load_document(cfg.source_file)

    chunks = chunk_text(text)

    prompts = [
        f"Explain the following concept clearly:\n\n{c}"
        for c in chunks
    ]

    if cfg.quality_mode == QualityMode.RESEARCH:
        prompts = evolve(prompts,3)

    elif cfg.quality_mode == QualityMode.BALANCED:
        prompts = evolve(prompts,2)

    prompts = prompts[:cfg.dataset_size]

    teacher = create_teacher(cfg)

    build_with_vllm(
        prompts,
        teacher,
        cfg.raw_dataset,
        batch_size=64
    )

    clean_dataset(cfg.raw_dataset, cfg.clean_dataset)

    if cfg.quality_mode != QualityMode.FAST:

        judge_engine = TeacherEngine(cfg.judge_model)

        judge = JudgeEngine(judge_engine)

        threshold = 6 if cfg.quality_mode == QualityMode.BALANCED else 8

        evaluate_dataset(
            cfg.clean_dataset,
            cfg.scored_dataset,
            judge,
            threshold
        )

        export_alpaca(cfg.scored_dataset, cfg.final_dataset)

    else:

        export_alpaca(cfg.clean_dataset, cfg.final_dataset)

    if cfg.train_model:

        train_lora(
            cfg.final_dataset,
            cfg.base_model,
            "trained_model"
        )

    if cfg.publish_dataset:

        publish_dataset(
            cfg.final_dataset,
            cfg.hf_repo
        )

    return cfg.final_dataset
