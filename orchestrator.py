import tempfile
from pathlib import Path
from typing import Any, Callable, Optional
import os
import structlog
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import EvolInstruct, TextGeneration
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.base import Step
from distilabel.llms import OpenAILLM, vLLM

from config import DistillationConfig, QualityMode
from pipeline.document_loader import semantic_chunk
from pipeline.exporter import export_alpaca
from training.lora_trainer import train_lora
from publish.hf_publisher import publish_dataset

logger = structlog.get_logger(__name__)

MAX_SOURCE_BYTES = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Custom step: rename 'generation' → 'output' and filter short rows.
# distilabel 1.5.x has no built-in FilterRows or RenameColumns — this is the
# correct pattern: subclass Step, declare inputs/outputs, implement process().
# ---------------------------------------------------------------------------
class FilterAndRenameOutputs(Step):
    """Rename the 'generation' column to 'output' and drop rows below min_length chars."""

    min_length: int = 100

    @property
    def inputs(self) -> list[str]:
        return ["generation"]

    @property
    def outputs(self) -> list[str]:
        return ["output"]

    def process(self, inputs: list[dict[str, Any]]) -> None:  # type: ignore[override]
        kept = []
        for row in inputs:
            gen = row.get("generation", "")
            if isinstance(gen, str) and len(gen) > self.min_length:
                kept.append({**row, "output": gen})
        yield kept


def run_distillation(
    cfg: DistillationConfig,
    source_file: Path,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Path:
    # Redact api_key before logging
    safe_cfg = cfg.model_dump(exclude_none=True)
    if "api_key" in safe_cfg:
        safe_cfg["api_key"] = "***REDACTED***"
    logger.info("Starting distillation", config=safe_cfg)

    def _progress(pct: int) -> None:
        if progress_callback:
            progress_callback(min(pct, 100))

    # ── Stage 1: load & chunk ────────────────────────────────────────────────
    source_bytes = source_file.stat().st_size
    if source_bytes > MAX_SOURCE_BYTES:
        raise ValueError(
            f"Source file is {source_bytes / 1e6:.0f} MB — exceeds the 100 MB limit. "
            "Split the document into smaller files and run multiple times."
        )

    text = source_file.read_text(encoding="utf-8")
    _progress(5)

    chunks = semantic_chunk(text)
    prompts = [
        f"Explain the following concept from the document clearly and completely:\n\n{c}"
        for c in chunks
    ][: cfg.dataset_size]
    logger.info("Document chunked", chunks=len(prompts))
    _progress(15)

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.jsonl"
        final_path = Path("alpaca_dataset.jsonl")

        # ── Stage 2: initialise LLM backend ─────────────────────────────────
        if cfg.use_vllm:
            llm = vLLM(
                model=cfg.teacher_model.split(",")[0],
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )
        else:
            llm = OpenAILLM(
                model=cfg.teacher_model.split(",")[0],
                api_key=cfg.api_key,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )
        _progress(20)

        # ── Stage 3: build pipeline ──────────────────────────────────────────
        # distilabel 1.5.x requires the context-manager pattern + >> chaining.
        # Pipeline(steps=[...]) is NOT a valid API in 1.5.x.
        num_evolutions = (
            3 if cfg.quality_mode == QualityMode.RESEARCH
            else 2 if cfg.quality_mode == QualityMode.BALANCED
            else 1
        )

        with Pipeline(name="brainbrew") as pipeline:
            loader = LoadDataFromDicts(
                data=[{"instruction": p} for p in prompts],
                batch_size=cfg.batch_size,
            )
            evol = EvolInstruct(llm=llm, num_evolutions=num_evolutions)
            # TextGeneration reads 'evolved_instruction' after EvolInstruct
            gen = TextGeneration(
                llm=llm,
                input_mappings={"instruction": "evolved_instruction"},
            )
            # Custom step: renames 'generation'→'output' and filters short rows
            filter_rename = FilterAndRenameOutputs(min_length=100)
            keep = KeepColumns(columns=["instruction", "output"])

            loader >> evol >> gen >> filter_rename >> keep

        # ── Stage 4: run pipeline ────────────────────────────────────────────
        logger.info("Running distilabel pipeline", prompts=len(prompts))
        distiset = pipeline.run(use_cache=False)
        _progress(70)

        # ── Stage 5: export ──────────────────────────────────────────────────
        # Distiset["default"] is a DatasetDict with a "train" split; call
        # to_json on the Dataset, not the DatasetDict.
        distiset["default"]["train"].to_json(str(raw_path))
        export_alpaca(str(raw_path), str(final_path))
        logger.info("Dataset exported", path=str(final_path))
        _progress(80)

        # ── Stage 6: optional LoRA training ─────────────────────────────────
        if cfg.train_model:
            train_lora(str(final_path), cfg.base_model, "trained_adapter", cfg.lora_rank)
            _progress(92)

        # ── Stage 7: optional HF publish ─────────────────────────────────────
        if cfg.publish_dataset and cfg.hf_repo:
            publish_dataset(str(final_path), cfg.hf_repo, os.getenv("HF_TOKEN"))
            _progress(96)

        _progress(100)
        logger.info("Finished", path=str(final_path))
        return final_path
