import tempfile
from pathlib import Path
from typing import Callable, Optional
import os
import structlog
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import EvolInstruct, TextGeneration
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.filtering import FilterRows
from distilabel.llms import OpenAILLM, vLLM

from config import DistillationConfig, QualityMode
from pipeline.document_loader import semantic_chunk
from pipeline.exporter import export_alpaca
from training.lora_trainer import train_lora
from publish.hf_publisher import publish_dataset

logger = structlog.get_logger(__name__)

# FIX: guard against silent slowdowns from huge source files
MAX_SOURCE_BYTES = 100 * 1024 * 1024  # 100 MB


def run_distillation(cfg: DistillationConfig, source_file: Path, progress_callback: Optional[Callable[[int], None]] = None) -> Path:
    # FIX S-01: redact api_key before logging
    safe_cfg = cfg.model_dump(exclude_none=True)
    if "api_key" in safe_cfg:
        safe_cfg["api_key"] = "***REDACTED***"
    logger.info("Starting distillation", config=safe_cfg)

    # FIX: honest progress — helper so every stage reports where we actually are
    def _progress(pct: int) -> None:
        if progress_callback:
            progress_callback(min(pct, 100))

    # ── Stage 1: load & chunk (5 → 15%) ──────────────────────────────────────
    source_bytes = source_file.stat().st_size
    if source_bytes > MAX_SOURCE_BYTES:
        raise ValueError(
            f"Source file is {source_bytes / 1e6:.0f} MB — exceeds the 100 MB limit. "
            "Split the document into smaller files and run multiple times."
        )

    text = source_file.read_text(encoding="utf-8")
    _progress(5)

    chunks = semantic_chunk(text)
    prompts = [f"Explain the following concept from the document clearly and completely:\n\n{c}" for c in chunks][:cfg.dataset_size]
    logger.info("Document chunked", chunks=len(prompts))
    _progress(15)

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.jsonl"
        final_path = Path("alpaca_dataset.jsonl")

        # ── Stage 2: initialise LLM backend (15 → 20%) ───────────────────────
        if cfg.use_vllm:
            llm = vLLM(model=cfg.teacher_model.split(",")[0], max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)
        else:
            llm = OpenAILLM(model=cfg.teacher_model.split(",")[0], api_key=cfg.api_key, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)
        _progress(20)

        # ── Stage 3: build pipeline (20%) ────────────────────────────────────
        pipeline = Pipeline(steps=[
            LoadDataFromDicts(data=[{"instruction": p} for p in prompts]),
            EvolInstruct(llm=llm, num_evolutions=3 if cfg.quality_mode == QualityMode.RESEARCH else 2 if cfg.quality_mode == QualityMode.BALANCED else 1),
            # FIX: TextGeneration reads evolved_instruction as input prompt
            TextGeneration(llm=llm, input_mappings={"instruction": "evolved_instruction"}),
            # FIX: TextGeneration outputs to 'generation', rename to 'output' for exporter
            RenameColumns(rename_mappings={"generation": "output"}),
            KeepColumns(columns=["instruction", "output"]),
            # FIX: filter on 'output' (now correctly renamed from 'generation')
            FilterRows(filter_func=lambda row: len(row.get("output", "")) > 100),
        ])

        # ── Stage 4: run pipeline — the long one (20 → 70%) ──────────────────
        # pipeline.run() is blocking; we honestly report 20% going in and 70% coming out.
        # The progress bar shows real movement at real checkpoints, not fake ticks.
        logger.info("Running distilabel pipeline", prompts=len(prompts))
        distiset = pipeline.run()
        _progress(70)

        # ── Stage 5: export (70 → 80%) ────────────────────────────────────────
        distiset["default"].to_json(str(raw_path))
        export_alpaca(str(raw_path), str(final_path))
        logger.info("Dataset exported", path=str(final_path))
        _progress(80)

        # ── Stage 6: optional LoRA training (80 → 92%) ────────────────────────
        if cfg.train_model:
            train_lora(str(final_path), cfg.base_model, "trained_adapter", cfg.lora_rank)
            _progress(92)

        # ── Stage 7: optional HF publish (92 → 96%) ───────────────────────────
        if cfg.publish_dataset and cfg.hf_repo:
            publish_dataset(str(final_path), cfg.hf_repo, os.getenv("HF_TOKEN"))
            _progress(96)

        _progress(100)
        logger.info("Finished", path=str(final_path))
        return final_path
# FIX: removed the second run_distillation() definition that was overriding this one
