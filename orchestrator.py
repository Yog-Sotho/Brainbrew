"""
Brainbrew orchestrator — coordinates document loading, distilabel pipeline,
optional LoRA training, and optional HF publishing.

Heavy GPU imports (via lora_trainer) and optional HF imports (via hf_publisher)
are deferred to inside their respective conditional blocks so this module is
safely importable on CPU-only hosts.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional
import os
import structlog
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import EvolInstruct, TextGeneration, UltraFeedback
from distilabel.steps import LoadDataFromDicts, KeepColumns
from distilabel.steps.base import Step
from distilabel.llms import OpenAILLM, vLLM

from config import DistillationConfig, OutputFormat, QualityMode
from pipeline.document_loader import semantic_chunk, character_chunk
from pipeline.exporter import export_dataset

logger = structlog.get_logger(__name__)

MAX_SOURCE_BYTES: int = 100 * 1024 * 1024  # 100 MB
MAX_EXPORT_RECORDS: int = 500_000

# Map output formats to the fields the sanitizer should require as non-empty.
_SANITIZER_REQUIRE_FIELDS: dict[str, list[str]] = {
    "alpaca":   ["instruction", "output"],
    "sharegpt":  ["conversations"],
    "chatml":    ["messages"],
    "openai":    ["messages"],
}


# ---------------------------------------------------------------------------
# Custom distilabel Step: rename 'generation' -> 'output' and filter short rows.
# distilabel 1.5.x has no built-in FilterRows or RenameColumns.
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

    def process(self, inputs: list[dict[str, Any]]) -> Any:  # type: ignore[override]
        # FIX M-11: yield individual batches correctly per distilabel Step protocol
        kept = []
        for row in inputs:
            gen = row.get("generation", "")
            if isinstance(gen, str) and len(gen) > self.min_length:
                kept.append({**row, "output": gen})
        yield kept


# ---------------------------------------------------------------------------
# Enhancement 10: Quality scoring
# ---------------------------------------------------------------------------
_QUALITY_THRESHOLDS = {
    "SUPER":    {"min_records": 100, "min_avg_len": 300, "min_unique_ratio": 0.95},
    "GOOD":     {"min_records": 50,  "min_avg_len": 200, "min_unique_ratio": 0.85},
    "NORMAL":   {"min_records": 20,  "min_avg_len": 100, "min_unique_ratio": 0.70},
    "BAD":      {"min_records": 5,   "min_avg_len": 50,  "min_unique_ratio": 0.50},
}


def score_dataset(dataset_path: Path) -> dict[str, Any]:
    """Score the generated dataset and return a quality report.

    Returns dict with keys: grade, record_count, avg_output_length,
    unique_ratio, details.
    """
    records: list[dict] = []
    try:
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except (OSError, IOError):
        return {
            "grade": "DISASTER",
            "record_count": 0,
            "avg_output_length": 0,
            "unique_ratio": 0.0,
            "details": "Could not read dataset file.",
        }

    if not records:
        return {
            "grade": "DISASTER",
            "record_count": 0,
            "avg_output_length": 0,
            "unique_ratio": 0.0,
            "details": "Dataset is empty — no valid records produced.",
        }

    # Compute metrics
    record_count = len(records)
    output_lengths = [len(r.get("output", "")) for r in records]
    avg_output_len = sum(output_lengths) / len(output_lengths) if output_lengths else 0

    # Unique instruction ratio
    instructions = [r.get("instruction", "") for r in records]
    unique_instructions = len(set(instructions))
    unique_ratio = unique_instructions / len(instructions) if instructions else 0.0

    # Determine grade
    grade = "BAD"
    for level in ["SUPER", "GOOD", "NORMAL", "BAD"]:
        thresholds = _QUALITY_THRESHOLDS[level]
        if (record_count >= thresholds["min_records"]
                and avg_output_len >= thresholds["min_avg_len"]
                and unique_ratio >= thresholds["min_unique_ratio"]):
            grade = level
            break

    # Build human-readable details
    detail_parts = [
        f"{record_count} records generated",
        f"Average output length: {avg_output_len:.0f} chars",
        f"Instruction uniqueness: {unique_ratio:.0%}",
    ]
    if avg_output_len < 100:
        detail_parts.append("⚠ Outputs are very short — consider using Research mode.")
    if unique_ratio < 0.70:
        detail_parts.append("⚠ Many duplicate instructions — increase dataset_size or source material.")

    return {
        "grade": grade,
        "record_count": record_count,
        "avg_output_length": avg_output_len,
        "unique_ratio": unique_ratio,
        "details": " · ".join(detail_parts),
    }


# ---------------------------------------------------------------------------
# Enhancement 7: Checkpoint support
# ---------------------------------------------------------------------------
def _load_checkpoint(checkpoint_dir: Optional[str]) -> dict[str, Any]:
    """Load checkpoint state if it exists."""
    if not checkpoint_dir:
        return {}
    cp_path = Path(checkpoint_dir) / "brainbrew_checkpoint.json"
    if cp_path.exists():
        try:
            return json.loads(cp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_checkpoint(checkpoint_dir: Optional[str], state: dict[str, Any]) -> None:
    """Save checkpoint state."""
    if not checkpoint_dir:
        return
    cp_dir = Path(checkpoint_dir)
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_path = cp_dir / "brainbrew_checkpoint.json"
    cp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Enhancement 4: Multi-model support
# ---------------------------------------------------------------------------
def _create_llm(model_name: str, cfg: DistillationConfig) -> Any:
    """Create an LLM instance for a single model."""
    if cfg.use_vllm:
        return vLLM(
            model=model_name,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
    else:
        return OpenAILLM(
            model=model_name,
            api_key=cfg.api_key,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )


def _run_single_pipeline(
    prompts: list[str],
    llm: Any,
    num_evolutions: int,
    batch_size: int,
    raw_path: Path,
    judge_llm: Optional[Any] = None,
) -> None:
    """Run a single distilabel pipeline for one model and write results."""
    with Pipeline(name="brainbrew") as pipeline:
        loader = LoadDataFromDicts(
            data=[{"instruction": p} for p in prompts],
            batch_size=batch_size,
        )
        evol = EvolInstruct(llm=llm, num_evolutions=num_evolutions)
        gen = TextGeneration(
            llm=llm,
            input_mappings={"instruction": "evolved_instruction"},
        )
        filter_rename = FilterAndRenameOutputs(min_length=100)

        if judge_llm:
            judge = UltraFeedback(
                llm=judge_llm,
                aspect="overall-rating",
                input_mappings={"instruction": "evolved_instruction", "generations": "output"}
            )
            keep = KeepColumns(columns=["instruction", "output", "rating", "rationale"])
            loader >> evol >> gen >> filter_rename >> judge >> keep
        else:
            keep = KeepColumns(columns=["instruction", "output"])
            loader >> evol >> gen >> filter_rename >> keep

    distiset = pipeline.run(use_cache=False)
    distiset["default"]["train"].to_json(str(raw_path))


# ---------------------------------------------------------------------------
# Post-export dataset sanitization (native — no subprocess)
# ---------------------------------------------------------------------------
def _run_sanitizer(
    dataset_path: Path,
    output_format: str,
) -> Path:
    """Run the native sanitizer on the exported dataset.

    Applies PII redaction, HTML cleaning, deduplication, and quality gates.
    On success the sanitized file replaces the original. On failure the
    original is left untouched and a warning is logged.

    Args:
        dataset_path: Path to the exported JSONL dataset.
        output_format: One of 'alpaca', 'sharegpt', 'chatml', 'openai'.

    Returns:
        Path to the (possibly sanitized) dataset — same as dataset_path.
    """
    from pipeline.sanitizer import SanitizerConfig, sanitize_dataset

    sanitized_path = dataset_path.with_suffix(".sanitized.jsonl")

    require_fields = _SANITIZER_REQUIRE_FIELDS.get(output_format, ["instruction", "output"])

    san_cfg = SanitizerConfig(
        remove_pii=True,
        pii_mask=False,
        clean_html=True,
        deduplicate=True,
        require_fields=require_fields,
    )

    try:
        stats = sanitize_dataset(dataset_path, sanitized_path, san_cfg)

        # Verify sanitized output is non-empty
        if not sanitized_path.exists() or sanitized_path.stat().st_size == 0:
            logger.warning(
                "Sanitizer produced empty output — keeping original dataset",
            )
            if sanitized_path.exists():
                sanitized_path.unlink()
            return dataset_path

        # Replace original with sanitized version
        shutil.move(str(sanitized_path), str(dataset_path))

        logger.info(
            "Dataset sanitized",
            original_records=stats.total,
            kept_records=stats.kept,
            filtered_quality=stats.filtered_quality,
            filtered_require=stats.filtered_require,
            deduplicated=stats.deduplicated,
            pii_redacted=stats.pii_redacted,
        )

        return dataset_path

    except Exception as exc:
        logger.warning(
            "Sanitizer failed — keeping original dataset",
            error=str(exc),
        )
        if sanitized_path.exists():
            try:
                sanitized_path.unlink()
            except OSError:
                pass
        return dataset_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_distillation(
    cfg: DistillationConfig,
    source_file: Path,
    progress_callback: Optional[Callable[[int], None]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Run the full Brainbrew distillation pipeline.

    Args:
        cfg: Validated pipeline configuration.
        source_file: Path to concatenated source text.
        progress_callback: Optional callback receiving progress 0-100.
        output_dir: Directory for output files (default: temp dir adjacent to source).

    Returns:
        Path to the final exported dataset JSONL.
    """
    # Redact api_key before logging
    safe_cfg = cfg.safe_dict()
    logger.info("Starting distillation", config=safe_cfg)

    def _progress(pct: int) -> None:
        if progress_callback:
            progress_callback(min(pct, 100))

    # -- Stage 1: load & validate source -------------------------------------
    source_bytes = source_file.stat().st_size
    if source_bytes > MAX_SOURCE_BYTES:
        raise ValueError(
            f"Source file is {source_bytes / 1e6:.0f} MB — exceeds the 100 MB limit. "
            "Split the document into smaller files and run multiple times."
        )

    text = source_file.read_text(encoding="utf-8")
    _progress(5)

    # -- Stage 2: chunk text ------------------------------------------------
    # Enhancement 9: use semantic chunking when enabled
    if cfg.use_semantic_chunking:
        chunks = semantic_chunk(text)
    else:
        chunks = character_chunk(text)

    prompts = [
        f"Explain the following concept from the document clearly and completely:\n\n{c}"
        for c in chunks
    ][: cfg.dataset_size]
    logger.info("Document chunked", chunks=len(prompts))
    _progress(15)

    # -- Enhancement 7: check for checkpoint --------------------------------
    checkpoint = _load_checkpoint(cfg.checkpoint_dir)
    completed_prompts = set(checkpoint.get("completed_hashes", []))
    if completed_prompts:
        original_count = len(prompts)
        prompts = [
            p for p in prompts
            if hashlib.sha256(p.encode()).hexdigest() not in completed_prompts
        ]
        logger.info(
            "Checkpoint resumed",
            skipped=original_count - len(prompts),
            remaining=len(prompts),
        )
        if not prompts:
            logger.info("All prompts already processed. Skipping pipeline.")
            final_path = Path(checkpoint.get("final_path", "alpaca_dataset.jsonl"))
            _progress(100)
            return final_path

    # FIX M-02: use configurable output directory instead of cwd
    if output_dir is None:
        output_dir = source_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename based on format
    format_extensions = {
        "alpaca": "alpaca_dataset.jsonl",
        "sharegpt": "sharegpt_dataset.jsonl",
        "chatml": "chatml_dataset.jsonl",
        "openai": "openai_dataset.jsonl",
    }
    output_filename = format_extensions.get(cfg.output_format.value, "dataset.jsonl")
    final_path = output_dir / output_filename

    with tempfile.TemporaryDirectory() as tmp:
        # -- Stage 3: initialise LLM backend(s) ------------------------------
        model_names = [m.strip() for m in cfg.teacher_model.split(",") if m.strip()]
        num_evolutions = (
            3 if cfg.quality_mode == QualityMode.RESEARCH
            else 2 if cfg.quality_mode == QualityMode.BALANCED
            else 1
        )
        _progress(20)

        # -- Stage 4: run pipeline(s) ----------------------------------------
        raw_parts: list[Path] = []

        judge_llm = None
        if cfg.judge_model:
            judge_llm = _create_llm(cfg.judge_model, cfg)

        if len(model_names) == 1:
            # Single model — standard path
            llm = _create_llm(model_names[0], cfg)
            raw_path = Path(tmp) / "raw.jsonl"
            logger.info("Running distilabel pipeline", model=model_names[0], prompts=len(prompts))
            _run_single_pipeline(
                prompts, llm, num_evolutions, cfg.batch_size, raw_path, judge_llm=judge_llm
            )
            raw_parts.append(raw_path)
        else:
            # Enhancement 4: multi-model ensemble — split prompts across models
            logger.info(
                "Multi-model ensemble",
                models=model_names,
                prompts_per_model=len(prompts) // len(model_names),
            )
            chunk_size = max(1, len(prompts) // len(model_names))
            for i, model_name in enumerate(model_names):
                start = i * chunk_size
                end = start + chunk_size if i < len(model_names) - 1 else len(prompts)
                model_prompts = prompts[start:end]
                if not model_prompts:
                    continue
                llm = _create_llm(model_name, cfg)
                raw_path = Path(tmp) / f"raw_{i}.jsonl"
                logger.info(
                    "Running pipeline for model",
                    model=model_name,
                    prompts=len(model_prompts),
                )
                _run_single_pipeline(
                    model_prompts, llm, num_evolutions, cfg.batch_size, raw_path, judge_llm=judge_llm
                )
                raw_parts.append(raw_path)

        _progress(70)

        # -- Stage 5: merge multi-model outputs if needed ---------------------
        merged_raw = Path(tmp) / "merged_raw.jsonl"
        with open(merged_raw, "w", encoding="utf-8") as fout:
            for part in raw_parts:
                if part.exists():
                    with open(part, encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)

        # -- Stage 6: export in chosen format --------------------------------
        record_count = export_dataset(
            str(merged_raw),
            str(final_path),
            output_format=cfg.output_format.value,
            enable_dedup=cfg.enable_dedup,
            max_records=MAX_EXPORT_RECORDS,
        )
        logger.info("Dataset exported", path=str(final_path), records=record_count)
        _progress(80)

        # -- Stage 6.5: optional post-export sanitization --------------------
        if cfg.sanitize_dataset:
            final_path = _run_sanitizer(
                final_path,
                output_format=cfg.output_format.value,
            )
        _progress(85)

        # -- Enhancement 7: save checkpoint -----------------------------------
        all_hashes = list(completed_prompts)
        for p in prompts:
            all_hashes.append(hashlib.sha256(p.encode()).hexdigest())
        _save_checkpoint(cfg.checkpoint_dir, {
            "completed_hashes": all_hashes,
            "final_path": str(final_path),
        })

        # -- Stage 7: optional LoRA training ---------------------------------
        if cfg.train_model:
            from training.lora_trainer import train_lora
            train_lora(str(final_path), cfg.base_model, "trained_adapter", cfg.lora_rank)
            _progress(92)

        # -- Stage 8: optional HF publish ------------------------------------
        if cfg.publish_dataset and cfg.hf_repo:
            from publish.hf_publisher import publish_dataset
            publish_dataset(str(final_path), cfg.hf_repo, os.getenv("HF_TOKEN"))
            _progress(96)

        _progress(100)
        logger.info("Finished", path=str(final_path))
        return final_path
