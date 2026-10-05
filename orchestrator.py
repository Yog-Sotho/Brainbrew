"""
Brainbrew orchestrator — coordinates document chunking, the distilabel pipeline,
dedup/sanitizing, export, optional LoRA training and optional HF publishing.

Every stage works on canonical records (pipeline/records.py) inside a
persistent run directory (pipeline/runs.py); formatting happens only at export.
Generation itself runs in a child process (pipeline/generation.py).

Heavy GPU imports (via lora_trainer) and optional HF imports (via hf_publisher)
are deferred to inside their respective conditional blocks so this module is
safely importable on CPU-only hosts.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog
from distilabel.models import OpenAILLM, vLLM
from pydantic import ValidationError

from config import DistillationConfig, QualityMode
from pipeline.document_loader import character_chunk, semantic_chunk
from pipeline.exporter import deduplicate_records, export_dataset
from pipeline.generation import generate_rows
from pipeline.quality import QualityReport, score_records
from pipeline.records import Record, write_records
from pipeline.runs import RunDir, create_run, utc_now

logger = structlog.get_logger(__name__)

MAX_SOURCE_BYTES: int = 100 * 1024 * 1024  # 100 MB

_NUM_EVOLUTIONS: dict[QualityMode, int] = {
    QualityMode.FAST: 1,
    QualityMode.BALANCED: 2,
    QualityMode.RESEARCH: 3,
}


# ---------------------------------------------------------------------------
# LLM + pipeline
# ---------------------------------------------------------------------------
def _create_llm(model_name: str, cfg: DistillationConfig) -> Any:
    """Create a distilabel LLM. Sampling settings go in generation_kwargs."""
    generation_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
    }
    if cfg.use_vllm:
        return vLLM(model=model_name, generation_kwargs=generation_kwargs)
    # base_url defaults to $OPENAI_BASE_URL, so any OpenAI-compatible server works.
    return OpenAILLM(
        model=model_name,
        api_key=cfg.api_key,
        generation_kwargs=generation_kwargs,
    )


def _rows_to_records(rows: list[dict[str, Any]]) -> list[Record]:
    records: list[Record] = []
    for row in rows:
        try:
            records.append(Record(
                instruction=row["instruction"],
                output=row["output"],
                meta={"seed": row.get("seed"), "model": row.get("model_name")},
            ))
        except (KeyError, ValidationError):
            continue
    return records


def _split_prompts(prompts: list[str], n_models: int) -> list[list[str]]:
    """Split prompts across models; the last model takes the remainder."""
    size = max(1, len(prompts) // n_models)
    parts = []
    for i in range(n_models):
        start = i * size
        end = start + size if i < n_models - 1 else len(prompts)
        parts.append(prompts[start:end])
    return parts


# ---------------------------------------------------------------------------
# Sanitizing
# ---------------------------------------------------------------------------
def _sanitize(records: list[Record], run: RunDir) -> list[Record]:
    """PII redaction, HTML cleaning, dedup and quality gates on canonical records.

    Raises instead of falling back to the unsanitized data: a user who asked
    for PII removal must never silently get the raw records.
    """
    from pipeline.sanitizer import SanitizerConfig, sanitize_records

    kept, stats = sanitize_records(
        records,
        SanitizerConfig(remove_pii=True, pii_mask=False, clean_html=True, deduplicate=True),
    )
    run.update_manifest(sanitizer=asdict(stats))
    logger.info("Dataset sanitized", **asdict(stats))
    if not kept:
        raise RuntimeError(
            f"Sanitizing removed all {stats.total} records "
            f"({stats.filtered_quality} failed quality checks, "
            f"{stats.filtered_require} were missing fields). "
            "Turn off 'Clean & sanitize' or use a longer source document."
        )
    return kept


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunResult:
    run: RunDir
    dataset_path: Path
    record_count: int
    quality: QualityReport
    adapter_zip: Path | None = None
    published_repo: str | None = None


def run_distillation(
    cfg: DistillationConfig,
    source_file: Path,
    progress_callback: Callable[[int], None] | None = None,
    run: RunDir | None = None,
) -> RunResult:
    """Run the full Brainbrew pipeline inside a persistent run directory.

    Args:
        cfg: Validated pipeline configuration.
        source_file: Path to concatenated source text (copied into the run dir).
        progress_callback: Optional callback receiving progress 0-100.
        run: Run directory to use (a new one is created when omitted).

    Returns:
        RunResult pointing at the exported dataset and run directory.
    """
    run = run or create_run()
    logger.info("Starting distillation", run_id=run.run_id, config=cfg.safe_dict())
    run.update_manifest(status="running", started_at=utc_now(), config=cfg.public_dict())

    try:
        result = _run(cfg, source_file, run, progress_callback)
    except BaseException as exc:
        run.update_manifest(status="failed", finished_at=utc_now(), error=str(exc)[:1000])
        raise
    run.update_manifest(status="succeeded", finished_at=utc_now())
    logger.info("Finished", run_id=run.run_id, path=str(result.dataset_path))
    return result


def _run(
    cfg: DistillationConfig,
    source_file: Path,
    run: RunDir,
    progress_callback: Callable[[int], None] | None,
) -> RunResult:
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
    if source_file.resolve() != run.source.resolve():
        shutil.copyfile(source_file, run.source)
    text = run.source.read_text(encoding="utf-8")
    _progress(5)

    # -- Stage 2: chunk text ------------------------------------------------
    chunks = semantic_chunk(text) if cfg.use_semantic_chunking else character_chunk(text)
    prompts = [
        f"Explain the following concept from the document clearly and completely:\n\n{c}"
        for c in chunks
    ][: cfg.dataset_size]
    logger.info("Document chunked", chunks=len(prompts))
    _progress(15)

    # -- Stage 3: initialise LLM backend(s) ----------------------------------
    model_names = [m.strip() for m in cfg.teacher_model.split(",") if m.strip()]
    num_evolutions = _NUM_EVOLUTIONS[cfg.quality_mode]
    _progress(20)

    # -- Stage 4: run pipeline(s); multi-model ensemble splits the prompts ---
    records: list[Record] = []
    for i, (model_name, model_prompts) in enumerate(
        zip(model_names, _split_prompts(prompts, len(model_names)), strict=True)
    ):
        if not model_prompts:
            continue
        name = f"brainbrew-{run.run_id}" + (f"-m{i}" if len(model_names) > 1 else "")
        logger.info("Running distilabel pipeline", model=model_name, prompts=len(model_prompts))
        rows = generate_rows(
            model_prompts, _create_llm(model_name, cfg), num_evolutions,
            cfg.batch_size, name, run.distilabel_cache,
        )
        records.extend(_rows_to_records(rows))

    write_records(run.raw, records)
    counts: dict[str, int] = {"chunks": len(prompts), "generated": len(records)}
    run.update_manifest(counts=counts)
    if not records:
        raise RuntimeError(
            "The teacher model produced no usable records. Check the model name, "
            "API key / endpoint and the logs, then try again."
        )
    _progress(70)

    # -- Stage 5: dedup + optional sanitizing on canonical records -----------
    if cfg.enable_dedup:
        records = deduplicate_records(records)
        counts["after_dedup"] = len(records)
    _progress(80)

    if cfg.sanitize_dataset:
        records = _sanitize(records, run)
        counts["after_sanitize"] = len(records)
    write_records(run.records, records)
    _progress(85)

    # -- Stage 6: score + export in the chosen format ------------------------
    quality = score_records(records)
    dataset_path = run.dataset(cfg.output_format.value)
    counts["exported"] = export_dataset(records, dataset_path, cfg.output_format.value)
    run.update_manifest(counts=counts, quality=dict(quality), dataset_file=dataset_path.name)
    logger.info("Dataset exported", path=str(dataset_path), records=counts["exported"])

    # -- Stage 7: optional LoRA training on canonical records ----------------
    adapter_zip: Path | None = None
    if cfg.train_model:
        from training.lora_trainer import train_lora

        train_lora(run.records, cfg.base_model, run.adapter_dir, cfg.lora_rank)
        adapter_zip = Path(shutil.make_archive(
            str(run.adapter_zip.with_suffix("")), "zip", root_dir=run.adapter_dir,
        ))
        run.update_manifest(adapter_file=adapter_zip.name)
        _progress(92)

    # -- Stage 8: optional HF publish ----------------------------------------
    published_repo: str | None = None
    if cfg.publish_dataset and cfg.hf_repo:
        from publish.hf_publisher import publish_dataset

        publish_dataset(str(dataset_path), cfg.hf_repo, cfg.hf_token)
        published_repo = cfg.hf_repo
        run.update_manifest(published_repo=published_repo)
        _progress(96)

    _progress(100)
    return RunResult(
        run=run,
        dataset_path=dataset_path,
        record_count=counts["exported"],
        quality=quality,
        adapter_zip=adapter_zip,
        published_repo=published_repo,
    )
