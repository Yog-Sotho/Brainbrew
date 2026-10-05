"""
Brainbrew orchestrator — coordinates chunking, grounded generation (with
filters, an LLM judge and near-duplicate removal), optional sanitizing,
export, optional LoRA training and optional HF publishing.

Every stage works on canonical records (pipeline/records.py) inside a
persistent run directory (pipeline/runs.py); formatting happens only at export.
Generation (pipeline/synth.py) talks to any OpenAI-compatible endpoint through
the async engine (engine/client.py).

Heavy GPU imports (via lora_trainer) and optional HF imports (via hf_publisher)
are deferred to inside their respective conditional blocks so this module is
safely importable on CPU-only hosts.
"""
from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog

from config import DistillationConfig
from engine import ChatClient, EndpointSettings
from pipeline.document_loader import source_chunks
from pipeline.exporter import export_dataset
from pipeline.quality import QualityReport, score_records
from pipeline.records import Record, write_records
from pipeline.runs import RunDir, create_run, utc_now
from pipeline.synth import SynthSettings, SynthStats, synthesize

logger = structlog.get_logger(__name__)

MAX_SOURCE_BYTES: int = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
def _make_client(settings: EndpointSettings) -> ChatClient:
    """Create a chat client (tests replace this to plug in a fake server)."""
    return ChatClient(settings)


def _endpoint(cfg: DistillationConfig, model: str, temperature: float | None = None) -> EndpointSettings:
    return EndpointSettings(
        model=model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        temperature=cfg.temperature if temperature is None else temperature,
        max_tokens=cfg.max_new_tokens,
        timeout_s=float(cfg.request_timeout),
        concurrency=cfg.concurrency,
    )


async def _generate(
    cfg: DistillationConfig,
    chunks: list[str],
    progress: Callable[[float], None],
) -> tuple[list[Record], SynthStats, dict[str, dict[str, int]]]:
    teachers = [_make_client(_endpoint(cfg, m)) for m in cfg.teacher_models]
    judge = None
    if cfg.uses_judge:
        judge = _make_client(_endpoint(cfg, cfg.judge_model or cfg.teacher_models[0], temperature=0.0))
    embedder = None
    if cfg.enable_dedup and cfg.embedding_model:
        embedder = _make_client(_endpoint(cfg, cfg.embedding_model))
    settings = SynthSettings(
        target=cfg.dataset_size,
        evolve=cfg.evolves,
        judge_threshold=cfg.judge_threshold,
        dedup_threshold=0.85 if cfg.enable_dedup else None,
        semantic_threshold=cfg.semantic_dedup_threshold,
    )
    extra = [c for c in (judge, embedder) if c is not None]
    try:
        records, stats = await synthesize(chunks, teachers, settings, judge=judge, progress=progress,
                                          embedder=embedder)
    finally:
        for client in [*teachers, *extra]:
            await client.close()
    usage = {f"teacher:{t.settings.model}": t.usage.as_dict() for t in teachers}
    if judge:
        usage[f"judge:{judge.settings.model}"] = judge.usage.as_dict()
    if embedder:
        usage[f"embeddings:{embedder.settings.model}"] = embedder.usage.as_dict()
    return records, stats, usage


# ---------------------------------------------------------------------------
# Decontamination and sanitizing
# ---------------------------------------------------------------------------
def _decontaminate(records: list[Record], cfg: DistillationConfig, run: RunDir) -> list[Record]:
    """Drop records that overlap the selected public benchmarks."""
    from pipeline import decontam

    kept, removed = decontam.decontaminate(records, cfg.decontaminate, loader=decontam.load_eval_texts)
    run.update_manifest(decontamination=removed)
    logger.info("Benchmark overlap removed", **removed)
    if not kept:
        raise RuntimeError(
            f"All {len(records)} records overlap the selected benchmarks "
            f"({', '.join(cfg.decontaminate)}); nothing is left to export."
        )
    return kept


def _sanitize(records: list[Record], cfg: DistillationConfig, run: RunDir) -> list[Record]:
    """PII redaction, HTML cleaning, dedup and quality gates on canonical records.

    Raises instead of falling back to the unsanitized data: a user who asked
    for PII removal must never silently get the raw records.
    """
    from pipeline.sanitizer import SanitizerConfig, sanitize_records

    kept, stats = sanitize_records(
        records,
        SanitizerConfig(remove_pii=True, pii_mask=False, clean_html=True, deduplicate=True,
                        url_policy=cfg.pii_url_policy, presidio=cfg.pii_presidio),
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
    chunks = source_chunks(text, cfg.use_semantic_chunking)
    logger.info("Document chunked", chunks=len(chunks))
    run.update_manifest(counts={"chunks": len(chunks)})
    _progress(15)

    # -- Stage 3-4: grounded generation, filters, judge, dedup ---------------
    def _gen_progress(fraction: float) -> None:
        _progress(15 + int(fraction * 55))

    records, stats, usage = asyncio.run(_generate(cfg, chunks, _gen_progress))
    write_records(run.raw, records)
    counts: dict[str, int] = {"chunks": len(chunks), "generated": len(records)}
    run.update_manifest(counts=counts, generation=stats.as_dict(), usage=usage)
    logger.info("Generation finished", **{k: v for k, v in stats.as_dict().items() if k != "answers_filtered"})
    if not records:
        raise RuntimeError(
            "No usable question/answer pairs were produced "
            f"({stats.errors} failed requests, {stats.judge_rejected} rejected by the judge, "
            f"{sum(stats.answers_filtered.values())} filtered answers). Check the model name, "
            "API key / endpoint and the logs, then try again."
        )
    _progress(75)

    # -- Stage 5: optional benchmark decontamination and sanitizing ----------
    if cfg.decontaminate:
        records = _decontaminate(records, cfg, run)
        counts["after_decontamination"] = len(records)
    _progress(80)
    if cfg.sanitize_dataset:
        records = _sanitize(records, cfg, run)
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
