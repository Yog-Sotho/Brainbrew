"""
Brainbrew orchestrator — coordinates chunking, grounded generation (with
filters, an LLM judge and near-duplicate removal), optional sanitizing,
export, optional LoRA training and optional HF publishing.

Every stage works on canonical records (pipeline/records.py) inside a
persistent run directory (pipeline/runs.py); formatting happens only at export.
Generation (pipeline/synth.py) talks to any OpenAI-compatible endpoint through
the async engine (engine/client.py).

Runs can be cancelled through a threading.Event (in-flight model requests are
cancelled too), training waits for the machine's single GPU slot, and the
manifest records progress, models, seed, token usage and cost, stage timings
and every count, so a run can be followed and audited from its folder alone.

Heavy GPU imports (via lora_trainer) and optional HF imports (via hf_publisher)
are deferred to inside their respective conditional blocks so this module is
safely importable on CPU-only hosts.
"""
from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog

from config import DistillationConfig
from engine import ChatClient, EndpointSettings
from pipeline.document_loader import source_chunks
from pipeline.exporter import export_dataset
from pipeline.gpu import gpu_slot
from pipeline.logs import configure_logging, run_log
from pipeline.pricing import is_local, run_cost
from pipeline.quality import QualityReport, score_records
from pipeline.records import Record, write_records
from pipeline.runs import RunCancelled, RunDir, create_run, utc_now
from pipeline.synth import SynthSettings, SynthStats, synthesize
from pipeline.version import __version__

logger = structlog.get_logger(__name__)

__all__ = ["RunCancelled", "RunResult", "run_distillation"]

MAX_SOURCE_BYTES: int = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
ClientFactory = Callable[[EndpointSettings], ChatClient]


def _make_client(settings: EndpointSettings) -> ChatClient:
    """Create a chat client (tests replace this to plug in a fake server)."""
    return ChatClient(settings)


def _endpoint(
    cfg: DistillationConfig, model: str, temperature: float | None = None, seed: int | None = None,
) -> EndpointSettings:
    return EndpointSettings(
        seed=seed,
        model=model,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        temperature=cfg.temperature if temperature is None else temperature,
        max_tokens=cfg.max_new_tokens,
        timeout_s=float(cfg.request_timeout),
        concurrency=cfg.concurrency,
        reasoning_effort=cfg.reasoning_effort,
    )


async def _watch(cancel: threading.Event, task: asyncio.Task[Any]) -> None:
    """Cancel *task* (and with it every in-flight request) once *cancel* is set."""
    while not cancel.is_set():
        await asyncio.sleep(0.2)
    task.cancel()


async def _generate(
    cfg: DistillationConfig,
    chunks: list[str],
    progress: Callable[[float], None],
    make_client: ClientFactory,
    seed: int | None = None,
    cancel: threading.Event | None = None,
) -> tuple[list[Record], SynthStats, dict[str, dict[str, int]]]:
    teachers = [make_client(_endpoint(cfg, m, seed=seed)) for m in cfg.teacher_models]
    judge = None
    if cfg.uses_judge:
        judge = make_client(_endpoint(cfg, cfg.judge_model or cfg.teacher_models[0], temperature=0.0, seed=seed))
    embedder = None
    if cfg.enable_dedup and cfg.embedding_model:
        embedder = make_client(_endpoint(cfg, cfg.embedding_model))
    settings = SynthSettings(
        target=cfg.dataset_size,
        evolve=cfg.evolves,
        judge_threshold=cfg.judge_threshold,
        dedup_threshold=0.85 if cfg.enable_dedup else None,
        semantic_threshold=cfg.semantic_dedup_threshold,
    )
    extra = [c for c in (judge, embedder) if c is not None]
    task = asyncio.ensure_future(
        synthesize(chunks, teachers, settings, judge=judge, progress=progress, embedder=embedder)
    )
    watcher = asyncio.ensure_future(_watch(cancel, task)) if cancel is not None else None
    try:
        records, stats = await task
    except asyncio.CancelledError:
        if cancel is not None and cancel.is_set():
            raise RunCancelled() from None
        raise
    finally:
        if watcher is not None:
            watcher.cancel()
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


ProgressCallback = Callable[[int, str], None]


class _Tracker:
    """Progress, stage labels, stage timings and cancellation for one run."""

    def __init__(self, run: RunDir, callback: ProgressCallback | None, cancel: threading.Event | None) -> None:
        self.run = run
        self.callback = callback
        self.cancel = cancel
        self.timings: dict[str, float] = {}
        self.pct = 0
        self.stage = ""

    def check(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise RunCancelled()

    def progress(self, pct: int, stage: str | None = None) -> None:
        self.check()
        self.pct = min(max(pct, self.pct), 100)  # never goes backwards
        if stage is not None:
            self.stage = stage
        if self.callback:
            self.callback(self.pct, self.stage)

    @contextmanager
    def step(self, name: str, pct: int, label: str) -> Iterator[None]:
        """Time one stage and record it in the manifest when it starts and ends."""
        self.progress(pct, label)
        self.run.update_manifest(stage=label, progress=self.pct)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round(time.perf_counter() - start, 3)
            self.run.update_manifest(timings=self.timings)


def run_distillation(
    cfg: DistillationConfig,
    source_file: Path,
    progress_callback: ProgressCallback | None = None,
    run: RunDir | None = None,
    client_factory: ClientFactory | None = None,
    cancel: threading.Event | None = None,
    owner: str | None = None,
) -> RunResult:
    """Run the full Brainbrew pipeline inside a persistent run directory.

    Args:
        cfg: Validated pipeline configuration.
        source_file: Path to concatenated source text (copied into the run dir).
        progress_callback: Optional callback receiving (progress 0-100, stage label).
        run: Run directory to use (a new one is created when omitted).
        client_factory: Builds the model clients (default: a real ChatClient per endpoint).
        cancel: Set it to stop the run; it then ends with status "cancelled".
        owner: Who started the run (recorded in the manifest, used to scope run history).

    Returns:
        RunResult pointing at the exported dataset and run directory.

    Raises:
        RunCancelled: when *cancel* was set.
    """
    configure_logging()
    run = run or create_run()
    seed = cfg.seed if cfg.seed is not None else secrets.randbelow(2**31)
    fields: dict[str, Any] = {
        "status": "running",
        "started_at": utc_now(),
        "brainbrew_version": __version__,
        "config": cfg.public_dict(),
        "models": {
            "teacher": cfg.teacher_models,
            "judge": (cfg.judge_model or cfg.teacher_models[0]) if cfg.uses_judge else None,
            "embedding": cfg.embedding_model if cfg.enable_dedup else None,
        },
        "seed": seed,
        "pid": os.getpid(),
        "host": socket.gethostname(),
    }
    if owner is not None:
        fields["owner"] = owner
    run.update_manifest(**fields)
    tracker = _Tracker(run, progress_callback, cancel)

    with run_log(run.log, run.run_id):
        logger.info("Starting distillation", config=cfg.safe_dict(), seed=seed)
        try:
            result = _run(cfg, source_file, run, tracker, client_factory or _make_client, seed)
        except RunCancelled:
            run.update_manifest(status="cancelled", finished_at=utc_now(), stage="Cancelled")
            logger.info("Run cancelled")
            raise
        except BaseException as exc:
            run.update_manifest(status="failed", finished_at=utc_now(), error=str(exc)[:1000])
            logger.error("Run failed", error=str(exc)[:1000])
            raise
        run.update_manifest(status="succeeded", finished_at=utc_now(), stage="Done", progress=100)
        logger.info("Finished", path=str(result.dataset_path))
    return result


def _run(
    cfg: DistillationConfig,
    source_file: Path,
    run: RunDir,
    t: _Tracker,
    make_client: ClientFactory,
    seed: int,
) -> RunResult:
    """The pipeline stages in order; each one is timed and recorded by *t*."""
    with t.step("read", 2, "Reading documents"):
        text = _read_source(source_file, run)

    with t.step("chunk", 5, "Splitting into chunks"):
        chunks = source_chunks(text, cfg.use_semantic_chunking)
        logger.info("Document chunked", chunks=len(chunks))
        run.update_manifest(counts={"chunks": len(chunks)})

    records = _generate_stage(cfg, chunks, run, t, make_client, seed)
    counts: dict[str, int] = {"chunks": len(chunks), "generated": len(records)}

    if cfg.decontaminate:
        with t.step("decontaminate", 76, "Removing benchmark overlap"):
            records = _decontaminate(records, cfg, run)
            counts["after_decontamination"] = len(records)
    if cfg.sanitize_dataset:
        with t.step("sanitize", 80, "Cleaning and removing PII"):
            records = _sanitize(records, cfg, run)
            counts["after_sanitize"] = len(records)

    with t.step("export", 85, "Exporting dataset"):
        write_records(run.records, records)
        quality = score_records(records)
        dataset_path = run.dataset(cfg.output_format.value)
        counts["exported"] = export_dataset(records, dataset_path, cfg.output_format.value)
        run.update_manifest(counts=counts, quality=dict(quality), dataset_file=dataset_path.name)
        logger.info("Dataset exported", path=str(dataset_path), records=counts["exported"])

    adapter_zip = _train_stage(cfg, run, t) if cfg.train_model else None
    published_repo = _publish_stage(cfg, run, t, dataset_path)

    t.progress(100, "Done")
    return RunResult(
        run=run,
        dataset_path=dataset_path,
        record_count=counts["exported"],
        quality=quality,
        adapter_zip=adapter_zip,
        published_repo=published_repo,
    )


def _read_source(source_file: Path, run: RunDir) -> str:
    """Copy the source text into the run folder (checking its size) and return it."""
    source_bytes = source_file.stat().st_size
    if source_bytes > MAX_SOURCE_BYTES:
        raise ValueError(
            f"Source file is {source_bytes / 1e6:.0f} MB — exceeds the 100 MB limit. "
            "Split the document into smaller files and run multiple times."
        )
    if source_file.resolve() != run.source.resolve():
        shutil.copyfile(source_file, run.source)
    return run.source.read_text(encoding="utf-8")


def _generate_stage(
    cfg: DistillationConfig,
    chunks: list[str],
    run: RunDir,
    t: _Tracker,
    make_client: ClientFactory,
    seed: int,
) -> list[Record]:
    """Grounded generation, answer filters, judge and dedup; fails if nothing is left."""
    label = "Writing questions, answering, judging"

    def on_progress(fraction: float) -> None:
        t.progress(15 + int(fraction * 60), label)

    with t.step("generate", 15, label):
        records, stats, usage = asyncio.run(_generate(cfg, chunks, on_progress, make_client, seed, t.cancel))
        write_records(run.raw, records)
        write_records(run.rejected, stats.rejected)
        run.update_manifest(counts={"chunks": len(chunks), "generated": len(records)},
                            generation=stats.as_dict(), usage=usage,
                            cost_usd=run_cost(usage, is_local(cfg.base_url)))
        logger.info("Generation finished", **{k: v for k, v in stats.as_dict().items() if k != "answers_filtered"})
    if not records:
        raise RuntimeError(
            "No usable question/answer pairs were produced "
            f"({stats.errors} failed requests, {stats.judge_rejected} rejected by the judge, "
            f"{sum(stats.answers_filtered.values())} filtered answers)"
            + (f"; last error: {stats.last_error}" if stats.last_error else "")
            + ". Check the model name, API key / endpoint and the logs, then try again."
        )
    return records


def _train_stage(cfg: DistillationConfig, run: RunDir, t: _Tracker) -> Path:
    """Train the LoRA adapter on the canonical records (one GPU holder at a time); return its zip."""
    with t.step("train", 88, "Training LoRA adapter"):
        from training.lora_trainer import train_lora

        def waiting() -> None:
            t.progress(88, "Waiting for the GPU (another run is training)")
            run.update_manifest(stage=t.stage)

        with gpu_slot(t.cancel, on_wait=waiting):
            t.progress(88, "Training LoRA adapter")
            train_lora(run.records, cfg.base_model, run.adapter_dir, cfg.lora_rank)
        adapter_zip = Path(shutil.make_archive(str(run.adapter_zip.with_suffix("")), "zip", root_dir=run.adapter_dir))
        run.update_manifest(adapter_file=adapter_zip.name)
    return adapter_zip


def _publish_stage(cfg: DistillationConfig, run: RunDir, t: _Tracker, dataset_path: Path) -> str | None:
    """Publish to Hugging Face: dataset card first, then the data; then the adapter."""
    if not ((cfg.publish_dataset and cfg.hf_repo) or cfg.publish_adapter):
        return None
    published_repo: str | None = None
    with t.step("publish", 96, "Publishing to Hugging Face"):
        from publish.dataset_card import dataset_card, model_card
        from publish.hf_publisher import publish_adapter, publish_dataset

        manifest = run.read_manifest()
        if cfg.publish_dataset and cfg.hf_repo:
            card = dataset_card(cfg.hf_repo, manifest, dataset_path, run.source, cfg.dataset_license)
            publish_dataset(str(dataset_path), cfg.hf_repo, cfg.hf_token, private=cfg.hf_private, card=card)
            published_repo = cfg.hf_repo
            run.update_manifest(published_repo=published_repo, published_private=cfg.hf_private)
        if cfg.publish_adapter and cfg.model_repo:
            card = model_card(cfg.model_repo, manifest, published_repo, cfg.dataset_license)
            publish_adapter(run.adapter_dir, cfg.model_repo, cfg.hf_token, private=cfg.hf_private, card=card)
            run.update_manifest(published_model_repo=cfg.model_repo)
    return published_repo
