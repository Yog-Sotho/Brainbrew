"""
Runs the distilabel generation pipeline in a child process.

Why a child process:
  * distilabel's Pipeline.run installs a SIGINT handler, which Python allows
    only on a process's main thread. Streamlit runs scripts on worker threads,
    so running the pipeline in the server process always fails.
  * distilabel has process-wide side effects: it replaces the root logger's
    handlers and forks worker processes. Those stay out of the long-lived
    server.

This module must not import distilabel at the top level. The child is spawned
(fresh interpreter, single thread) and sets the start method distilabel
normally runs with *before* the first distilabel import; distilabel fixes its
worker-pool context at import time and creates its queues from the default
context at run time, so the two have to agree.
"""
from __future__ import annotations

import json
import multiprocessing
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from distilabel.pipeline import Pipeline


def _distilabel_start_method() -> str:
    """The start method distilabel runs with in a normal script on this platform."""
    return "fork" if sys.platform.startswith("linux") else "spawn"


def build_pipeline(
    prompts: list[str],
    llm: Any,
    num_evolutions: int,
    batch_size: int,
    name: str,
    cache_dir: Path,
) -> Pipeline:
    """Seed prompts -> Evol-Instruct -> answer the evolved instruction -> canonical rows."""
    from distilabel.pipeline import Pipeline
    from distilabel.steps import KeepColumns, LoadDataFromDicts
    from distilabel.steps.tasks import EvolInstruct, TextGeneration

    from pipeline.steps import ToCanonical

    with Pipeline(name=name, cache_dir=str(cache_dir)) as pipeline:
        loader = LoadDataFromDicts(
            data=[{"instruction": p} for p in prompts],
            batch_size=batch_size,
        )
        evol = EvolInstruct(llm=llm, num_evolutions=num_evolutions, input_batch_size=batch_size)
        gen = TextGeneration(
            llm=llm,
            input_mappings={"instruction": "evolved_instruction"},
            input_batch_size=batch_size,
        )
        canonical = ToCanonical()
        keep = KeepColumns(columns=["instruction", "output", "seed", "model_name"])

        loader >> evol >> gen >> canonical >> keep
    return pipeline


def run_pipeline(pipeline: Pipeline) -> list[dict[str, Any]]:
    distiset = pipeline.run(use_cache=False)
    try:
        return list(distiset["default"]["train"].to_list())
    except KeyError:  # every row was filtered out
        return []


def _worker(
    llm_pickle: bytes,
    prompts: list[str],
    num_evolutions: int,
    batch_size: int,
    name: str,
    cache_dir: Path,
    rows_path: Path,
) -> None:
    """Child-process entry point: run the pipeline and write its rows as JSONL."""
    multiprocessing.set_start_method(_distilabel_start_method(), force=True)
    try:
        llm = pickle.loads(llm_pickle)  # first distilabel import happens here
        rows = run_pipeline(build_pipeline(prompts, llm, num_evolutions, batch_size, name, cache_dir))
        with open(rows_path, "w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except BaseException as exc:
        rows_path.with_suffix(".error").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        raise SystemExit(1) from exc


def generate_rows(
    prompts: list[str],
    llm: Any,
    num_evolutions: int,
    batch_size: int,
    name: str,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Run one generation pipeline in a spawned child process and return its rows.

    Raises:
        RuntimeError: the child failed; the message includes the child's error.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows_path = cache_dir / f"{name}.rows.jsonl"
    error_path = rows_path.with_suffix(".error")
    proc = multiprocessing.get_context("spawn").Process(
        target=_worker,
        # The LLM goes over as bytes so the child can set its start method
        # before unpickling it imports distilabel.
        args=(pickle.dumps(llm), prompts, num_evolutions, batch_size, name, cache_dir, rows_path),
        name=name,
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        detail = (
            error_path.read_text(encoding="utf-8") if error_path.exists()
            else f"the generation process exited with code {proc.exitcode}"
        )
        raise RuntimeError(f"Generation failed: {detail}")
    with open(rows_path, encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]
