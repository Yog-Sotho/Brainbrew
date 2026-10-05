"""
Brainbrew exporter — deduplicates canonical records and formats them for training.

Formatting happens only here, at the very end of the pipeline:
  - Alpaca:   {"instruction": ..., "input": ..., "output": ...}
  - ShareGPT: {"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}
  - ChatML:   {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
  - OpenAI:   {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}

Also provides exact-match and near-duplicate deduplication (Enhancement 5).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pipeline.dedup import deduplicate
from pipeline.records import Record

logger = logging.getLogger(__name__)


# ── Enhancement 5: Deduplication (MinHash-LSH, pipeline/dedup.py) ───────────

def deduplicate_records(
    records: list[Record],
    similarity_threshold: float = 0.85,
) -> list[Record]:
    """Remove exact and near-duplicate records, keeping the first of each group.

    Near-duplicates are found with MinHash + LSH over the normalised
    instruction and output, so the cost grows roughly linearly with the data.
    """
    if not records:
        return records
    unique = deduplicate(records, threshold=similarity_threshold)
    removed = len(records) - len(unique)
    if removed:
        logger.info("Deduplication removed %d records (%d → %d)", removed, len(records), len(unique))
    return unique


# ── Format converters ────────────────────────────────────────────────────────

OPENAI_SYSTEM_PROMPT = "You are a helpful assistant."


def to_alpaca(rec: Record) -> dict[str, Any]:
    return {"instruction": rec.instruction, "input": rec.input, "output": rec.output}


def to_sharegpt(rec: Record) -> dict[str, Any]:
    return {
        "conversations": [
            {"from": "human", "value": rec.prompt},
            {"from": "gpt", "value": rec.output},
        ]
    }


def to_chatml(rec: Record) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": rec.prompt},
            {"role": "assistant", "content": rec.output},
        ]
    }


def to_openai(rec: Record) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
            {"role": "user", "content": rec.prompt},
            {"role": "assistant", "content": rec.output},
        ]
    }


FORMATTERS: dict[str, Callable[[Record], dict[str, Any]]] = {
    "alpaca": to_alpaca,
    "sharegpt": to_sharegpt,
    "chatml": to_chatml,
    "openai": to_openai,
}


# ── Public API ───────────────────────────────────────────────────────────────

def export_dataset(
    records: Iterable[Record],
    output_path: Path,
    output_format: str = "alpaca",
) -> int:
    """Write canonical records to *output_path* in the chosen training format.

    Returns:
        Number of records written.
    """
    formatter = FORMATTERS.get(output_format)
    if formatter is None:
        raise ValueError(
            f"Unknown output format: {output_format!r}. "
            f"Supported: {', '.join(FORMATTERS)}"
        )
    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(formatter(rec), ensure_ascii=False) + "\n")
            count += 1
    return count
