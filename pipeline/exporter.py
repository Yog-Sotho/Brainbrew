"""
Brainbrew exporter: formats canonical records for training.

Formatting happens only here, at the very end of the pipeline:
  - Alpaca:   {"instruction": ..., "input": ..., "output": ...}
  - ShareGPT: {"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}
  - ChatML:   {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
  - OpenAI:   {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}

Deduplication lives in pipeline/dedup.py.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pipeline.records import Record, jsonl_line

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
            fout.write(jsonl_line(json.dumps(formatter(rec), ensure_ascii=False)))
            count += 1
    return count
