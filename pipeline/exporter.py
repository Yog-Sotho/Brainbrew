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

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pipeline.records import Record

logger = logging.getLogger(__name__)


# ── Enhancement 5: Deduplication ─────────────────────────────────────────────

def _ngram_shingles(text: str, n: int = 3) -> set[str]:
    """Return set of character n-gram shingles for Jaccard similarity."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two shingle sets.

    ⚡ Optimization: Replaces the costly set union (a | b) with set length arithmetic
    to avoid allocating a new set, hashing its elements, and copying values.
    """
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_records(
    records: list[Record],
    similarity_threshold: float = 0.85,
) -> list[Record]:
    """Remove exact and near-duplicate records.

    Strategy:
      1. Exact dedup via instruction+output hash.
      2. Near-dedup via Jaccard similarity on character trigram shingles.

    ⚡ Optimization: Uses mathematical bounding to skip Jaccard set operations entirely
    for pairs that cannot possibly be duplicates. Jaccard similarity is upper-bounded
    by min(|A|, |B|) / max(|A|, |B|). Since combined similarity is the average of
    instruction and output similarities, both must be >= 2 * threshold - 1.0.

    Args:
        records: Canonical records.
        similarity_threshold: Jaccard threshold above which records are
                              considered duplicates (default 0.85).

    Returns:
        Deduplicated list in original order.
    """
    if not records:
        return records

    seen_hashes: set[str] = set()
    unique: list[Record] = []
    # Store shingles alongside their lengths for O(1) ratio pruning
    shingle_index: list[tuple[set[str], set[str], int, int]] = []

    # Precalculate minimum similarity required on either field to meet the combined threshold
    min_sim = 2.0 * similarity_threshold - 1.0

    for rec in records:
        # Step 1: exact hash dedup
        content_key = f"{rec.instruction}|||{rec.output}"
        content_hash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        # Step 2: near-duplicate via shingle Jaccard
        inst_shingles = _ngram_shingles(rec.instruction)
        out_shingles = _ngram_shingles(rec.output)
        len_inst = len(inst_shingles)
        len_out = len(out_shingles)

        is_near_dup = False
        for existing_inst, existing_out, len_exist_inst, len_exist_out in shingle_index:
            # Pruning Check 1: Instruction shingle ratio bound
            if len_inst == 0 or len_exist_inst == 0:
                inst_ratio = 0.0
            else:
                inst_ratio = (len_inst / len_exist_inst) if len_inst < len_exist_inst else (len_exist_inst / len_inst)
            if inst_ratio < min_sim:
                continue

            # Pruning Check 2: Output shingle ratio bound
            if len_out == 0 or len_exist_out == 0:
                out_ratio = 0.0
            else:
                out_ratio = (len_out / len_exist_out) if len_out < len_exist_out else (len_exist_out / len_out)
            if out_ratio < min_sim:
                continue

            # Pruning Check 3: Actual instruction similarity bound
            inst_sim = _jaccard_similarity(inst_shingles, existing_inst)
            if inst_sim < min_sim:
                continue

            # If all bounds pass, compute final Jaccard similarity
            out_sim = _jaccard_similarity(out_shingles, existing_out)
            combined = (inst_sim + out_sim) / 2.0
            if combined >= similarity_threshold:
                is_near_dup = True
                break

        if not is_near_dup:
            unique.append(rec)
            shingle_index.append((inst_shingles, out_shingles, len_inst, len_out))

    removed = len(records) - len(unique)
    if removed > 0:
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
