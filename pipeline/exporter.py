"""
Brainbrew exporter — converts raw distilabel output to training-ready formats.

Supports four output formats:
  - Alpaca:   {"instruction": ..., "input": "", "output": ...}
  - ShareGPT: {"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}
  - ChatML:   {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
  - OpenAI:   {"messages": [{"role": "system", ...}, {"role": "user", ...}, ...]}

Also provides exact-match and near-duplicate deduplication (Enhancement 5).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# FIX M-08: hard limit to prevent runaway memory on malformed input
MAX_RECORDS: int = 500_000


def _read_raw_records(
    input_path: str,
    max_records: int = MAX_RECORDS,
) -> list[dict]:
    """Stream-read JSONL and return valid instruction/output records.

    FIX L-08: reads line-by-line to avoid loading entire file into memory.
    FIX M-08: stops after max_records to prevent OOM on corrupted files.
    """
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            if line_num > max_records:
                logger.warning(
                    "Record limit reached (%d). Remaining lines skipped.", max_records
                )
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Support both 'output' (post-rename) and 'generation' (raw distilabel)
            output = obj.get("output") or obj.get("generation") or ""
            instruction = obj.get("instruction", "")
            if not instruction or not output:
                continue
            records.append({
                "instruction": instruction,
                "input": obj.get("input", ""),
                "output": output,
            })
    return records


# ── Enhancement 5: Deduplication ─────────────────────────────────────────────

def _ngram_shingles(text: str, n: int = 3) -> set[str]:
    """Return set of character n-gram shingles for Jaccard similarity."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two shingle sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def deduplicate_records(
    records: list[dict],
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """Remove exact and near-duplicate records.

    Strategy:
      1. Exact dedup via instruction+output hash.
      2. Near-dedup via Jaccard similarity on character trigram shingles.

    Args:
        records: List of {"instruction", "input", "output"} dicts.
        similarity_threshold: Jaccard threshold above which records are
                              considered duplicates (default 0.85).

    Returns:
        Deduplicated list in original order.
    """
    if not records:
        return records

    seen_hashes: set[str] = set()
    unique: list[dict] = []
    shingle_index: list[tuple[set[str], set[str]]] = []

    for rec in records:
        # Step 1: exact hash dedup
        content_key = f"{rec['instruction']}|||{rec['output']}"
        content_hash = hashlib.sha256(content_key.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        # Step 2: near-duplicate via shingle Jaccard
        inst_shingles = _ngram_shingles(rec["instruction"])
        out_shingles = _ngram_shingles(rec["output"])

        is_near_dup = False
        for existing_inst, existing_out in shingle_index:
            inst_sim = _jaccard_similarity(inst_shingles, existing_inst)
            out_sim = _jaccard_similarity(out_shingles, existing_out)
            combined = (inst_sim + out_sim) / 2.0
            if combined >= similarity_threshold:
                is_near_dup = True
                break

        if not is_near_dup:
            unique.append(rec)
            shingle_index.append((inst_shingles, out_shingles))

    removed = len(records) - len(unique)
    if removed > 0:
        logger.info("Deduplication removed %d records (%d → %d)", removed, len(records), len(unique))
    return unique


# ── Format converters ────────────────────────────────────────────────────────

def _to_alpaca(rec: dict) -> dict:
    return {
        "instruction": rec["instruction"],
        "input": rec.get("input", ""),
        "output": rec["output"],
    }


def _to_sharegpt(rec: dict) -> dict:
    conversations = [
        {"from": "human", "value": rec["instruction"]},
        {"from": "gpt", "value": rec["output"]},
    ]
    if rec.get("input"):
        conversations[0]["value"] += f"\n\nContext: {rec['input']}"
    return {"conversations": conversations}


def _to_chatml(rec: dict) -> dict:
    user_content = rec["instruction"]
    if rec.get("input"):
        user_content += f"\n\n{rec['input']}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": rec["output"]},
        ]
    }


def _to_openai(rec: dict) -> dict:
    user_content = rec["instruction"]
    if rec.get("input"):
        user_content += f"\n\n{rec['input']}"
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": rec["output"]},
        ]
    }


_FORMATTERS = {
    "alpaca": _to_alpaca,
    "sharegpt": _to_sharegpt,
    "chatml": _to_chatml,
    "openai": _to_openai,
}


# ── Public API ───────────────────────────────────────────────────────────────

def export_dataset(
    input_path: str,
    output_path: str,
    output_format: str = "alpaca",
    enable_dedup: bool = True,
    max_records: int = MAX_RECORDS,
) -> int:
    """Read raw JSONL, optionally deduplicate, and export in the chosen format.

    Args:
        input_path: Path to raw distilabel JSONL.
        output_path: Destination path for the formatted JSONL.
        output_format: One of 'alpaca', 'sharegpt', 'chatml', 'openai'.
        enable_dedup: Whether to run deduplication (Enhancement 5).
        max_records: Maximum records to process (FIX M-08).

    Returns:
        Number of records written.
    """
    formatter = _FORMATTERS.get(output_format)
    if formatter is None:
        raise ValueError(
            f"Unknown output format: {output_format!r}. "
            f"Supported: {', '.join(_FORMATTERS.keys())}"
        )

    records = _read_raw_records(input_path, max_records=max_records)

    if enable_dedup:
        records = deduplicate_records(records)

    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in records:
            formatted = formatter(rec)
            fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    return len(records)


# ── Backward-compatible alias ────────────────────────────────────────────────

def export_alpaca(input_path: str, output_path: str) -> None:
    """Legacy wrapper — calls export_dataset with alpaca format, no dedup."""
    export_dataset(input_path, output_path, output_format="alpaca", enable_dedup=False)
