"""
Benchmark decontamination.

Drops generated records that share a long word n-gram with a public eval set
the user selected, so a model trained on the dataset is not evaluated on text
it has seen. Uses the 13-gram overlap rule from the GPT-3 paper; eval items
shorter than 13 words are matched whole (items under 8 words are ignored as
too generic to signal contamination).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import xxhash

from pipeline.records import Record

NGRAM = 13
MIN_WORDS = 8


@dataclass(frozen=True)
class EvalSet:
    label: str
    repo: str
    config: str
    split: str
    fields: tuple[str, ...]  # "a.b" reads a nested field; lists are flattened


EVAL_SETS: dict[str, EvalSet] = {
    "gsm8k": EvalSet("GSM8K", "openai/gsm8k", "main", "test", ("question", "answer")),
    "mmlu": EvalSet("MMLU", "cais/mmlu", "all", "test", ("question", "choices")),
    "arc": EvalSet("ARC-Challenge", "allenai/ai2_arc", "ARC-Challenge", "test", ("question", "choices.text")),
    "truthfulqa": EvalSet("TruthfulQA", "truthfulqa/truthful_qa", "generation", "validation",
                          ("question", "best_answer")),
    "humaneval": EvalSet("HumanEval", "openai/openai_humaneval", "openai_humaneval", "test",
                         ("prompt", "canonical_solution")),
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _grams(toks: Sequence[str], n: int) -> Iterable[int]:
    for i in range(len(toks) - n + 1):
        yield xxhash.xxh64_intdigest(" ".join(toks[i:i + n]).encode())


class EvalIndex:
    """Hashed n-grams of one or more eval sets."""

    def __init__(self) -> None:
        self._grams: dict[int, set[int]] = {}  # n -> gram hashes

    def add(self, text: str) -> None:
        toks = words(text)
        if len(toks) < MIN_WORDS:
            return
        n = min(NGRAM, len(toks))
        self._grams.setdefault(n, set()).update(_grams(toks, n))

    def __len__(self) -> int:
        return sum(len(g) for g in self._grams.values())

    def overlaps(self, text: str) -> bool:
        toks = words(text)
        return any(
            any(g in grams for g in _grams(toks, n))
            for n, grams in self._grams.items()
            if len(toks) >= n
        )


def _field(row: dict[str, Any], path: str) -> list[str]:
    value: Any = row
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def load_eval_texts(key: str) -> list[str]:
    """Download (or read from the HF cache) the texts of one eval set."""
    from datasets import load_dataset

    spec = EVAL_SETS[key]
    try:
        rows = load_dataset(spec.repo, spec.config, split=spec.split)
    except Exception as exc:  # network, auth, missing dataset
        raise RuntimeError(
            f"Could not download the {spec.label} benchmark ({spec.repo}) for decontamination: {exc}"
        ) from exc
    texts: list[str] = []
    for row in rows:
        item = " ".join(t for f in spec.fields for t in _field(row, f))
        if item:
            texts.append(item)
    return texts


def build_index(key: str, loader: Callable[[str], list[str]] = load_eval_texts) -> EvalIndex:
    index = EvalIndex()
    for text in loader(key):
        index.add(text)
    return index


def decontaminate(
    records: list[Record],
    keys: Sequence[str],
    loader: Callable[[str], list[str]] = load_eval_texts,
) -> tuple[list[Record], dict[str, int]]:
    """Drop records overlapping any selected eval set; returns (kept, removed per set)."""
    indexes = {key: build_index(key, loader) for key in keys}
    removed = dict.fromkeys(keys, 0)
    kept: list[Record] = []
    for rec in records:
        text = "\n".join((rec.instruction, rec.input, rec.output))
        hit = next((key for key, index in indexes.items() if index.overlaps(text)), None)
        if hit is None:
            kept.append(rec)
        else:
            removed[hit] += 1
    return kept, removed
