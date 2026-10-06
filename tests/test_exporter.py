"""
tests/test_exporter.py

Formatting of canonical records into the four export formats, and dedup before export.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pipeline.dedup import deduplicate
from pipeline.exporter import (
    FORMATTERS,
    OPENAI_SYSTEM_PROMPT,
    export_dataset,
)
from pipeline.records import Record


def _rec(instruction: str, output: str, input: str = "", **meta) -> Record:
    return Record(instruction=instruction, input=input, output=output, meta=meta)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


RECORDS = [
    _rec("What is AI?", "AI stands for Artificial Intelligence.", seed="s1"),
    _rec("Summarise this.", "It is about transformers.", input="Transformers use attention."),
]


class TestFormats:

    def test_alpaca(self, tmp_path):
        out = tmp_path / "a.jsonl"
        assert export_dataset(RECORDS, out, "alpaca") == 2
        rows = _read(out)
        assert rows[0] == {"instruction": "What is AI?", "input": "",
                           "output": "AI stands for Artificial Intelligence."}
        assert rows[1]["input"] == "Transformers use attention."

    def test_sharegpt(self, tmp_path):
        out = tmp_path / "s.jsonl"
        export_dataset(RECORDS, out, "sharegpt")
        rows = _read(out)
        assert rows[0] == {"conversations": [
            {"from": "human", "value": "What is AI?"},
            {"from": "gpt", "value": "AI stands for Artificial Intelligence."},
        ]}
        assert rows[1]["conversations"][0]["value"] == "Summarise this.\n\nTransformers use attention."

    def test_chatml(self, tmp_path):
        out = tmp_path / "c.jsonl"
        export_dataset(RECORDS, out, "chatml")
        msgs = _read(out)[1]["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "Summarise this.\n\nTransformers use attention."

    def test_openai(self, tmp_path):
        out = tmp_path / "o.jsonl"
        export_dataset(RECORDS, out, "openai")
        msgs = _read(out)[0]["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[0]["content"] == OPENAI_SYSTEM_PROMPT

    @pytest.mark.parametrize("fmt", sorted(FORMATTERS))
    def test_meta_never_exported(self, fmt, tmp_path):
        out = tmp_path / f"{fmt}.jsonl"
        export_dataset(RECORDS, out, fmt)
        assert "seed" not in out.read_text(encoding="utf-8")
        assert "s1" not in out.read_text(encoding="utf-8")

    def test_invalid_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown output format"):
            export_dataset(RECORDS, tmp_path / "x.jsonl", "parquet")

    def test_empty_input_creates_empty_file(self, tmp_path):
        out = tmp_path / "empty.jsonl"
        assert export_dataset([], out, "alpaca") == 0
        assert out.read_text(encoding="utf-8") == ""

    def test_unicode_preserved(self, tmp_path):
        out = tmp_path / "u.jsonl"
        export_dataset([_rec("Qu'est-ce que l'IA ? 🤖", "L'intelligence artificielle — 人工智能")], out, "alpaca")
        text = out.read_text(encoding="utf-8")
        assert "🤖" in text and "人工智能" in text  # ensure_ascii=False

    def test_export_is_idempotent(self, tmp_path):
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        export_dataset(RECORDS, a, "sharegpt")
        export_dataset(RECORDS, b, "sharegpt")
        assert a.read_bytes() == b.read_bytes()


class TestDeduplication:

    def test_exact_duplicates_removed(self):
        recs = [_rec("What is AI?", "Artificial Intelligence.")] * 3
        assert len(deduplicate(recs)) == 1

    def test_near_duplicates_removed(self):
        recs = [
            _rec("What is machine learning?", "Machine learning is a subset of AI that learns from data."),
            _rec("What is machine learning ?", "Machine learning is a subset of AI that learns from data!"),
        ]
        assert len(deduplicate(recs)) == 1

    def test_empty_input_returns_empty(self):
        assert deduplicate([]) == []

    def test_unique_records_preserved_in_order(self):
        recs = [
            _rec("What is AI?", "Artificial intelligence is the simulation of human intelligence."),
            _rec("Explain photosynthesis.", "Plants convert light energy into chemical energy."),
            _rec("Who wrote Hamlet?", "William Shakespeare wrote Hamlet around 1600."),
        ]
        assert deduplicate(recs) == recs

    def test_meta_kept_on_survivors(self):
        recs = [_rec("Q?", "A long enough answer.", seed="first"), _rec("Q?", "A long enough answer.", seed="second")]
        assert deduplicate(recs)[0].meta == {"seed": "first"}


class TestPerformance:

    def test_ten_thousand_records(self, tmp_path):
        recs = [_rec(f"Question number {i}?", f"Answer number {i} with some detail.") for i in range(10_000)]
        start = time.perf_counter()
        assert export_dataset(recs, tmp_path / "big.jsonl", "chatml") == 10_000
        assert time.perf_counter() - start < 5.0
