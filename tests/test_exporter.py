"""
tests/test_exporter.py

Tests for pipeline/exporter.py — export_dataset(), export_alpaca(),
deduplicate_records(), and all four output formats.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.exporter import (
    export_alpaca,
    export_dataset,
    deduplicate_records,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Alpaca format (backward compatibility via export_alpaca)
# ---------------------------------------------------------------------------

class TestExportAlpaca:

    def test_output_column_exported_correctly(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [{"instruction": "What is AI?", "output": "AI is a field of computer science."}])
        export_alpaca(str(src), str(out))
        records = read_jsonl(out)
        assert len(records) == 1
        assert records[0]["instruction"] == "What is AI?"
        assert records[0]["output"] == "AI is a field of computer science."
        assert records[0]["input"] == ""

    def test_generation_column_used_as_fallback(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [{"instruction": "Explain ML.", "generation": "ML is a subset of AI."}])
        export_alpaca(str(src), str(out))
        records = read_jsonl(out)
        assert len(records) == 1
        assert records[0]["output"] == "ML is a subset of AI."

    def test_all_valid_records_exported(self, raw_jsonl_file, tmp_path):
        out = tmp_path / "alpaca.jsonl"
        export_alpaca(str(raw_jsonl_file), str(out))
        records = read_jsonl(out)
        assert len(records) == 5


# ---------------------------------------------------------------------------
# Multi-format export (Enhancement 6)
# ---------------------------------------------------------------------------

class TestMultiFormatExport:

    @pytest.fixture()
    def src_file(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        write_jsonl(src, [
            {"instruction": "What is AI?", "output": "Artificial Intelligence is a field of CS."},
            {"instruction": "Explain ML.", "output": "Machine Learning is a subset of AI."},
        ])
        return src

    def test_alpaca_format(self, src_file, tmp_path):
        out = tmp_path / "out.jsonl"
        count = export_dataset(str(src_file), str(out), output_format="alpaca", enable_dedup=False)
        records = read_jsonl(out)
        assert count == 2
        assert set(records[0].keys()) == {"instruction", "input", "output"}

    def test_sharegpt_format(self, src_file, tmp_path):
        out = tmp_path / "out.jsonl"
        count = export_dataset(str(src_file), str(out), output_format="sharegpt", enable_dedup=False)
        records = read_jsonl(out)
        assert count == 2
        assert "conversations" in records[0]
        assert records[0]["conversations"][0]["from"] == "human"
        assert records[0]["conversations"][1]["from"] == "gpt"

    def test_chatml_format(self, src_file, tmp_path):
        out = tmp_path / "out.jsonl"
        count = export_dataset(str(src_file), str(out), output_format="chatml", enable_dedup=False)
        records = read_jsonl(out)
        assert count == 2
        assert "messages" in records[0]
        assert records[0]["messages"][0]["role"] == "user"
        assert records[0]["messages"][1]["role"] == "assistant"

    def test_openai_format(self, src_file, tmp_path):
        out = tmp_path / "out.jsonl"
        count = export_dataset(str(src_file), str(out), output_format="openai", enable_dedup=False)
        records = read_jsonl(out)
        assert count == 2
        assert "messages" in records[0]
        roles = [m["role"] for m in records[0]["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_invalid_format_raises(self, src_file, tmp_path):
        out = tmp_path / "out.jsonl"
        with pytest.raises(ValueError, match="Unknown output format"):
            export_dataset(str(src_file), str(out), output_format="invalid")


# ---------------------------------------------------------------------------
# Deduplication (Enhancement 5)
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_exact_duplicates_removed(self):
        records = [
            {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
            {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
            {"instruction": "What is ML?", "input": "", "output": "ML is machine learning."},
        ]
        result = deduplicate_records(records)
        assert len(result) == 2

    def test_near_duplicates_removed(self):
        records = [
            {"instruction": "What is artificial intelligence?", "input": "",
             "output": "Artificial intelligence is a branch of computer science."},
            {"instruction": "What is artificial intelligence?", "input": "",
             "output": "Artificial intelligence is a branch of computer science that studies."},
            {"instruction": "Completely different question?", "input": "",
             "output": "Completely different answer about a totally separate topic."},
        ]
        result = deduplicate_records(records, similarity_threshold=0.85)
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        assert deduplicate_records([]) == []

    def test_unique_records_preserved(self):
        records = [
            {"instruction": f"Question {i}?", "input": "", "output": f"Answer {i}."}
            for i in range(10)
        ]
        result = deduplicate_records(records)
        assert len(result) == 10

    def test_export_dataset_with_dedup(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "out.jsonl"
        write_jsonl(src, [
            {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
            {"instruction": "What is AI?", "output": "AI is artificial intelligence."},
            {"instruction": "What is ML?", "output": "ML is machine learning."},
        ])
        count = export_dataset(str(src), str(out), enable_dedup=True)
        assert count == 2


# ---------------------------------------------------------------------------
# Resilience / skip logic
# ---------------------------------------------------------------------------

class TestSkipBehaviour:

    def test_malformed_json_line_skipped(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "out.jsonl"
        src.write_text(
            '{"instruction": "Valid?", "output": "Yes."}\n'
            'NOT JSON\n'
            '{"instruction": "Also valid?", "output": "Also yes."}\n',
            encoding="utf-8",
        )
        count = export_dataset(str(src), str(out), enable_dedup=False)
        assert count == 2

    def test_empty_lines_skipped(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "out.jsonl"
        src.write_text(
            "\n\n"
            '{"instruction": "Q?", "output": "Answer."}\n'
            "\n",
            encoding="utf-8",
        )
        count = export_dataset(str(src), str(out), enable_dedup=False)
        assert count == 1

    def test_empty_instruction_skipped(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "out.jsonl"
        write_jsonl(src, [
            {"instruction": "", "output": "No instruction."},
            {"instruction": "Real?", "output": "Yes."},
        ])
        count = export_dataset(str(src), str(out), enable_dedup=False)
        assert count == 1

    def test_output_file_created_even_if_all_skipped(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "out.jsonl"
        src.write_text("NOT JSON\n", encoding="utf-8")
        export_dataset(str(src), str(out), enable_dedup=False)
        assert out.exists()
        assert read_jsonl(out) == []


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestEncoding:

    def test_unicode_preserved(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "out.jsonl"
        write_jsonl(src, [{"instruction": "Explain in Japanese", "output": "AI is important."}])
        export_dataset(str(src), str(out), enable_dedup=False)
        records = read_jsonl(out)
        assert records[0]["instruction"] == "Explain in Japanese"


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:

    def test_ten_thousand_records(self, tmp_path):
        import time
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "out.jsonl"
        records = [
            {"instruction": f"Question {i}?", "output": f"Answer {i} here."}
            for i in range(10_000)
        ]
        src.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        start = time.monotonic()
        count = export_dataset(str(src), str(out), enable_dedup=False)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"10000 records took {elapsed:.2f}s"
        assert count == 10_000


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_export_is_idempotent(raw_jsonl_file, tmp_path):
    out1 = tmp_path / "run1.jsonl"
    out2 = tmp_path / "run2.jsonl"
    export_alpaca(str(raw_jsonl_file), str(out1))
    export_alpaca(str(raw_jsonl_file), str(out2))
    assert out1.read_text() == out2.read_text()
