"""
tests/test_exporter.py

Tests for pipeline/exporter.py — export_alpaca().

Covers:
  - Happy path with 'output' column
  - Happy path with 'generation' column (distilabel raw)
  - Mixed 'output' and 'generation' columns in same file
  - Malformed JSON lines skipped without crash
  - Empty lines skipped silently
  - Records with empty instruction skipped
  - Records with empty output AND empty generation skipped
  - Output file is valid JSONL (each line is valid JSON)
  - Output records have exactly the Alpaca schema: instruction, input, output
  - 'input' field is always empty string
  - Large file performance (thousands of records)
  - File is created even if all records are skipped
  - Encoding: UTF-8 with non-ASCII characters preserved
  - Idempotency: running twice produces same output
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.exporter import export_alpaca


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def read_alpaca(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_output_column_exported_correctly(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [{"instruction": "What is AI?", "output": "AI is a field of computer science."}])
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert len(records) == 1
        assert records[0]["instruction"] == "What is AI?"
        assert records[0]["output"] == "AI is a field of computer science."

    def test_generation_column_used_as_fallback(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [{"instruction": "Explain ML.", "generation": "ML is a subset of AI."}])
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert len(records) == 1
        assert records[0]["output"] == "ML is a subset of AI."

    def test_output_takes_precedence_over_generation(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [{"instruction": "Q?", "output": "correct answer", "generation": "wrong answer"}])
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert records[0]["output"] == "correct answer"

    def test_input_field_always_empty_string(self, raw_jsonl_file, tmp_path):
        out = tmp_path / "alpaca.jsonl"
        export_alpaca(str(raw_jsonl_file), str(out))
        records = read_alpaca(out)
        assert all(r["input"] == "" for r in records), "All 'input' fields must be empty string"

    def test_all_valid_records_are_exported(self, raw_jsonl_file, tmp_path):
        out = tmp_path / "alpaca.jsonl"
        export_alpaca(str(raw_jsonl_file), str(out))
        records = read_alpaca(out)
        assert len(records) == 5  # raw_jsonl_file fixture has 5 records


# ---------------------------------------------------------------------------
# Alpaca schema enforcement
# ---------------------------------------------------------------------------

class TestAlpacaSchema:

    def test_output_records_have_exactly_three_keys(self, raw_jsonl_file, tmp_path):
        out = tmp_path / "alpaca.jsonl"
        export_alpaca(str(raw_jsonl_file), str(out))
        records = read_alpaca(out)
        for rec in records:
            assert set(rec.keys()) == {"instruction", "input", "output"}, (
                f"Expected exactly {{instruction, input, output}}, got {set(rec.keys())}"
            )

    def test_instruction_field_preserved_verbatim(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        original = "Explain convolutional neural networks in detail."
        write_jsonl(src, [{"instruction": original, "output": "CNNs use convolution operations."}])
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert records[0]["instruction"] == original


# ---------------------------------------------------------------------------
# Resilience / skip logic
# ---------------------------------------------------------------------------

class TestSkipBehaviour:

    def test_malformed_json_line_skipped_no_crash(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        src.write_text(
            '{"instruction": "Valid?", "output": "Yes, perfectly valid."}\n'
            'NOT JSON AT ALL\n'
            '{"instruction": "Also valid?", "output": "Also perfectly valid."}\n',
            encoding="utf-8",
        )
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert len(records) == 2

    def test_empty_lines_silently_skipped(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        src.write_text(
            "\n\n"
            '{"instruction": "Q?", "output": "Answer."}\n'
            "\n",
            encoding="utf-8",
        )
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert len(records) == 1

    def test_record_with_empty_instruction_skipped(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [
            {"instruction": "", "output": "An output without an instruction."},
            {"instruction": "Real question?", "output": "Real answer here."},
        ])
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert len(records) == 1
        assert records[0]["instruction"] == "Real question?"

    def test_record_with_empty_output_and_generation_skipped(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [
            {"instruction": "Q?", "output": "", "generation": ""},
            {"instruction": "Q2?", "output": "Non-empty answer here."},
        ])
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert len(records) == 1

    def test_record_missing_both_output_and_generation_skipped(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [
            {"instruction": "Q?"},  # no output or generation key at all
            {"instruction": "Q2?", "output": "Good answer."},
        ])
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert len(records) == 1

    def test_output_file_created_even_if_all_records_skipped(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        src.write_text("NOT JSON\n", encoding="utf-8")
        export_alpaca(str(src), str(out))
        assert out.exists()
        assert read_alpaca(out) == []


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestEncoding:

    def test_unicode_characters_preserved(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [{"instruction": "日本語で説明して", "output": "人工知能とは機械が人間の知能を模倣することです。"}])
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert records[0]["instruction"] == "日本語で説明して"
        assert "人工知能" in records[0]["output"]

    def test_emoji_in_text_preserved(self, tmp_path):
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        write_jsonl(src, [{"instruction": "🧠 What is Brainbrew?", "output": "🚀 A dataset generator!"}])
        export_alpaca(str(src), str(out))
        records = read_alpaca(out)
        assert "🧠" in records[0]["instruction"]
        assert "🚀" in records[0]["output"]


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:

    def test_ten_thousand_records_exported_within_reasonable_time(self, tmp_path):
        import time
        src = tmp_path / "raw.jsonl"
        out = tmp_path / "alpaca.jsonl"
        records = [
            {"instruction": f"Question number {i}?", "output": f"Answer number {i} is here."}
            for i in range(10_000)
        ]
        src.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        start = time.monotonic()
        export_alpaca(str(src), str(out))
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"10 000 records took {elapsed:.2f}s — too slow"
        exported = read_alpaca(out)
        assert len(exported) == 10_000


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_export_is_idempotent(raw_jsonl_file, tmp_path):
    out1 = tmp_path / "run1.jsonl"
    out2 = tmp_path / "run2.jsonl"
    export_alpaca(str(raw_jsonl_file), str(out1))
    export_alpaca(str(raw_jsonl_file), str(out2))
    assert out1.read_text() == out2.read_text()
