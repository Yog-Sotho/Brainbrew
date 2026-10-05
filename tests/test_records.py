"""
tests/test_records.py

The canonical Record model and its JSONL reader/writer.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pipeline.records import Record, read_records, write_records


class TestRecord:

    def test_minimal_record(self):
        rec = Record(instruction="Q?", output="A.")
        assert rec.input == "" and rec.meta == {}

    @pytest.mark.parametrize("field", ["instruction", "output"])
    @pytest.mark.parametrize("value", ["", "   \n\t"])
    def test_blank_required_fields_rejected(self, field, value):
        kwargs = {"instruction": "Q?", "output": "A.", field: value}
        with pytest.raises(ValidationError):
            Record(**kwargs)

    def test_unknown_fields_rejected(self):
        with pytest.raises(ValidationError):
            Record(instruction="Q?", output="A.", conversations=[])

    def test_prompt_without_input(self):
        assert Record(instruction="Q?", output="A.").prompt == "Q?"

    def test_prompt_with_input(self):
        assert Record(instruction="Q?", input="ctx", output="A.").prompt == "Q?\n\nctx"

    def test_whitespace_only_input_ignored_in_prompt(self):
        assert Record(instruction="Q?", input="  ", output="A.").prompt == "Q?"


class TestReadWrite:

    def test_round_trip(self, tmp_path):
        recs = [Record(instruction=f"Q{i}?", output=f"A{i}.", meta={"seed": i}) for i in range(3)]
        path = tmp_path / "r.jsonl"
        assert write_records(path, recs) == 3
        assert read_records(path) == recs

    def test_invalid_lines_skipped(self, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text(
            "\n".join([
                json.dumps({"instruction": "Q?", "output": "A."}),
                "{not json",
                "",
                json.dumps({"instruction": "", "output": "A."}),
                json.dumps({"instruction": "Q?"}),
                json.dumps({"instruction": "Q2?", "output": "A2."}),
            ]),
            encoding="utf-8",
        )
        assert [r.instruction for r in read_records(path)] == ["Q?", "Q2?"]

    def test_max_records_limit(self, tmp_path):
        path = tmp_path / "r.jsonl"
        write_records(path, [Record(instruction=f"Q{i}?", output="A.") for i in range(10)])
        assert len(read_records(path, max_records=4)) == 4

    def test_unicode_round_trip(self, tmp_path):
        path = tmp_path / "r.jsonl"
        rec = Record(instruction="¿Qué es? 🤖", output="人工智能")
        write_records(path, [rec])
        assert read_records(path) == [rec]
