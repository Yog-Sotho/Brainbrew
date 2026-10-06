"""
Canonical training record — the one data model every pipeline stage works on.

Generation writes canonical records, and dedup, sanitizing, quality scoring and
LoRA training all read them. Formatting to Alpaca / ShareGPT / ChatML / OpenAI
happens only in the final export (pipeline/exporter.py), so no stage has to
understand four different layouts.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

# Hard limit to prevent runaway memory on a corrupted file.
MAX_RECORDS: int = 500_000


class Record(BaseModel):
    """One instruction/response pair.

    `meta` carries provenance (seed prompt, teacher model) and is never part of
    the exported training text.
    """

    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    input: str = ""
    output: str = Field(min_length=1)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("instruction", "output")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @property
    def prompt(self) -> str:
        """The user turn: instruction, plus the input as context when present."""
        if self.input.strip():
            return f"{self.instruction}\n\n{self.input}"
        return self.instruction


# JSON leaves these unescaped, but str.splitlines() and many JSONL readers treat
# them as line breaks, which would split one record across two lines.
_LINE_BREAKS = str.maketrans({"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"})


def jsonl_line(json_text: str) -> str:
    """One JSONL line: *json_text* with every Unicode line break escaped, plus a newline."""
    return json_text.translate(_LINE_BREAKS) + "\n"


def write_records(path: Path, records: Iterable[Record]) -> int:
    """Write records as canonical JSONL. Returns the number written."""
    count = 0
    with open(path, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(jsonl_line(rec.model_dump_json()))
            count += 1
    return count


def read_records(path: Path, max_records: int = MAX_RECORDS) -> list[Record]:
    """Read canonical JSONL, skipping (and counting) lines that are not valid records."""
    records: list[Record] = []
    skipped = 0
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            if len(records) >= max_records:
                logger.warning("Record limit reached (%d). Remaining lines skipped.", max_records)
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(Record.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError):
                skipped += 1
    if skipped:
        logger.warning("Skipped %d invalid lines in %s", skipped, path)
    return records
