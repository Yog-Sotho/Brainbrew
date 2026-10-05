"""
Dataset quality report (Enhancement 10).

Scores canonical records, so a dataset gets the same grade whichever export
format the user picked.
"""
from __future__ import annotations

from typing import TypedDict

from pipeline.records import Record

GRADES: tuple[str, ...] = ("SUPER", "GOOD", "NORMAL", "BAD", "DISASTER")

_QUALITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "SUPER":  {"min_records": 100, "min_avg_len": 300, "min_unique_ratio": 0.95},
    "GOOD":   {"min_records": 50,  "min_avg_len": 200, "min_unique_ratio": 0.85},
    "NORMAL": {"min_records": 20,  "min_avg_len": 100, "min_unique_ratio": 0.70},
    "BAD":    {"min_records": 5,   "min_avg_len": 50,  "min_unique_ratio": 0.50},
}


class QualityReport(TypedDict):
    grade: str
    record_count: int
    avg_output_length: float
    unique_ratio: float
    details: str


def score_records(records: list[Record]) -> QualityReport:
    """Grade a dataset from its record count, answer length and instruction uniqueness."""
    if not records:
        return {
            "grade": "DISASTER",
            "record_count": 0,
            "avg_output_length": 0.0,
            "unique_ratio": 0.0,
            "details": "Dataset is empty — no valid records produced.",
        }

    record_count = len(records)
    avg_output_len = sum(len(r.output) for r in records) / record_count
    unique_ratio = len({r.instruction for r in records}) / record_count

    grade = "BAD"
    for level in ("SUPER", "GOOD", "NORMAL", "BAD"):
        t = _QUALITY_THRESHOLDS[level]
        if (record_count >= t["min_records"]
                and avg_output_len >= t["min_avg_len"]
                and unique_ratio >= t["min_unique_ratio"]):
            grade = level
            break

    detail_parts = [
        f"{record_count} records generated",
        f"Average output length: {avg_output_len:.0f} chars",
        f"Instruction uniqueness: {unique_ratio:.0%}",
    ]
    if avg_output_len < 100:
        detail_parts.append("⚠ Outputs are very short — consider using Research mode.")
    if unique_ratio < 0.70:
        detail_parts.append("⚠ Many duplicate instructions — increase dataset_size or source material.")

    return {
        "grade": grade,
        "record_count": record_count,
        "avg_output_length": avg_output_len,
        "unique_ratio": unique_ratio,
        "details": " · ".join(detail_parts),
    }
