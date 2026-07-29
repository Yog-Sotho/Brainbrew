"""
tests/test_sanitizer.py

Tests for pipeline/sanitizer.py.
Covers PII redaction, HTML stripping, quality checking, record sanitization,
dataset-level sanitization, and performance optimizations.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from pipeline.sanitizer import (
    SanitizerConfig,
    SanitizeStats,
    strip_html,
    redact_pii,
    clean_text,
    check_quality,
    get_record_hash,
    sanitize_record,
    sanitize_dataset,
    _sanitize_record_internal,
)


def test_strip_html():
    assert strip_html("<p>Hello <b>world</b>!</p>") == " Hello  world ! "
    assert strip_html("No HTML tags here.") == "No HTML tags here."


def test_redact_pii():
    # Email redaction
    text1, found1 = redact_pii("Contact me at test@example.com.")
    assert found1 is True
    assert "[PII_EMAIL]" in text1
    assert "test@example.com" not in text1

    # Email masking
    text1_masked, found1_masked = redact_pii("Contact me at test@example.com.", mask=True)
    assert found1_masked is True
    assert "t***t@example.com" in text1_masked

    # IP Address redaction
    text2, found2 = redact_pii("Connecting to 192.168.1.1.")
    assert found2 is True
    assert "[PII_IP]" in text2

    # IP Address masking
    text2_masked, found2_masked = redact_pii("Connecting to 192.168.1.1.", mask=True)
    assert found2_masked is True
    assert "192.168.***.***" in text2_masked

    # Phone redaction
    text3, found3 = redact_pii("Call 123-456-7890 now.")
    assert found3 is True
    assert "[PII_PHONE]" in text3

    # Phone masking
    text3_masked, found3_masked = redact_pii("Call 123-456-7890 now.", mask=True)
    assert found3_masked is True
    assert "***-***-7890" in text3_masked


def test_clean_text():
    # Normalize Unicode and remove control chars
    text = "Hello\x00world!\tThis is normalized: f\u0301."
    cleaned = clean_text(text)
    # NFC normalization of f\u0301 is f\u0301 (NFKC used in clean_text normalizes to 'f')
    assert "Hello world! This is normalized: f" in cleaned
    assert "\x00" not in cleaned

    # Strip HTML and collapse whitespace
    html_text = "  <div> \n  Hello \t <b>World</b>! \n </div> "
    assert clean_text(html_text, remove_html=True) == "Hello World !"


def test_check_quality():
    cfg = SanitizerConfig(
        min_chars=10,
        max_chars=100,
        min_words=3,
        min_unique_ratio=0.5,
        min_ascii_ratio=0.8,
    )

    # Valid
    assert check_quality("This is a high quality response.", cfg) is None

    # Too short
    assert "too short" in check_quality("Short", cfg)

    # Too long
    assert "too long" in check_quality("This string is incredibly long and will definitely exceed the max character threshold we set." * 5, cfg)

    # Too few words (must be at least 10 characters to bypass the length check first)
    assert "too few words" in check_quality("Word Word.", cfg)

    # Low unique ratio
    assert "low unique-word ratio" in check_quality("word word word word word word word", cfg)

    # Low ASCII ratio
    assert "low ASCII ratio" in check_quality("Искусственный интеллект для всех людей.", cfg)


def test_get_record_hash():
    rec1 = {"instruction": "Hello ", "output": " World!"}
    rec2 = {"instruction": "hello", "output": "world!"}
    assert get_record_hash(rec1) == get_record_hash(rec2)


def test_sanitize_record():
    cfg = SanitizerConfig(remove_pii=True, min_chars=10)
    record = {
        "instruction": "What is my email?",
        "output": "Your email is test@example.com.",
    }

    # Public API works
    sanitized, rejection = sanitize_record(record, cfg)
    assert rejection is None
    assert sanitized is not None
    assert sanitized["output"] == "Your email is [PII_EMAIL]."

    # Internal API returns pii_found=True
    sanitized_int, rejection_int, pii_found = _sanitize_record_internal(record, cfg)
    assert rejection_int is None
    assert sanitized_int is not None
    assert pii_found is True


def test_sanitize_dataset(tmp_path: Path):
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"

    records = [
        {"instruction": "What is AI?", "output": "Artificial Intelligence is simulation of human intelligence."},
        {"instruction": "What is your email?", "output": "Please contact test@example.com for help."},
        {"instruction": "What is AI?", "output": "Artificial Intelligence is simulation of human intelligence."}, # Exact duplicate
        {"instruction": "Too short", "output": "Short"}, # Quality filter
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cfg = SanitizerConfig(remove_pii=True, min_chars=10, deduplicate=True)
    stats = sanitize_dataset(input_file, output_file, cfg)

    assert stats.total == 4
    assert stats.kept == 2
    assert stats.deduplicated == 1
    assert stats.filtered_quality == 1
    assert stats.pii_redacted == 1

    # Verify output content
    with open(output_file, "r", encoding="utf-8") as f:
        output_records = [json.loads(line) for line in f]

    assert len(output_records) == 2
    assert output_records[0]["instruction"] == "What is AI?"
    assert output_records[1]["output"] == "Please contact [PII_EMAIL] for help."
