"""
tests/test_properties.py

Property-based tests (Hypothesis) for the parts that handle arbitrary text:
chunkers, formatters and record I/O, dedup and the sanitizer. Each test states
an invariant that must hold for every input, and Hypothesis searches for a
counterexample.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pipeline.dedup import deduplicate
from pipeline.document_loader import character_chunk, semantic_chunk
from pipeline.exporter import FORMATTERS, export_dataset
from pipeline.pii import redact_pii
from pipeline.records import Record, read_records, write_records
from pipeline.sanitizer import clean_text

PROFILE = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])

# Words of letters/digits, joined by spaces, sentence ends and paragraph breaks.
words = st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), min_size=1, max_size=12)
separators = st.sampled_from([" ", " ", " ", ". ", "\n", "\n\n", ", "])
documents = st.lists(st.tuples(words, separators), min_size=1, max_size=300).map(
    lambda pairs: "".join(w + s for w, s in pairs)
)
sizes = st.integers(min_value=40, max_value=400).flatmap(
    lambda size: st.tuples(st.just(size), st.integers(min_value=0, max_value=size // 2))
)


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text)


# ── chunkers ─────────────────────────────────────────────────────────────────

@PROFILE
@given(documents, sizes)
def test_character_chunks_are_bounded_and_cover_every_word(text, size_overlap):
    size, overlap = size_overlap
    chunks = character_chunk(text, size, overlap)
    assert chunks and all(c.strip() for c in chunks)
    assert all(len(c) <= size for c in chunks)
    covered = set().union(*(_words(c) for c in chunks))
    assert set(_words(text)) <= covered


@PROFILE
@given(documents, sizes)
def test_semantic_chunks_are_bounded_and_cover_every_word(text, size_overlap):
    size, overlap = size_overlap
    chunks = semantic_chunk(text, size, overlap)
    assert chunks and all(c.strip() for c in chunks)
    # A chunk is at most chunk_size, plus the overlap tail and one space it is prefixed with.
    assert all(len(c) <= size + overlap + 1 for c in chunks)
    covered = set().union(*(_words(c) for c in chunks))
    assert set(_words(text)) <= covered


@PROFILE
@given(documents, sizes, st.booleans())
def test_chunking_is_deterministic(text, size_overlap, semantic):
    size, overlap = size_overlap
    chunker = semantic_chunk if semantic else character_chunk
    assert chunker(text, size, overlap) == chunker(text, size, overlap)


# ── records and formatters ───────────────────────────────────────────────────

any_text = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())
records = st.builds(
    Record,
    instruction=any_text,
    input=st.one_of(st.just(""), st.text(max_size=100)),
    output=any_text,
    meta=st.dictionaries(st.text(min_size=1, max_size=10), st.integers() | st.text(max_size=20), max_size=3),
)


@PROFILE
@given(records)
def test_every_format_carries_the_record_exactly(rec):
    alpaca = FORMATTERS["alpaca"](rec)
    assert alpaca == {"instruction": rec.instruction, "input": rec.input, "output": rec.output}
    sharegpt = FORMATTERS["sharegpt"](rec)["conversations"]
    assert [m["value"] for m in sharegpt] == [rec.prompt, rec.output]
    for fmt in ("chatml", "openai"):
        msgs = FORMATTERS[fmt](rec)["messages"]
        assert msgs[-2] == {"role": "user", "content": rec.prompt}
        assert msgs[-1] == {"role": "assistant", "content": rec.output}
    if rec.input.strip():
        assert rec.instruction in rec.prompt and rec.input in rec.prompt
    else:
        assert rec.prompt == rec.instruction


@PROFILE
@given(st.lists(records, max_size=15), st.sampled_from(sorted(FORMATTERS)))
def test_export_writes_one_parseable_line_per_record(recs, fmt):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.jsonl"
        assert export_dataset(recs, path, fmt) == len(recs)
        lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [FORMATTERS[fmt](r) for r in recs]


@PROFILE
@given(st.lists(records, max_size=15))
def test_canonical_records_round_trip(recs):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.jsonl"
        assert write_records(path, recs) == len(recs)
        assert read_records(path) == recs


# ── dedup ────────────────────────────────────────────────────────────────────

@PROFILE
@given(st.lists(records, max_size=25))
def test_dedup_keeps_order_and_is_idempotent(recs):
    once = deduplicate(recs)
    assert deduplicate(once) == once
    positions = [next(i for i, r in enumerate(recs) if r is kept) for kept in once]
    assert positions == sorted(positions)


@PROFILE
@given(st.lists(records, min_size=1, max_size=10))
def test_exact_duplicates_never_survive(recs):
    doubled = recs + [r.model_copy() for r in recs]
    assert len(deduplicate(doubled)) <= len(recs)


# ── sanitizer ────────────────────────────────────────────────────────────────

@PROFILE
@given(st.text(max_size=300))
def test_clean_text_is_idempotent(text):
    once = clean_text(text)
    assert clean_text(once) == once


@PROFILE
@given(st.text(max_size=300), st.sampled_from(["domain", "redact", "keep"]))
def test_redaction_is_idempotent(text, policy):
    once, _ = redact_pii(text, url_policy=policy)
    twice, found_again = redact_pii(once, url_policy=policy)
    assert twice == once and not found_again


emails = st.builds(
    "{}@{}.{}".format,
    st.from_regex(r"[a-z][a-z0-9._%+-]{0,10}[a-z0-9]", fullmatch=True),
    st.from_regex(r"[a-z][a-z0-9-]{0,10}[a-z0-9]", fullmatch=True),
    st.sampled_from(["com", "org", "io", "co.uk"]),
)


@PROFILE
@given(st.text(max_size=80), emails, st.text(max_size=80))
def test_emails_never_survive_redaction(before, email, after):
    # Keep the address a whole token: letters glued to it would make it a different address.
    assume(not before or not before[-1].isalnum())
    assume(not after or not after[0].isalnum())
    text = f"{before} {email} {after}"
    out, found = redact_pii(text)
    assert found and email not in out
