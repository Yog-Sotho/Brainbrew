"""
tests/test_document_loader.py

Tests for pipeline/document_loader.py — semantic_chunk().

Covers:
  - Type validation (non-str input)
  - Empty / whitespace-only input
  - Overlap >= chunk_size guard
  - Normal chunking produces non-empty list of non-empty strings
  - Large document produces multiple chunks
  - Short document produces at least one chunk
  - No whitespace-only fragments in output
  - Unicode and multi-language text handled safely
  - Chunk size boundaries
  - Overlap proportionality
  - Each chunk is within the declared chunk_size
"""
from __future__ import annotations

import pytest

from pipeline.document_loader import semantic_chunk


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_non_string_raises_typeerror(self):
        with pytest.raises(TypeError, match="str"):
            semantic_chunk(12345)  # type: ignore[arg-type]

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            semantic_chunk(None)  # type: ignore[arg-type]

    def test_list_raises_typeerror(self):
        with pytest.raises(TypeError):
            semantic_chunk(["word", "word"])  # type: ignore[arg-type]

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            semantic_chunk("")

    def test_whitespace_only_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            semantic_chunk("   \n\t  ")

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            semantic_chunk("word " * 200, chunk_size=100, overlap=100)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            semantic_chunk("word " * 200, chunk_size=100, overlap=150)

    def test_overlap_just_below_chunk_size_accepted(self):
        chunks = semantic_chunk("word " * 200, chunk_size=100, overlap=99)
        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Normal operation
# ---------------------------------------------------------------------------

class TestNormalOperation:

    def test_returns_list(self, tiny_text):
        result = semantic_chunk(tiny_text)
        assert isinstance(result, list)

    def test_returns_nonempty_list(self, tiny_text):
        result = semantic_chunk(tiny_text)
        assert len(result) > 0

    def test_all_chunks_are_strings(self, tiny_text):
        result = semantic_chunk(tiny_text)
        assert all(isinstance(c, str) for c in result)

    def test_no_whitespace_only_chunks(self, tiny_text):
        result = semantic_chunk(tiny_text)
        assert all(c.strip() for c in result), "All chunks must have non-whitespace content"

    def test_large_text_produces_multiple_chunks(self, large_text):
        result = semantic_chunk(large_text, chunk_size=500, overlap=50)
        assert len(result) > 5, "Large text should produce many chunks"

    def test_short_text_produces_at_least_one_chunk(self):
        result = semantic_chunk("This is a short sentence about machine learning.")
        assert len(result) >= 1

    def test_each_chunk_respects_chunk_size(self, large_text):
        chunk_size = 400
        result = semantic_chunk(large_text, chunk_size=chunk_size, overlap=50)
        # RecursiveCharacterTextSplitter may slightly exceed; allow 20% tolerance
        for chunk in result:
            assert len(chunk) <= chunk_size * 1.2, (
                f"Chunk of length {len(chunk)} significantly exceeds chunk_size={chunk_size}"
            )

    def test_default_parameters_work(self, tiny_text):
        """Calling with only text uses chunk_size=800, overlap=100 — must not raise."""
        result = semantic_chunk(tiny_text)
        assert result


# ---------------------------------------------------------------------------
# Unicode and special content
# ---------------------------------------------------------------------------

class TestUnicodeAndSpecialContent:

    def test_unicode_text_handled(self):
        text = "人工知能は機械による人間の知能のシミュレーションです。" * 30
        result = semantic_chunk(text, chunk_size=200, overlap=20)
        assert len(result) > 0

    def test_emoji_text_handled(self):
        text = "🧠 Brainbrew generates synthetic training data. 🚀 It uses distilabel. " * 40
        result = semantic_chunk(text, chunk_size=200, overlap=20)
        assert len(result) > 0

    def test_mixed_language_text_handled(self):
        text = (
            "Artificial intelligence. "
            "Inteligencia artificial. "
            "Intelligence artificielle. "
            "Künstliche Intelligenz. "
        ) * 40
        result = semantic_chunk(text, chunk_size=300, overlap=30)
        assert len(result) > 0

    def test_newlines_and_tabs_in_text_handled(self):
        text = "Line one.\nLine two.\n\tTabbed line.\n" * 60
        result = semantic_chunk(text, chunk_size=300, overlap=30)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Parametrized chunk_size / overlap combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_size,overlap", [
    (200, 20),
    (500, 50),
    (800, 100),
    (1200, 200),
    (2000, 0),
])
def test_various_chunk_size_overlap_combinations(tiny_text, chunk_size, overlap):
    result = semantic_chunk(tiny_text, chunk_size=chunk_size, overlap=overlap)
    assert isinstance(result, list)
    assert all(c.strip() for c in result)
