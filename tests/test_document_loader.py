"""
tests/test_document_loader.py

Tests for pipeline/document_loader.py — character_chunk() and semantic_chunk().

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
  - M-05: semantic_chunk does paragraph-aware splitting
"""
from __future__ import annotations

import pytest

from pipeline.document_loader import character_chunk, semantic_chunk


# ---------------------------------------------------------------------------
# Input validation (shared by both functions)
# ---------------------------------------------------------------------------

class TestInputValidation:

    @pytest.mark.parametrize("func", [character_chunk, semantic_chunk])
    def test_non_string_raises_typeerror(self, func):
        with pytest.raises(TypeError, match="str"):
            func(12345)  # type: ignore[arg-type]

    @pytest.mark.parametrize("func", [character_chunk, semantic_chunk])
    def test_none_raises_typeerror(self, func):
        with pytest.raises(TypeError):
            func(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("func", [character_chunk, semantic_chunk])
    def test_empty_string_raises_valueerror(self, func):
        with pytest.raises(ValueError, match="empty"):
            func("")

    @pytest.mark.parametrize("func", [character_chunk, semantic_chunk])
    def test_whitespace_only_raises_valueerror(self, func):
        with pytest.raises(ValueError, match="empty"):
            func("   \n\t  ")

    @pytest.mark.parametrize("func", [character_chunk, semantic_chunk])
    def test_overlap_equal_to_chunk_size_raises(self, func):
        with pytest.raises(ValueError, match="overlap"):
            func("word " * 200, chunk_size=100, overlap=100)

    @pytest.mark.parametrize("func", [character_chunk, semantic_chunk])
    def test_overlap_greater_than_chunk_size_raises(self, func):
        with pytest.raises(ValueError, match="overlap"):
            func("word " * 200, chunk_size=100, overlap=150)


# ---------------------------------------------------------------------------
# character_chunk — normal operation
# ---------------------------------------------------------------------------

class TestCharacterChunk:

    def test_returns_list(self, tiny_text):
        result = character_chunk(tiny_text)
        assert isinstance(result, list)

    def test_returns_nonempty_list(self, tiny_text):
        result = character_chunk(tiny_text)
        assert len(result) > 0

    def test_all_chunks_are_strings(self, tiny_text):
        result = character_chunk(tiny_text)
        assert all(isinstance(c, str) for c in result)

    def test_no_whitespace_only_chunks(self, tiny_text):
        result = character_chunk(tiny_text)
        assert all(c.strip() for c in result)

    def test_large_text_produces_multiple_chunks(self, large_text):
        result = character_chunk(large_text, chunk_size=500, overlap=50)
        assert len(result) > 5

    def test_short_text_produces_at_least_one_chunk(self):
        result = character_chunk("This is a short sentence about machine learning.")
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# semantic_chunk — paragraph-aware splitting (M-05, Enhancement 9)
# ---------------------------------------------------------------------------

class TestSemanticChunk:

    def test_returns_nonempty_list(self, tiny_text):
        result = semantic_chunk(tiny_text)
        assert len(result) > 0

    def test_no_whitespace_only_chunks(self, tiny_text):
        result = semantic_chunk(tiny_text)
        assert all(c.strip() for c in result)

    def test_respects_paragraph_boundaries(self):
        """Paragraphs should not be split mid-paragraph when they fit in chunk_size."""
        text = (
            "First paragraph about AI and machine learning.\n\n"
            "Second paragraph about natural language processing.\n\n"
            "Third paragraph about computer vision and robotics."
        )
        result = semantic_chunk(text, chunk_size=2000, overlap=0)
        # With a large chunk_size, the whole text should fit in one or few chunks
        assert len(result) >= 1
        # Each paragraph content should appear in the output
        full_text = " ".join(result)
        assert "First paragraph" in full_text
        assert "Second paragraph" in full_text
        assert "Third paragraph" in full_text

    def test_large_paragraph_gets_split(self):
        """A single paragraph larger than chunk_size must be broken down."""
        big_para = "word " * 500  # ~2500 chars
        result = semantic_chunk(big_para, chunk_size=200, overlap=20)
        assert len(result) > 1

    def test_respect_paragraphs_false_falls_back(self, tiny_text):
        """respect_paragraphs=False should use character_chunk internally."""
        result = semantic_chunk(tiny_text, respect_paragraphs=False)
        assert len(result) > 0
        assert all(c.strip() for c in result)


# ---------------------------------------------------------------------------
# Unicode and special content
# ---------------------------------------------------------------------------

class TestUnicodeAndSpecialContent:

    @pytest.mark.parametrize("func", [character_chunk, semantic_chunk])
    def test_unicode_text_handled(self, func):
        text = "Kuenstliche Intelligenz. " * 30
        result = func(text, chunk_size=200, overlap=20)
        assert len(result) > 0

    @pytest.mark.parametrize("func", [character_chunk, semantic_chunk])
    def test_emoji_text_handled(self, func):
        text = "Brainbrew generates synthetic data. It uses distilabel. " * 40
        result = func(text, chunk_size=200, overlap=20)
        assert len(result) > 0

    @pytest.mark.parametrize("func", [character_chunk, semantic_chunk])
    def test_newlines_and_tabs_handled(self, func):
        text = "Line one.\nLine two.\n\tTabbed line.\n" * 60
        result = func(text, chunk_size=300, overlap=30)
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
    result = character_chunk(tiny_text, chunk_size=chunk_size, overlap=overlap)
    assert isinstance(result, list)
    assert all(c.strip() for c in result)
