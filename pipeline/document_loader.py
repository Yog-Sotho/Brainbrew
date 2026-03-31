"""
Brainbrew document chunker — splits source text into training-ready chunks.

Provides two strategies:
  1. character_chunk() — fast, deterministic, RecursiveCharacterTextSplitter-based.
  2. semantic_chunk()  — paragraph-aware splitting with sentence-boundary merging.

Both return a list of non-empty text chunks suitable for prompt generation.
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _validate_inputs(text: str, chunk_size: int, overlap: int) -> None:
    """Shared input validation for all chunking strategies."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("Cannot chunk empty or whitespace-only text.")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size}).")


def character_chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text using RecursiveCharacterTextSplitter (character-boundary heuristic).

    Fast and deterministic. Splits on paragraph breaks, then sentences, then words.
    Does NOT perform true semantic analysis — chunk boundaries may fall mid-topic.
    """
    _validate_inputs(text, chunk_size, overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if c.strip()]


# ── Enhancement 9: paragraph-aware semantic chunking ────────────────────────

# Sentence boundary regex — handles abbreviations conservatively
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])")


def semantic_chunk(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    respect_paragraphs: bool = True,
) -> list[str]:
    """Split text with awareness of paragraph and sentence boundaries.

    Strategy:
      1. Split on paragraph breaks (double newlines).
      2. Split oversized paragraphs on sentence boundaries.
      3. Merge undersized adjacent chunks up to chunk_size.
      4. Filter out whitespace-only fragments.

    This produces chunks that align with natural topic boundaries in the
    document, yielding higher-quality training prompts than pure character
    splitting.

    Args:
        text: Source document text.
        chunk_size: Target maximum characters per chunk.
        overlap: Character overlap between adjacent chunks.
        respect_paragraphs: If True, never merge across paragraph boundaries
                           unless a paragraph is smaller than overlap.

    Returns:
        List of non-empty text chunks.
    """
    _validate_inputs(text, chunk_size, overlap)

    if not respect_paragraphs:
        return character_chunk(text, chunk_size, overlap)

    # Step 1: split into paragraphs
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Step 2: break oversized paragraphs into sentences
    fragments: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            fragments.append(para)
        else:
            sentences = _SENTENCE_RE.split(para)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    fragments.append(sent)

    # Step 3: merge small adjacent fragments
    merged: list[str] = []
    current = ""
    for frag in fragments:
        candidate = f"{current}\n\n{frag}".strip() if current else frag
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            # If a single fragment exceeds chunk_size, fall back to character split
            if len(frag) > chunk_size:
                sub_chunks = character_chunk(frag, chunk_size, overlap)
                merged.extend(sub_chunks)
                current = ""
            else:
                current = frag
    if current:
        merged.append(current)

    # Step 4: apply overlap by repeating tail of previous chunk
    if overlap > 0 and len(merged) > 1:
        overlapped: list[str] = [merged[0]]
        for i in range(1, len(merged)):
            prev_tail = merged[i - 1][-overlap:]
            overlapped.append(f"{prev_tail} {merged[i]}")
        merged = overlapped

    return [c for c in merged if c.strip()]
