"""
Document chunking module.
Provides semantic text chunking using langchain's RecursiveCharacterTextSplitter.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional


class DocumentLoaderError(Exception):
    """Custom exception for document loading failures."""
    pass


def validate_chunking_params(chunk_size: int, overlap: int) -> tuple[int, int]:
    """
    Validate and normalize chunking parameters.

    Args:
        chunk_size: Target size for each chunk
        overlap: Overlap between chunks

    Returns:
        Tuple of (validated_chunk_size, validated_overlap)

    Raises:
        DocumentLoaderError: If parameters are invalid
    """
    if chunk_size <= 0:
        raise DocumentLoaderError(f"chunk_size must be positive, got {chunk_size}")

    if overlap < 0:
        raise DocumentLoaderError(f"overlap must be non-negative, got {overlap}")

    if overlap >= chunk_size:
        raise DocumentLoaderError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    return chunk_size, overlap


def semantic_chunk(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    separators: Optional[list[str]] = None
) -> list[str]:
    """
    Split text into semantically meaningful chunks.

    Uses RecursiveCharacterTextSplitter to split text while trying to
    preserve semantic boundaries (paragraphs, sentences, words).

    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk (default: 800 characters)
        overlap: Number of characters to overlap between chunks (default: 100)
        separators: Custom separator list (optional)

    Returns:
        List of text chunks

    Raises:
        DocumentLoaderError: If text is empty or chunking fails
    """
    if not text:
        raise DocumentLoaderError("Input text cannot be empty")

    if not isinstance(text, str):
        raise DocumentLoaderError(f"Expected string input, got {type(text).__name__}")

    # Validate parameters
    chunk_size, overlap = validate_chunking_params(chunk_size, overlap)

    # Default separators for code and prose
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators,
            length_function=len,
            add_start_index=True,
        )
        chunks = splitter.split_text(text)

        if not chunks:
            raise DocumentLoaderError("No chunks generated from input text")

        return chunks

    except DocumentLoaderError:
        raise
    except Exception as e:
        raise DocumentLoaderError(f"Failed to chunk text: {e}")
