from langchain_text_splitters import RecursiveCharacterTextSplitter

def semantic_chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    # FIX: validate inputs before processing
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("Cannot chunk empty or whitespace-only text.")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size}).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    # Filter out any whitespace-only fragments
    return [c for c in chunks if c.strip()]
