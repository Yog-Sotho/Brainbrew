from langchain_text_splitters import RecursiveCharacterTextSplitter

def semantic_chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
