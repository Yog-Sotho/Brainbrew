from pdfminer.high_level import extract_text

def load_document(path):

    if path.endswith(".pdf"):
        return extract_text(path)

    with open(path) as f:
        return f.read()


def chunk_text(text, chunk_size=800):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))

    return chunks
