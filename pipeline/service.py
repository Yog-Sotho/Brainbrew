"""
Services shared by the web app and the CLI: turning documents into a run.
"""
from __future__ import annotations

from collections.abc import Iterable

from pipeline.document_loader import read_document
from pipeline.runs import RunDir, create_run


def read_documents(files: Iterable[tuple[str, bytes]]) -> tuple[str, list[str]]:
    """Extracted text of all readable documents, plus one message per unreadable one."""
    parts, errors = [], []
    for name, data in files:
        try:
            parts.append(read_document(name, data))
        except Exception as e:  # pdfminer raises many types on bad files
            errors.append(f"Could not parse '{name}': {e} — skipping.")
    return "\n\n".join(parts), errors


def new_run(source_text: str, owner: str | None = None) -> RunDir:
    """A fresh run folder holding *source_text*, ready to hand to a runner."""
    if not source_text.strip():
        raise ValueError("No text could be extracted from the documents.")
    run = create_run()
    run.source.write_text(source_text, encoding="utf-8")
    if owner is not None:
        run.update_manifest(owner=owner)
    return run
