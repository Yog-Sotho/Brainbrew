"""
tests/test_docs.py

The Markdown docs stay in step with the code: every config field is in the
reference, every page is in the site navigation, and docs/ holds no binary
documents that diffs cannot review.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from config import DistillationConfig

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def _nav_pages(nav: list) -> set[str]:
    pages: set[str] = set()
    for item in nav:
        for value in item.values():
            pages |= _nav_pages(value) if isinstance(value, list) else {value}
    return pages


def test_every_config_field_is_documented():
    reference = (DOCS / "configuration.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| `(\w+)` \|", reference, flags=re.MULTILINE))
    assert documented == set(DistillationConfig.model_fields)


def test_every_page_is_in_the_navigation():
    nav = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))["nav"]
    pages = {p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md")}
    pages = {p for p in pages if not any(part.startswith(".") for part in p.split("/"))}  # MkDocs skips these
    assert _nav_pages(nav) == pages


def test_no_binary_documents():
    assert not [p for p in DOCS.rglob("*") if p.suffix.lower() in {".pdf", ".docx", ".pptx"}]
