"""
Rebuild the benchmark corpus in tests/fixtures/bench/ (stdlib only).

Three public-domain texts of different kinds, downloaded from Project
Gutenberg, cut to one section each and rendered as plain multi-page PDFs:

* federalist_10.pdf      - James Madison, The Federalist No. 10 (1787): argument
* elements_of_style.pdf  - William Strunk Jr., The Elements of Style (1918),
                           ch. II "Elementary Rules of Usage": procedural rules
* origin_of_species.pdf  - Charles Darwin, On the Origin of Species (1859),
                           ch. III "Struggle for Existence": scientific exposition

    python bench/make_fixtures.py

The PDFs are committed, so this only needs to run to change the corpus.
"""
from __future__ import annotations

import re
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "bench"


@dataclass(frozen=True)
class Source:
    filename: str
    title: str
    gutenberg_id: int
    start: str  # regex for the first line of the section
    end: str    # regex for the first line after it


SOURCES = (
    Source("federalist_10.pdf", "The Federalist No. 10 - James Madison (1787)", 1404,
           r"^FEDERALIST No\. 10$", r"^FEDERALIST No\. 11$"),
    Source("elements_of_style.pdf", "The Elements of Style, ch. II - William Strunk Jr. (1918)", 37134,
           r"^II\. ELEMENTARY RULES OF USAGE$", r"^III\. ELEMENTARY PRINCIPLES OF COMPOSITION$"),
    Source("origin_of_species.pdf", "On the Origin of Species, ch. III - Charles Darwin (1859)", 1228,
           r"^CHAPTER III\.$", r"^CHAPTER IV\.$"),
)


def fetch(gutenberg_id: int, attempts: int = 4) -> str:
    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https URL
                data: bytes = resp.read()
                return data.decode("utf-8-sig").replace("\r\n", "\n")
        except OSError:
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def section(text: str, start: str, end: str) -> str:
    lines = text.split("\n")
    i = next(n for n, line in enumerate(lines) if re.match(start, line.strip()))
    j = next(n for n, line in enumerate(lines[i + 1:], i + 1) if re.match(end, line.strip()))
    return "\n".join(lines[i:j]).strip()


def paragraphs(text: str) -> list[str]:
    """Unwrap Gutenberg's hard-wrapped lines into paragraphs."""
    paras = re.split(r"\n\s*\n", text)
    return [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]


# ── a minimal PDF writer: Helvetica 10/12pt, US Letter, WinAnsi text ─────────
PAGE_W, PAGE_H, MARGIN, LEADING, WRAP = 612, 792, 72, 12, 92
LINES_PER_PAGE = (PAGE_H - 2 * MARGIN) // LEADING


def _pdf_string(line: str) -> bytes:
    raw = line.encode("cp1252", errors="replace")
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def render_pdf(title: str, paras: list[str]) -> bytes:
    lines: list[str] = [title, ""]
    for p in paras:
        lines += textwrap.wrap(p, WRAP) or [""]
        lines.append("")
    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]

    objects: list[bytes] = []  # object n is objects[n - 1]

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    catalog = add(b"")  # filled in below
    pages_obj = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    kids = []
    for page in pages:
        ops = [b"BT", b"/F1 10 Tf", f"{LEADING} TL".encode(), f"{MARGIN} {PAGE_H - MARGIN} Td".encode()]
        ops += [_pdf_string(line) + b" '" for line in page]
        ops.append(b"ET")
        stream = b"\n".join(ops)
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        kids.append(add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %d %d] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (pages_obj, PAGE_W, PAGE_H, font, content)
        ))
    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_obj
    objects[pages_obj - 1] = (
        b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % k for k in kids) + b"] /Count %d >>" % len(kids)
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog, xref)
    return bytes(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        text = section(fetch(src.gutenberg_id), src.start, src.end)
        pdf = render_pdf(src.title, paragraphs(text))
        (OUT_DIR / src.filename).write_bytes(pdf)
        print(f"{src.filename}: {len(text):,} chars, {len(pdf):,} bytes")


if __name__ == "__main__":
    main()
