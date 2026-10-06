"""
tests/test_runs_and_quality.py

Run directories (pipeline/runs.py), quality scoring (pipeline/quality.py) and
document reading (pipeline/document_loader.read_document).
"""
from __future__ import annotations

import json
import re

import pytest

from pipeline.document_loader import read_document
from pipeline.quality import score_records
from pipeline.records import Record
from pipeline.runs import RunDir, create_run, new_run_id, open_run, runs_base


class TestRuns:

    def test_runs_base_follows_env(self, isolated_runs_dir):
        assert runs_base() == isolated_runs_dir

    def test_runs_base_default(self, monkeypatch):
        monkeypatch.delenv("BRAINBREW_RUNS_DIR")
        assert str(runs_base()) == "runs"

    def test_new_run_ids_are_unique_and_sortable(self):
        ids = {new_run_id() for _ in range(50)}
        assert len(ids) == 50
        assert all(re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", i) for i in ids)

    def test_create_and_open(self):
        run = create_run()
        assert run.root.is_dir()
        assert run.read_manifest()["status"] == "created"
        assert open_run(run.run_id).root == run.root

    def test_open_missing_run(self):
        with pytest.raises(FileNotFoundError):
            open_run("20260101-000000-abcdef")

    def test_paths(self, tmp_path):
        run = RunDir(tmp_path / "20260101-000000-abcdef")
        assert run.dataset("sharegpt").name == "dataset.sharegpt.jsonl"
        assert run.adapter_zip.name == "adapter.zip"
        assert run.records.name == "records.jsonl"

    def test_manifest_updates_merge_and_stay_valid_json(self):
        run = create_run()
        run.update_manifest(status="running", counts={"chunks": 3})
        run.update_manifest(counts={"chunks": 3, "generated": 2})
        manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "running"
        assert manifest["counts"] == {"chunks": 3, "generated": 2}
        assert not list(run.root.glob("*.tmp"))

    def test_corrupt_manifest_reads_as_empty(self):
        run = create_run()
        run.manifest_path.write_text("{oops", encoding="utf-8")
        assert run.read_manifest() == {}


class TestQuality:

    def test_empty_is_disaster(self):
        assert score_records([])["grade"] == "DISASTER"

    def test_good_dataset(self):
        recs = [Record(instruction=f"Unique question {i}?", output="word " * 80) for i in range(200)]
        report = score_records(recs)
        assert report["grade"] == "SUPER"
        assert report["record_count"] == 200
        assert report["unique_ratio"] == 1.0

    def test_short_duplicate_outputs_warned(self):
        recs = [Record(instruction="Same?", output="Short.") for _ in range(10)]
        report = score_records(recs)
        assert report["grade"] == "BAD"
        assert "very short" in report["details"] and "duplicate" in report["details"]

    def test_input_does_not_change_score(self):
        a = [Record(instruction=f"Q{i}", output="x" * 250) for i in range(60)]
        b = [Record(instruction=f"Q{i}", input="context", output="x" * 250) for i in range(60)]
        assert score_records(a) == score_records(b)


class TestReadDocument:

    def test_utf8_text(self):
        assert read_document("a.txt", "Héllo 🤖".encode()) == "Héllo 🤖"

    def test_bom_stripped(self):
        assert read_document("a.txt", b"\xef\xbb\xbfhello") == "hello"

    def test_invalid_utf8_is_replaced_not_rejected(self):
        assert read_document("a.txt", b"caf\xe9") == "caf�"

    def test_pdf_detected_by_magic_bytes(self):
        from pdfminer.pdfparser import PDFSyntaxError

        with pytest.raises((PDFSyntaxError, ValueError)):
            read_document("no-extension", b"%PDF-1.7 broken")

    def test_real_pdf(self):
        pdf = _tiny_pdf("Brainbrew reads PDFs")
        assert "Brainbrew reads PDFs" in read_document("doc.PDF", pdf)


def _tiny_pdf(text: str) -> bytes:
    """A minimal single-page PDF with one line of text."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
