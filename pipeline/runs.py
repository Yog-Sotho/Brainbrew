"""
Persistent run directories.

Every generation run gets its own folder, so results survive Streamlit reruns,
page reloads and server restarts:

    runs/<run_id>/
        source.txt          concatenated document text
        raw.jsonl           canonical records straight from the teacher model
        records.jsonl       canonical records after dedup + sanitizing
        dataset.<fmt>.jsonl the export in the chosen training format
        manifest.json       config (no secrets), status, counts, quality report
        adapter/            LoRA adapter (when training was requested)
        adapter.zip         the adapter, zipped for download

The base folder is `./runs`, or $BRAINBREW_RUNS_DIR when set.
"""
from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNS_DIR_ENV = "BRAINBREW_RUNS_DIR"
_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def runs_base() -> Path:
    return Path(os.getenv(RUNS_DIR_ENV) or "runs")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunDir:
    """Paths inside one run directory."""

    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def source(self) -> Path:
        return self.root / "source.txt"

    @property
    def raw(self) -> Path:
        return self.root / "raw.jsonl"

    @property
    def records(self) -> Path:
        return self.root / "records.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def adapter_dir(self) -> Path:
        return self.root / "adapter"

    @property
    def adapter_zip(self) -> Path:
        return self.root / "adapter.zip"

    @property
    def distilabel_cache(self) -> Path:
        return self.root / ".distilabel"

    def dataset(self, output_format: str) -> Path:
        return self.root / f"dataset.{output_format}.jsonl"

    # ── manifest ─────────────────────────────────────────────────────────
    def read_manifest(self) -> dict[str, Any]:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def update_manifest(self, **fields: Any) -> dict[str, Any]:
        """Merge *fields* into manifest.json (atomic replace) and return it."""
        manifest = self.read_manifest()
        manifest.update(fields)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.manifest_path)
        return manifest


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def create_run(base: Path | None = None) -> RunDir:
    """Create a fresh, empty run directory."""
    base = base or runs_base()
    root = base / new_run_id()
    root.mkdir(parents=True, exist_ok=False)
    run = RunDir(root)
    run.update_manifest(run_id=run.run_id, status="created", created_at=utc_now())
    return run


def open_run(run_id: str, base: Path | None = None) -> RunDir:
    """Open an existing run by id. Rejects anything that is not a generated id."""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"Invalid run id: {run_id!r}")
    root = (base or runs_base()) / run_id
    if not root.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return RunDir(root)
