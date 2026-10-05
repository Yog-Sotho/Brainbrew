"""
tests/conftest.py

Shared fixtures and test helpers for the Brainbrew test suite.

The suite runs against the real core libraries (openai, datasets, streamlit,
...): pipeline tests drive the real async engine and openai SDK against the
in-memory model server in tests/fake_openai.py, so mocks cannot drift from the
real APIs. A few core libraries get stubs only when they are not installed.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Lightweight stubs for heavy GPU / ML libraries that may not be installed.
# These are injected into sys.modules BEFORE any project code is imported.
# ---------------------------------------------------------------------------

def _make_stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _missing(name: str) -> bool:
    """True when *name* is neither imported nor installed (so a stub is needed)."""
    return name not in sys.modules and importlib.util.find_spec(name) is None


def _install_heavy_stubs() -> None:
    """Inject minimal stubs so project imports don't fail on import-time."""

    # ── huggingface_hub ─────────────────────────────────────────────────────
    if _missing("huggingface_hub"):
        hfh = _make_stub("huggingface_hub")
        hfh.HfApi = MagicMock(name="HfApi")
        sys.modules["huggingface_hub"] = hfh

    # ── datasets ────────────────────────────────────────────────────────────
    if _missing("datasets"):
        ds = _make_stub("datasets")
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=MagicMock())
        ds.load_dataset = MagicMock(return_value=mock_ds)
        sys.modules["datasets"] = ds

    # ── structlog ───────────────────────────────────────────────────────────
    if _missing("structlog"):
        sl = _make_stub("structlog")
        sl.get_logger = MagicMock(return_value=MagicMock())
        sl.configure = MagicMock()
        sl.make_filtering_bound_logger = MagicMock(return_value=MagicMock())
        sys.modules["structlog"] = sl

    # ── langchain_text_splitters ─────────────────────────────────────────────
    # Only stub if not already installed (it IS in requirements.txt)
    if _missing("langchain_text_splitters"):
        lc = _make_stub("langchain_text_splitters")
        lc.RecursiveCharacterTextSplitter = MagicMock(name="RecursiveCharacterTextSplitter")
        sys.modules["langchain_text_splitters"] = lc

    # ── streamlit ────────────────────────────────────────────────────────────
    if _missing("streamlit"):
        st = _make_stub("streamlit")
        for attr in ["set_page_config", "title", "caption", "header", "checkbox",
                     "text_input", "selectbox", "slider", "file_uploader", "button",
                     "error", "warning", "success", "stop", "progress", "download_button",
                     "balloons", "sidebar", "expander", "divider", "columns",
                     "metric", "markdown", "info", "empty"]:
            setattr(st, attr, MagicMock())
        sys.modules["streamlit"] = st

    # ── dotenv ──────────────────────────────────────────────────────────────
    if _missing("dotenv"):
        dotenv = _make_stub("dotenv")
        dotenv.load_dotenv = MagicMock()
        sys.modules["dotenv"] = dotenv


_install_heavy_stubs()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test writes run directories under its own tmp_path, never ./runs."""
    runs = tmp_path / "runs"
    monkeypatch.setenv("BRAINBREW_RUNS_DIR", str(runs))
    return runs


@pytest.fixture()
def tiny_text() -> str:
    """A minimal, realistic document text for chunking tests."""
    return (
        "Artificial intelligence is the simulation of human intelligence by machines. "
        "Machine learning is a subset of AI that enables systems to learn from data. "
        "Deep learning uses neural networks with many layers to model complex patterns. "
        "Natural language processing allows computers to understand and generate human language. "
        "Reinforcement learning trains agents to make decisions by rewarding desired behaviours. "
    ) * 10  # ~500 chars * 10 = ~5000 chars


@pytest.fixture()
def large_text() -> str:
    """A large document text (>50 KB) to stress-test the chunker."""
    paragraph = (
        "The transformer architecture revolutionised natural language processing. "
        "Attention mechanisms allow models to weigh the relevance of each input token. "
        "Pre-training on large corpora followed by fine-tuning yields strong results. "
    )
    return paragraph * 300  # ~220 chars * 300 = ~66 KB


@pytest.fixture()
def source_file(tmp_path: Path, tiny_text: str) -> Path:
    """A real source text file on disk for orchestrator tests."""
    p = tmp_path / "source.txt"
    p.write_text(tiny_text, encoding="utf-8")
    return p
