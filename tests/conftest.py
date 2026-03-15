"""
tests/conftest.py

Shared fixtures, mock factories, and test helpers for the Brainbrew test suite.

All heavy dependencies (distilabel, unsloth, vllm, transformers, huggingface_hub)
are mocked at the module level so the suite runs on any machine — including those
without a GPU — making it safe for CI/CD and non-technical contributors.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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


def _install_heavy_stubs() -> None:
    """Inject minimal stubs so project imports don't fail on import-time."""

    # ── distilabel ──────────────────────────────────────────────────────────
    for mod_name in [
        "distilabel",
        "distilabel.pipeline",
        "distilabel.steps",
        "distilabel.steps.tasks",
        "distilabel.llms",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_stub(mod_name)

    # Pipeline
    MockPipeline = MagicMock(name="Pipeline")
    sys.modules["distilabel.pipeline"].Pipeline = MockPipeline

    # Steps
    for cls_name in ["LoadDataFromDicts", "KeepColumns", "FilterRows", "RenameColumns", "FilterStep"]:
        setattr(sys.modules["distilabel.steps"], cls_name, MagicMock(name=cls_name))

    # Tasks
    for cls_name in ["EvolInstruct", "TextGeneration"]:
        setattr(sys.modules["distilabel.steps.tasks"], cls_name, MagicMock(name=cls_name))

    # LLMs
    for cls_name in ["OpenAILLM", "vLLM"]:
        setattr(sys.modules["distilabel.llms"], cls_name, MagicMock(name=cls_name))

    # ── unsloth ─────────────────────────────────────────────────────────────
    if "unsloth" not in sys.modules:
        unsloth = _make_stub("unsloth")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        FastLM = MagicMock()
        FastLM.from_pretrained.return_value = (mock_model, mock_tokenizer)
        FastLM.get_peft_config.return_value = MagicMock()
        unsloth.FastLanguageModel = FastLM
        sys.modules["unsloth"] = unsloth

    # ── trl ─────────────────────────────────────────────────────────────────
    if "trl" not in sys.modules:
        trl = _make_stub("trl")
        trl.SFTTrainer = MagicMock(name="SFTTrainer")
        sys.modules["trl"] = trl

    # ── transformers ────────────────────────────────────────────────────────
    if "transformers" not in sys.modules:
        tf = _make_stub("transformers")
        tf.TrainingArguments = MagicMock(name="TrainingArguments")
        sys.modules["transformers"] = tf

    # ── huggingface_hub ─────────────────────────────────────────────────────
    if "huggingface_hub" not in sys.modules:
        hfh = _make_stub("huggingface_hub")
        hfh.HfApi = MagicMock(name="HfApi")
        sys.modules["huggingface_hub"] = hfh

    # ── datasets ────────────────────────────────────────────────────────────
    if "datasets" not in sys.modules:
        ds = _make_stub("datasets")
        mock_ds = MagicMock()
        mock_ds.__getitem__ = MagicMock(return_value=MagicMock())
        ds.load_dataset = MagicMock(return_value=mock_ds)
        sys.modules["datasets"] = ds

    # ── structlog ───────────────────────────────────────────────────────────
    if "structlog" not in sys.modules:
        sl = _make_stub("structlog")
        sl.get_logger = MagicMock(return_value=MagicMock())
        sl.configure = MagicMock()
        sl.make_filtering_bound_logger = MagicMock(return_value=MagicMock())
        sys.modules["structlog"] = sl

    # ── langchain_text_splitters ─────────────────────────────────────────────
    # Only stub if not already installed (it IS in requirements.txt)
    if "langchain_text_splitters" not in sys.modules:
        lc = _make_stub("langchain_text_splitters")
        lc.RecursiveCharacterTextSplitter = MagicMock(name="RecursiveCharacterTextSplitter")
        sys.modules["langchain_text_splitters"] = lc

    # ── streamlit ────────────────────────────────────────────────────────────
    if "streamlit" not in sys.modules:
        st = _make_stub("streamlit")
        for attr in ["set_page_config", "title", "caption", "header", "checkbox",
                     "text_input", "selectbox", "slider", "file_uploader", "button",
                     "error", "warning", "success", "stop", "progress", "download_button",
                     "balloons", "sidebar", "expander"]:
            setattr(st, attr, MagicMock())
        sys.modules["streamlit"] = st

    # ── dotenv ──────────────────────────────────────────────────────────────
    if "dotenv" not in sys.modules:
        dotenv = _make_stub("dotenv")
        dotenv.load_dotenv = MagicMock()
        sys.modules["dotenv"] = dotenv


_install_heavy_stubs()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
def raw_jsonl_file(tmp_path: Path) -> Path:
    """A valid raw distilabel JSONL file with mixed 'output' and 'generation' keys."""
    p = tmp_path / "raw.jsonl"
    records = [
        {"instruction": "What is AI?", "output": "AI stands for Artificial Intelligence, a field of computer science."},
        {"instruction": "Explain ML.", "generation": "Machine Learning is a method of data analysis that automates model building."},
        {"instruction": "Define NLP.", "output": "Natural Language Processing enables computers to understand human language."},
        {"instruction": "What is RL?", "output": "Reinforcement Learning trains an agent via rewards and punishments."},
        {"instruction": "Explain CNN.", "generation": "Convolutional Neural Networks are used primarily for image recognition tasks."},
    ]
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


@pytest.fixture()
def alpaca_output_file(tmp_path: Path) -> Path:
    """Empty destination path for Alpaca export tests."""
    return tmp_path / "alpaca.jsonl"


@pytest.fixture()
def base_config():
    """A minimal valid DistillationConfig with no GPU or API calls."""
    from config import DistillationConfig, QualityMode
    return DistillationConfig(
        teacher_model="gpt-4o",
        use_vllm=False,
        quality_mode=QualityMode.FAST,
        dataset_size=100,
        api_key=None,
    )


@pytest.fixture()
def source_file(tmp_path: Path, tiny_text: str) -> Path:
    """A real source text file on disk for orchestrator tests."""
    p = tmp_path / "source.txt"
    p.write_text(tiny_text, encoding="utf-8")
    return p
