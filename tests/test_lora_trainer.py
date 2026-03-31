"""
tests/test_lora_trainer.py

Tests for training/lora_trainer.py — train_lora() and _format_alpaca().

All GPU/ML dependencies (unsloth, trl, transformers, datasets) are mocked.
No GPU required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def dataset_file(tmp_path: Path) -> str:
    p = tmp_path / "alpaca.jsonl"
    records = [
        {"instruction": f"Q{i}?", "input": "", "output": f"Answer {i}."}
        for i in range(20)
    ]
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return str(p)


@pytest.fixture()
def output_dir(tmp_path: Path) -> str:
    d = tmp_path / "trained_adapter"
    d.mkdir()
    return str(d)


@pytest.fixture()
def mock_model():
    return MagicMock(name="model")


@pytest.fixture()
def mock_tokenizer():
    t = MagicMock(name="tokenizer")
    t.eos_token = "</s>"
    return t


def _run(dataset_file, output_dir, mock_model, mock_tokenizer,
         base_model="unsloth/mistral-7b-bnb-4bit", lora_rank=16):
    fast_lm = MagicMock()
    fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
    fast_lm.get_peft_config.return_value = MagicMock()

    mock_ds = MagicMock(name="Dataset")
    mock_mapped_ds = MagicMock(name="MappedDataset")
    mock_ds.map.return_value = mock_mapped_ds
    mock_trainer = MagicMock()
    mock_args = MagicMock()

    with patch("training.lora_trainer.FastLanguageModel", fast_lm), \
         patch("training.lora_trainer.SFTTrainer", return_value=mock_trainer) as MockSFT, \
         patch("training.lora_trainer.load_dataset", return_value=mock_ds) as MockLoad, \
         patch("training.lora_trainer.TrainingArguments", return_value=mock_args) as MockArgs:

        from training.lora_trainer import train_lora
        train_lora(dataset_file, base_model, output_dir, lora_rank=lora_rank)

    return fast_lm, MockSFT, MockArgs, mock_trainer, mock_ds, mock_mapped_ds, MockLoad


# ── Critical regression: train() called exactly once ─────────────────────────

class TestTrainCalledExactlyOnce:

    def test_trainer_train_called_once(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, _, _, mock_trainer, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        assert mock_trainer.train.call_count == 1

    def test_trainer_model_save_pretrained_not_called(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, _, _, mock_trainer, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        mock_trainer.model.save_pretrained.assert_not_called()


# ── FIX C-03: full instruction+output formatting ─────────────────────────────

class TestFullFormattingFix:

    def test_dataset_map_is_called(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, _, _, _, mock_ds, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        mock_ds.map.assert_called_once()

    def test_dataset_map_uses_batched_true(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, _, _, _, mock_ds, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = mock_ds.map.call_args
        assert kwargs.get("batched") is True

    def test_dataset_text_field_is_text_not_output(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, MockSFT, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = MockSFT.call_args
        assert kwargs.get("dataset_text_field") == "text"

    def test_format_alpaca_without_input(self):
        from training.lora_trainer import _format_alpaca
        result = _format_alpaca(
            {"instruction": ["What is AI?"], "input": [""], "output": ["AI is cool."]},
            eos_token="</s>",
        )
        text = result["text"][0]
        assert "What is AI?" in text
        assert "AI is cool." in text
        assert "### Instruction:" in text
        assert "### Response:" in text
        assert "</s>" in text

    def test_format_alpaca_with_input(self):
        from training.lora_trainer import _format_alpaca
        result = _format_alpaca(
            {"instruction": ["Translate"], "input": ["Hello"], "output": ["Bonjour"]},
            eos_token="</s>",
        )
        text = result["text"][0]
        assert "### Input:" in text
        assert "Hello" in text
        assert "Bonjour" in text

    def test_format_alpaca_eos_token_appended(self):
        from training.lora_trainer import _format_alpaca
        eos = "<|end_of_text|>"
        result = _format_alpaca(
            {"instruction": ["Q?"], "input": [""], "output": ["A."]},
            eos_token=eos,
        )
        assert result["text"][0].endswith(eos)

    def test_format_alpaca_batched_multiple_examples(self):
        from training.lora_trainer import _format_alpaca
        result = _format_alpaca(
            {
                "instruction": ["Q1?", "Q2?", "Q3?"],
                "input": ["", "context", ""],
                "output": ["A1.", "A2.", "A3."],
            },
            eos_token="</s>",
        )
        assert len(result["text"]) == 3
        assert "Q2?" in result["text"][1]
        assert "context" in result["text"][1]


# ── Model persistence ────────────────────────────────────────────────────────

class TestModelPersistence:

    def test_model_save_pretrained_called_with_output_dir(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        mock_model.save_pretrained.assert_called_once_with(output_dir)

    def test_model_save_pretrained_called_exactly_once(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        assert mock_model.save_pretrained.call_count == 1


# ── Correct arguments to dependencies ───────────────────────────────────────

class TestCorrectArguments:

    def test_from_pretrained_uses_correct_model_name(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer,
                            base_model="unsloth/llama-3-8b-bnb-4bit")
        _, kwargs = fast_lm.from_pretrained.call_args
        assert kwargs.get("model_name") == "unsloth/llama-3-8b-bnb-4bit"

    def test_from_pretrained_uses_load_in_4bit(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = fast_lm.from_pretrained.call_args
        assert kwargs.get("load_in_4bit") is True

    def test_from_pretrained_uses_max_seq_length_2048(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = fast_lm.from_pretrained.call_args
        assert kwargs.get("max_seq_length") == 2048

    def test_lora_rank_passed_to_get_peft_config(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer, lora_rank=32)
        fast_lm.get_peft_config.assert_called_once_with(lora_rank=32)

    def test_training_args_fp16_true(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, _, MockArgs, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = MockArgs.call_args
        assert kwargs.get("fp16") is True

    def test_training_args_output_dir_correct(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, _, MockArgs, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = MockArgs.call_args
        assert kwargs.get("output_dir") == output_dir


# ── Parametrized lora_rank ────────────────────────────────────────────────────

@pytest.mark.parametrize("lora_rank", [8, 16, 32, 64])
def test_lora_rank_values_accepted(dataset_file, output_dir, mock_model, mock_tokenizer, lora_rank):
    fast_lm, *_ = _run(dataset_file, output_dir, mock_model, mock_tokenizer, lora_rank=lora_rank)
    fast_lm.get_peft_config.assert_called_once_with(lora_rank=lora_rank)
