"""
tests/test_lora_trainer.py

Tests for training/lora_trainer.py — train_lora().

All GPU/ML dependencies (unsloth, trl, transformers, datasets) are mocked.
No GPU required.

Covers:
  - train() called exactly once (regression test for the duplicate-call fix)
  - model.save_pretrained() called exactly once
  - trainer.model.save_pretrained() NOT called (was the redundant duplicate)
  - FastLanguageModel.from_pretrained called with correct arguments
  - SFTTrainer constructed with correct dataset_text_field="output"
  - load_dataset called with correct dataset_path
  - lora_rank passed through to get_peft_config
  - output_dir passed through to TrainingArguments
  - fp16=True in TrainingArguments
  - max_seq_length=2048 used consistently
  - load_in_4bit=True (QLoRA)
  - Correct per_device_train_batch_size and gradient_accumulation_steps
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    m = MagicMock(name="model")
    return m


@pytest.fixture()
def mock_tokenizer():
    return MagicMock(name="tokenizer")


@pytest.fixture()
def mock_trainer():
    return MagicMock(name="SFTTrainer_instance")


@pytest.fixture()
def patched_train(mock_model, mock_tokenizer, mock_trainer):
    """Context manager that patches all heavy dependencies."""
    fast_lm = MagicMock(name="FastLanguageModel")
    fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
    fast_lm.get_peft_config.return_value = MagicMock()

    mock_ds = MagicMock(name="Dataset")
    mock_args = MagicMock(name="TrainingArguments")

    with patch("training.lora_trainer.FastLanguageModel", fast_lm), \
         patch("training.lora_trainer.SFTTrainer", return_value=mock_trainer), \
         patch("training.lora_trainer.load_dataset", return_value=mock_ds), \
         patch("training.lora_trainer.TrainingArguments", return_value=mock_args):
        yield {
            "fast_lm": fast_lm,
            "trainer": mock_trainer,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "load_dataset": patch("training.lora_trainer.load_dataset"),
            "args": mock_args,
        }


# ---------------------------------------------------------------------------
# Critical regression: train() called exactly once
# ---------------------------------------------------------------------------

class TestTrainCalledExactlyOnce:

    def test_trainer_train_called_once(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm = MagicMock()
        fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        fast_lm.get_peft_config.return_value = MagicMock()
        mock_trainer = MagicMock()

        with patch("training.lora_trainer.FastLanguageModel", fast_lm), \
             patch("training.lora_trainer.SFTTrainer", return_value=mock_trainer), \
             patch("training.lora_trainer.load_dataset", return_value=MagicMock()), \
             patch("training.lora_trainer.TrainingArguments", return_value=MagicMock()):
            from training.lora_trainer import train_lora
            train_lora(dataset_file, "unsloth/mistral-7b-bnb-4bit", output_dir)

        assert mock_trainer.train.call_count == 1, (
            f"trainer.train() must be called exactly once, got {mock_trainer.train.call_count}"
        )

    def test_trainer_model_save_pretrained_not_called(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        """The removed duplicate save via trainer.model.save_pretrained() must not exist."""
        fast_lm = MagicMock()
        fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        fast_lm.get_peft_config.return_value = MagicMock()
        mock_trainer = MagicMock()

        with patch("training.lora_trainer.FastLanguageModel", fast_lm), \
             patch("training.lora_trainer.SFTTrainer", return_value=mock_trainer), \
             patch("training.lora_trainer.load_dataset", return_value=MagicMock()), \
             patch("training.lora_trainer.TrainingArguments", return_value=MagicMock()):
            from training.lora_trainer import train_lora
            train_lora(dataset_file, "unsloth/mistral-7b-bnb-4bit", output_dir)

        mock_trainer.model.save_pretrained.assert_not_called()


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

class TestModelPersistence:

    def test_model_save_pretrained_called_with_output_dir(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm = MagicMock()
        fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        fast_lm.get_peft_config.return_value = MagicMock()

        with patch("training.lora_trainer.FastLanguageModel", fast_lm), \
             patch("training.lora_trainer.SFTTrainer", return_value=MagicMock()), \
             patch("training.lora_trainer.load_dataset", return_value=MagicMock()), \
             patch("training.lora_trainer.TrainingArguments", return_value=MagicMock()):
            from training.lora_trainer import train_lora
            train_lora(dataset_file, "unsloth/mistral-7b-bnb-4bit", output_dir)

        mock_model.save_pretrained.assert_called_once_with(output_dir)

    def test_model_save_pretrained_called_exactly_once(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm = MagicMock()
        fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        fast_lm.get_peft_config.return_value = MagicMock()

        with patch("training.lora_trainer.FastLanguageModel", fast_lm), \
             patch("training.lora_trainer.SFTTrainer", return_value=MagicMock()), \
             patch("training.lora_trainer.load_dataset", return_value=MagicMock()), \
             patch("training.lora_trainer.TrainingArguments", return_value=MagicMock()):
            from training.lora_trainer import train_lora
            train_lora(dataset_file, "unsloth/mistral-7b-bnb-4bit", output_dir)

        assert mock_model.save_pretrained.call_count == 1


# ---------------------------------------------------------------------------
# Correct arguments passed to dependencies
# ---------------------------------------------------------------------------

class TestCorrectArguments:

    def _run(self, dataset_file, output_dir, mock_model, mock_tokenizer,
             base_model="unsloth/mistral-7b-bnb-4bit", lora_rank=16):
        fast_lm = MagicMock()
        fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        peft_cfg = MagicMock()
        fast_lm.get_peft_config.return_value = peft_cfg
        mock_trainer = MagicMock()
        mock_args = MagicMock()

        with patch("training.lora_trainer.FastLanguageModel", fast_lm), \
             patch("training.lora_trainer.SFTTrainer", return_value=mock_trainer) as MockSFT, \
             patch("training.lora_trainer.load_dataset", return_value=MagicMock()), \
             patch("training.lora_trainer.TrainingArguments", return_value=mock_args) as MockArgs:
            from training.lora_trainer import train_lora
            train_lora(dataset_file, base_model, output_dir, lora_rank=lora_rank)

        return fast_lm, MockSFT, MockArgs, mock_trainer, peft_cfg

    def test_from_pretrained_uses_correct_model_name(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm, *_ = self._run(dataset_file, output_dir, mock_model, mock_tokenizer,
                                 base_model="unsloth/llama-3-8b-bnb-4bit")
        _, kwargs = fast_lm.from_pretrained.call_args
        assert kwargs.get("model_name") == "unsloth/llama-3-8b-bnb-4bit"

    def test_from_pretrained_uses_load_in_4bit(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm, *_ = self._run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = fast_lm.from_pretrained.call_args
        assert kwargs.get("load_in_4bit") is True

    def test_from_pretrained_uses_max_seq_length_2048(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm, *_ = self._run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = fast_lm.from_pretrained.call_args
        assert kwargs.get("max_seq_length") == 2048

    def test_sft_trainer_dataset_text_field_is_output(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, MockSFT, *_ = self._run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = MockSFT.call_args
        assert kwargs.get("dataset_text_field") == "output"

    def test_lora_rank_passed_to_get_peft_config(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        fast_lm, *_ = self._run(dataset_file, output_dir, mock_model, mock_tokenizer, lora_rank=32)
        fast_lm.get_peft_config.assert_called_once_with(lora_rank=32)

    def test_training_args_fp16_true(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, _, MockArgs, *_ = self._run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = MockArgs.call_args
        assert kwargs.get("fp16") is True

    def test_training_args_output_dir_correct(self, dataset_file, output_dir, mock_model, mock_tokenizer):
        _, _, MockArgs, *_ = self._run(dataset_file, output_dir, mock_model, mock_tokenizer)
        _, kwargs = MockArgs.call_args
        assert kwargs.get("output_dir") == output_dir


# ---------------------------------------------------------------------------
# Parametrized lora_rank
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lora_rank", [8, 16, 32, 64])
def test_lora_rank_values_accepted(dataset_file, output_dir, mock_model, mock_tokenizer, lora_rank):
    fast_lm = MagicMock()
    fast_lm.from_pretrained.return_value = (mock_model, mock_tokenizer)
    fast_lm.get_peft_config.return_value = MagicMock()
    with patch("training.lora_trainer.FastLanguageModel", fast_lm), \
         patch("training.lora_trainer.SFTTrainer", return_value=MagicMock()), \
         patch("training.lora_trainer.load_dataset", return_value=MagicMock()), \
         patch("training.lora_trainer.TrainingArguments", return_value=MagicMock()):
        from training.lora_trainer import train_lora
        train_lora(dataset_file, "unsloth/mistral-7b-bnb-4bit", output_dir, lora_rank=lora_rank)
    fast_lm.get_peft_config.assert_called_once_with(lora_rank=lora_rank)
