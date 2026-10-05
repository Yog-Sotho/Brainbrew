"""
tests/test_lora_trainer.py

Unit tests for the training-example formatting, plus contract tests that run
*real* TRL + PEFT LoRA training on CPU with tiny Hub models. The contract tests
are skipped unless the training packages are installed (the `train-contract`
CI job installs them).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.records import Record, write_records
from training.lora_trainer import to_training_example, train_lora

REC = Record(instruction="Explain recursion.", output="Recursion is when a function calls itself.")
REC_WITH_INPUT = Record(instruction="Summarise.", input="Long text here.", output="A summary.")


class TestTrainingExamples:

    def test_chat_model_gets_conversational_prompt_completion(self):
        assert to_training_example(REC_WITH_INPUT, chat=True) == {
            "prompt": [{"role": "user", "content": "Summarise.\n\nLong text here."}],
            "completion": [{"role": "assistant", "content": "A summary."}],
        }

    def test_base_model_gets_alpaca_text(self):
        ex = to_training_example(REC, chat=False, eos_token="</s>")
        assert ex["prompt"] == "### Instruction:\nExplain recursion.\n\n### Response:\n"
        assert ex["completion"] == "Recursion is when a function calls itself.</s>"

    def test_base_model_input_section(self):
        ex = to_training_example(REC_WITH_INPUT, chat=False)
        assert "### Input:\nLong text here.\n\n### Response:\n" in ex["prompt"]

    def test_instruction_and_output_both_present(self):
        # The original bug trained on the output column alone.
        ex = to_training_example(REC, chat=False)
        assert REC.instruction in ex["prompt"] and REC.output in ex["completion"]


def test_missing_packages_give_install_hint(tmp_path):
    # Block torch, the first import, so nothing else gets imported (and then
    # dropped from sys.modules) inside patch.dict; C extensions cannot re-import.
    with patch.dict(sys.modules, {"torch": None}), \
         pytest.raises(RuntimeError, match="uv sync --extra train"):
        train_lora(tmp_path / "r.jsonl", "any/model", tmp_path / "adapter")


# ── Real training (CPU, tiny models) ─────────────────────────────────────────

CHAT_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM-Instruct-2507"
BASE_MODEL = "trl-internal-testing/tiny-GPTNeoXForCausalLM"


@pytest.fixture()
def records_file(tmp_path: Path) -> Path:
    path = tmp_path / "records.jsonl"
    write_records(path, [
        Record(instruction=f"Explain concept {i}.", input="ctx" if i % 2 else "",
               output=f"Concept {i} explained in a couple of sentences. " * 3)
        for i in range(8)
    ])
    return path


@pytest.fixture()
def training_stack():
    for mod in ("torch", "transformers", "trl", "peft", "datasets"):
        pytest.importorskip(mod)


@pytest.mark.usefixtures("training_stack")
class TestRealTraining:

    @pytest.mark.parametrize("model", [CHAT_MODEL, BASE_MODEL])
    def test_trains_and_saves_loadable_adapter(self, model, records_file, tmp_path):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        out = train_lora(records_file, model, tmp_path / "adapter", lora_rank=4,
                         max_steps=1, max_length=128)

        files = {p.name for p in out.iterdir()}
        assert {"adapter_config.json", "adapter_model.safetensors"} <= files
        assert not {"checkpoints", "training_args.bin", "optimizer.pt"} & files
        cfg = json.loads((out / "adapter_config.json").read_text())
        assert cfg["r"] == 4 and cfg["lora_alpha"] == 4

        base = AutoModelForCausalLM.from_pretrained(model)
        PeftModel.from_pretrained(base, str(out))  # loads cleanly

    def test_loss_only_on_completion(self, records_file, tmp_path):
        """Prompt tokens are masked, so the model learns the answers, not the questions."""
        import trl

        seen = {}
        real_init = trl.SFTTrainer.__init__

        def spy(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            seen["batch"] = self.data_collator([self.train_dataset[0]])

        with patch.object(trl.SFTTrainer, "__init__", spy):
            train_lora(records_file, CHAT_MODEL, tmp_path / "adapter", max_steps=1, max_length=128)
        labels = seen["batch"]["labels"][0].tolist()
        masked = [lab == -100 for lab in labels]
        first_trained = masked.index(False)
        assert first_trained > 0                    # the prompt is masked...
        assert not any(masked[first_trained:])      # ...and every answer token is trained

    def test_empty_records_rejected(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No training records"):
            train_lora(empty, CHAT_MODEL, tmp_path / "adapter")
