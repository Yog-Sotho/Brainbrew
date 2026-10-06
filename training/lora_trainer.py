"""
LoRA fine-tuning with TRL + PEFT.

Trains on canonical records (pipeline/records.py), whatever export format the
user picked. Models with a chat template get conversational prompt-completion
data, so the template matches what the model sees at inference; base models
without one get an Alpaca-style text prompt. Either way the loss is computed on
the answer only.

All heavy imports (torch, transformers, trl, peft, datasets) are deferred to
inside train_lora(), so this module stays importable on CPU-only hosts without
the `train` extra installed.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pipeline.records import Record, read_records

ALPACA_PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"
ALPACA_PROMPT_WITH_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


def to_training_example(rec: Record, chat: bool, eos_token: str = "") -> dict[str, Any]:
    """One record as a TRL prompt-completion example.

    Args:
        rec: Canonical record.
        chat: True for a model with a chat template (conversational format).
        eos_token: Appended to text completions so the model learns to stop.
    """
    if chat:
        return {
            "prompt": [{"role": "user", "content": rec.prompt}],
            "completion": [{"role": "assistant", "content": rec.output}],
        }
    template = ALPACA_PROMPT_WITH_INPUT if rec.input.strip() else ALPACA_PROMPT
    return {
        "prompt": template.format(instruction=rec.instruction, input=rec.input),
        "completion": rec.output + eos_token,
    }


def train_lora(
    records_path: Path,
    base_model: str,
    output_dir: Path,
    lora_rank: int = 16,
    *,
    num_train_epochs: float = 1.0,
    max_steps: int = -1,
    max_length: int = 2048,
    load_in_4bit: bool | None = None,
) -> Path:
    """Fine-tune a LoRA adapter on canonical records and save it to *output_dir*.

    Args:
        records_path: Canonical records JSONL (runs/<id>/records.jsonl).
        base_model: Hugging Face model id or local path.
        output_dir: Where the adapter (and tokenizer) are saved.
        lora_rank: LoRA rank; alpha is set equal to the rank.
        num_train_epochs: Passes over the dataset (ignored if max_steps > 0).
        max_steps: Hard step limit, mainly for smoke tests; -1 means use epochs.
        max_length: Truncation length in tokens.
        load_in_4bit: QLoRA via bitsandbytes. Default: on when CUDA is available.

    Returns:
        output_dir.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTTrainer
    except ImportError as e:
        raise RuntimeError(
            f"LoRA training needs the training packages, which are not installed ({e}). "
            "Install them with: uv sync --extra train"
        ) from e

    records = read_records(records_path)
    if not records:
        raise ValueError(f"No training records in {records_path}")

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    fp16 = cuda and not bf16
    tokenizer, model = _load(base_model, bf16, fp16, cuda if load_in_4bit is None else load_in_4bit)
    chat = bool(getattr(tokenizer, "chat_template", None))
    eos = "" if chat else (tokenizer.eos_token or "")
    dataset = Dataset.from_list([to_training_example(r, chat, eos) for r in records])

    output_dir.mkdir(parents=True, exist_ok=True)
    # Trainer scratch output (logs, optimizer state) stays out of the adapter
    # folder, which users download as a zip.
    with tempfile.TemporaryDirectory(prefix="brainbrew-train-") as scratch:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                lora_dropout=0.05,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
            args=_sft_args(scratch, num_train_epochs, max_steps, max_length, bf16, fp16, cuda),
        )
        trainer.train()
        trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _load(base_model: str, bf16: bool, fp16: bool, load_in_4bit: bool) -> tuple[Any, Any]:
    """The tokenizer (with a pad token) and the model, 4-bit quantised for QLoRA if asked."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    return tokenizer, AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)


def _sft_args(
    output_dir: str, epochs: float, max_steps: int, max_length: int, bf16: bool, fp16: bool, cuda: bool
) -> Any:
    from trl import SFTConfig

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # a float in [0, 1) is a ratio of total steps
        max_length=max_length,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=cuda,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
