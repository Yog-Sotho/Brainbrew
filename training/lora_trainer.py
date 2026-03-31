"""
LoRA fine-tuning via Unsloth + TRL.

All heavy GPU imports (transformers, trl, unsloth, datasets) are intentionally
deferred to inside train_lora(). This keeps the module importable on CPU-only
hosts where these packages are not installed, while still working correctly
when called on a GPU machine.
"""
from __future__ import annotations

from typing import Any


# ── FIX C-03: formatting function for proper instruction+output training ────
def _format_alpaca(examples: dict[str, list[str]], eos_token: str = "</s>") -> dict[str, list[str]]:
    """Convert Alpaca-format records into a single 'text' column for SFTTrainer.

    This ensures the model trains on the full instruction+response pair,
    not just the output field alone.  Supports batched=True mode.

    Args:
        examples: Dict with lists for 'instruction', 'input', 'output' keys.
        eos_token: The model's end-of-sequence token to append.

    Returns:
        Dict with a single 'text' key containing formatted training strings.
    """
    texts: list[str] = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        if inp and inp.strip():
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n{output}{eos_token}"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}{eos_token}"
            )
        texts.append(prompt)
    return {"text": texts}


def train_lora(
    dataset_path: str,
    base_model: str,
    output_dir: str,
    lora_rank: int = 16,
) -> None:
    """Run LoRA fine-tuning on the generated dataset.

    All GPU-only imports are lazy — inside this function, not at module level.
    Importing at module level caused ModuleNotFoundError on every startup on
    hosts without GPU packages, even when LoRA training was never requested.
    """
    try:
        from datasets import load_dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise RuntimeError(
            f"LoRA training requires GPU packages that are not installed: {e}. "
            "Run: pip install transformers trl unsloth bitsandbytes"
        ) from e

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # FIX C-03: format full instruction+output into 'text' column
    dataset = dataset.map(
        lambda batch: _format_alpaca(batch, eos_token=tokenizer.eos_token or "</s>"),
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=60,
            learning_rate=2e-4,
            output_dir=output_dir,
            fp16=True,
        ),
        peft_config=FastLanguageModel.get_peft_config(lora_rank=lora_rank),
    )

    trainer.train()
    model.save_pretrained(output_dir)
