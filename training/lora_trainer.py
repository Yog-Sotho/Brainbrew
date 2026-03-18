"""
LoRA fine-tuning via Unsloth + TRL.

All heavy GPU imports (transformers, trl, unsloth) are intentionally deferred
to inside train_lora(). This keeps the module importable on CPU-only hosts
(e.g. Streamlit Cloud) where these packages are not installed, while still
working correctly when called on a GPU machine.
"""


def train_lora(
    dataset_path: str,
    base_model: str,
    output_dir: str,
    lora_rank: int = 16,
) -> None:
    # FIX: all GPU-only imports are lazy — inside the function, not at module level.
    # Importing at module level caused ModuleNotFoundError on every startup on
    # Streamlit Cloud, even when LoRA training was never requested.
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="output",
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
    # FIX: original had trainer.train() called twice and used both
    # trainer.model.save_pretrained and model.save_pretrained — deduplicated here.
    model.save_pretrained(output_dir)
