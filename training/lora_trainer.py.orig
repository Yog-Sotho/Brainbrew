from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def _format_alpaca(examples: dict, eos_token: str) -> dict:
    """
    Format Alpaca instruction/output pairs into full training text.

    FIX: Previously dataset_text_field="output" trained the model only on answers,
    ignoring the instruction entirely. The model learned to generate text but never
    learned to follow instructions. Using the full prompt+response text here means
    the model learns the connection between question and answer — i.e. it actually
    becomes useful for your domain.
    """
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        if input_text:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )
        texts.append(text + eos_token)
    return {"text": texts}


def train_lora(dataset_path: str, base_model: str, output_dir: str, lora_rank: int = 16) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=base_model, max_seq_length=2048, dtype=None, load_in_4bit=True)
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    # FIX: map full instruction+output formatting so the model learns to follow instructions,
    # not just generate text. One mapping call replaces the broken dataset_text_field="output".
    dataset = dataset.map(
        lambda examples: _format_alpaca(examples, tokenizer.eos_token),
        batched=True,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",   # FIX: was "output" — now uses full formatted text
        max_seq_length=2048,
        args=TrainingArguments(per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=10, max_steps=60, learning_rate=2e-4, output_dir=output_dir, fp16=True),
        peft_config=FastLanguageModel.get_peft_config(lora_rank=lora_rank),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    # FIX: removed duplicate trainer.train() call that was here — doubled cost/time
    # FIX: removed redundant trainer.model.save_pretrained(output_dir) that followed
