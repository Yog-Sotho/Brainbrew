from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def train_lora(dataset_path, base_model, output_dir):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    dataset = load_dataset(
        "json",
        data_files=dataset_path,
        split="train"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="output",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir=output_dir
        ),
    )

    trainer.train()

    trainer.model.save_pretrained(output_dir)
