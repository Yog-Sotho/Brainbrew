"""
LoRA trainer module for fine-tuning models using Unsloth.
Handles the training of LoRA adapters on generated datasets.
"""
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import structlog
import os
from typing import Optional

logger = structlog.get_logger(__name__)


def validate_dataset_path(dataset_path: str) -> str:
    """
    Validate that the dataset file exists and is readable.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        The validated path

    Raises:
        ValueError: If path is invalid or file doesn't exist
    """
    if not dataset_path:
        raise ValueError("Dataset path is required")

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset file not found: {dataset_path}")

    if not os.path.isfile(dataset_path):
        raise ValueError(f"Dataset path is not a file: {dataset_path}")

    file_size = os.path.getsize(dataset_path)
    if file_size == 0:
        raise ValueError("Dataset file is empty")

    return dataset_path


def train_lora(
    dataset_path: str,
    base_model: str,
    output_dir: str,
    lora_rank: int = 16,
    max_seq_length: int = 2048,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 10,
    max_steps: int = 60,
    learning_rate: float = 2e-4,
    fp16: bool = True
) -> str:
    """
    Train a LoRA adapter on the provided dataset.

    Args:
        dataset_path: Path to the training dataset (JSON/JSONL format)
        base_model: The base model to fine-tune (e.g., 'unsloth/mistral-7b-bnb-4bit')
        output_dir: Directory to save the trained adapter
        lora_rank: LoRA rank parameter (default: 16)
        max_seq_length: Maximum sequence length (default: 2048)
        per_device_train_batch_size: Batch size per device (default: 2)
        gradient_accumulation_steps: Gradient accumulation steps (default: 4)
        warmup_steps: Number of warmup steps (default: 10)
        max_steps: Maximum training steps (default: 60)
        learning_rate: Learning rate (default: 2e-4)
        fp16: Whether to use FP16 precision (default: True)

    Returns:
        Path to the saved adapter

    Raises:
        ValueError: If validation fails
        RuntimeError: If training fails
    """
    logger.info(
        "Starting LoRA training",
        dataset=dataset_path,
        model=base_model,
        output_dir=output_dir,
        lora_rank=lora_rank
    )

    # Validate dataset path
    dataset_path = validate_dataset_path(dataset_path)

    # Validate base model
    if not base_model or not base_model.strip():
        raise ValueError("Base model name is required")

    # Validate output directory
    if not output_dir or not output_dir.strip():
        raise ValueError("Output directory is required")

    # Validate LoRA rank
    if lora_rank < 8:
        raise ValueError("LoRA rank must be at least 8")

    # Load base model
    try:
        logger.info("Loading base model", model=base_model)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True
        )
    except Exception as e:
        logger.exception("Failed to load base model")
        raise RuntimeError(f"Failed to load base model '{base_model}': {e}")

    # Load dataset
    try:
        logger.info("Loading training dataset", path=dataset_path)
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        if dataset.num_rows == 0:
            raise ValueError("Training dataset is empty")

        logger.info("Dataset loaded", num_rows=dataset.num_rows)
    except ValueError:
        raise
    except Exception as e:
        logger.exception("Failed to load dataset")
        raise RuntimeError(f"Failed to load dataset: {e}")

    # Configure PEFT
    try:
        peft_config = FastLanguageModel.get_peft_config(lora_rank=lora_rank)
    except Exception as e:
        logger.exception("Failed to configure PEFT")
        raise RuntimeError(f"Failed to configure PEFT: {e}")

    # Create trainer
    try:
        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            output_dir=output_dir,
            fp16=fp16,
            logging_steps=10,
            save_steps=20,
            save_total_limit=2,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="output",
            max_seq_length=max_seq_length,
            args=training_args,
            peft_config=peft_config,
        )
    except Exception as e:
        logger.exception("Failed to create trainer")
        raise RuntimeError(f"Failed to create trainer: {e}")

    # Train model
    try:
        logger.info("Starting training", max_steps=max_steps)
        trainer.train()
    except Exception as e:
        logger.exception("Training failed")
        raise RuntimeError(f"Training failed: {e}")

    # Save the trained adapter
    try:
        logger.info("Saving trained adapter", output_dir=output_dir)
        model.save_pretrained(output_dir)
        trainer.model.save_pretrained(output_dir)
    except Exception as e:
        logger.exception("Failed to save model")
        raise RuntimeError(f"Failed to save trained model: {e}")

    logger.info("Training completed successfully", output_dir=output_dir)
    return output_dir
