import tempfile
from pathlib import Path
from typing import Callable, Optional
import os
import structlog
import shutil
import uuid
from datetime import datetime
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import EvolInstruct, TextGeneration
from distilabel.steps import LoadDataFromDicts, FilterStep, KeepColumns
from distilabel.llms import OpenAILLM, vLLM

from config import DistillationConfig, QualityMode
from pipeline.document_loader import semantic_chunk
from pipeline.exporter import export_alpaca
from training.lora_trainer import train_lora
from publish.hf_publisher import publish_dataset

logger = structlog.get_logger(__name__)


def run_distillation(
    cfg: DistillationConfig,
    source_file: Path,
    progress_callback: Optional[Callable[[int], None]] = None
) -> Path:
    """
    Run the complete distillation pipeline to generate training dataset.

    Args:
        cfg: Configuration for the distillation process
        source_file: Path to the source document
        progress_callback: Optional callback for progress updates (0-100)

    Returns:
        Path to the generated dataset file

    Raises:
        RuntimeError: If pipeline execution or export fails
        ValueError: If configuration is invalid
    """
    logger.info("Starting distillation", config=cfg.model_dump(exclude_none=True))

    # Validate inputs
    if not cfg.teacher_model or not cfg.teacher_model.strip():
        raise ValueError("Teacher model is required")

    # Read and process source file
    try:
        text = source_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        logger.exception("Failed to decode source file")
        raise ValueError(f"Source file must be valid UTF-8 encoded text: {e}")
    except Exception as e:
        logger.exception("Failed to read source file")
        raise ValueError(f"Failed to read source file: {e}")

    if not text or not text.strip():
        raise ValueError("Source document is empty")

    # Chunk the document
    try:
        chunks = semantic_chunk(text)
    except Exception as e:
        logger.exception("Failed to chunk document")
        raise RuntimeError(f"Failed to process document: {e}")

    if not chunks:
        raise ValueError("Failed to extract any content from the document")

    # Limit prompts to dataset size
    prompts = [
        f"Explain the following concept from the document clearly and completely:\n\n{c}"
        for c in chunks
    ][:cfg.dataset_size]

    if not prompts:
        raise ValueError("No prompts generated from document")

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.jsonl"

        # Use secure, unique output filename to prevent path traversal
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        final_path = Path(f"alpaca_dataset_{timestamp}_{unique_id}.jsonl")

        # Initialize LLM
        try:
            if cfg.use_vllm:
                llm = vLLM(
                    model=cfg.teacher_model.split(",")[0],
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature
                )
            else:
                if not cfg.api_key:
                    raise ValueError(
                        "OpenAI API key is required when not using vLLM. "
                        "Please provide it in the sidebar or set OPENAI_API_KEY environment variable."
                    )
                llm = OpenAILLM(
                    model=cfg.teacher_model.split(",")[0],
                    api_key=cfg.api_key,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature
                )
        except ValueError:
            raise
        except Exception as e:
            logger.exception("Failed to initialize LLM")
            raise RuntimeError(f"Failed to initialize model: {e}")

        # Run pipeline
        try:
            num_evolutions = 1
            if cfg.quality_mode == QualityMode.RESEARCH:
                num_evolutions = 3
            elif cfg.quality_mode == QualityMode.BALANCED:
                num_evolutions = 2

            pipeline = Pipeline(steps=[
                LoadDataFromDicts(data=[{"instruction": p} for p in prompts]),
                EvolInstruct(llm=llm, num_evolutions=num_evolutions),
                TextGeneration(llm=llm),
                KeepColumns(columns=["instruction", "output"]),
                FilterStep(condition="len(output) > 100"),
            ])

            distiset = pipeline.run()
            distiset["default"].to_json(str(raw_path))
        except Exception as e:
            logger.exception("Pipeline execution failed")
            raise RuntimeError(f"Failed to generate dataset: {e}")

        # Export dataset
        try:
            records_exported = export_alpaca(str(raw_path), str(final_path))
            logger.info(f"Exported {records_exported} records to Alpaca format")
        except Exception as e:
            logger.exception("Export failed")
            raise RuntimeError(f"Failed to export dataset: {e}")

        # Validate output
        if not os.path.exists(final_path):
            raise RuntimeError("Export failed: output file not created")

        file_size = os.path.getsize(final_path)
        if file_size == 0:
            raise RuntimeError("Export failed: output file is empty")

        # Count lines
        with open(final_path) as f:
            line_count = sum(1 for _ in f)

        logger.info(f"Dataset generated with {line_count} samples")

        # Copy to workspace directory for user access
        workspace_path = Path("alpaca_dataset.jsonl")
        shutil.copy(str(final_path), str(workspace_path))

        # Train model if requested
        if cfg.train_model:
            try:
                train_lora(str(workspace_path), cfg.base_model, "trained_adapter", cfg.lora_rank)
            except Exception as e:
                logger.exception("Training failed")
                raise RuntimeError(f"Failed to train model: {e}")

        # Publish to Hugging Face if requested
        if cfg.publish_dataset and cfg.hf_repo:
            try:
                publish_dataset(str(workspace_path), cfg.hf_repo, os.getenv("HF_TOKEN"))
            except Exception as e:
                logger.exception("Publishing failed")
                raise RuntimeError(f"Failed to publish dataset: {e}")

        if progress_callback:
            progress_callback(100)

        logger.info("Finished", path=str(workspace_path))
        return workspace_path
