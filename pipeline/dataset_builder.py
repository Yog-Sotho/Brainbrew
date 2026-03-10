import json
from pathlib import Path
from typing import List
from tqdm import tqdm
import structlog

from engines.base import BaseEngine

logger = structlog.get_logger(__name__)

def build_with_vllm(
    prompts: List[str],
    engine: BaseEngine,
    output_path: str | Path,
    batch_size: int = 64,
) -> None:
    """Build JSONL dataset using batch generation (vLLM or compatible engine)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Building dataset"):
            batch = prompts[i : i + batch_size]
            try:
                outputs = engine.generate_batch(batch)
                for p, o in zip(batch, outputs):
                    rec = {"instruction": p, "output": o}
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                logger.error("Batch failed", error=str(e))
                continue

    logger.info("Dataset built", path=output_path, samples=len(prompts))
