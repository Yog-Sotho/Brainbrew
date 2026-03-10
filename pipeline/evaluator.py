import json
from pathlib import Path
from typing import Optional
import structlog

from engines.judge_engine import JudgeEngine

logger = structlog.get_logger(__name__)

def evaluate_dataset(
    input_path: str | Path,
    output_path: str | Path,
    judge: JudgeEngine,
    threshold: float = 7.0,
) -> None:
    """Score and filter dataset using judge engine."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                obj = json.loads(line.strip())
                score = judge.score(obj["instruction"], obj.get("output", ""))
                if score >= threshold:
                    obj["score"] = score
                    fout.write(json.dumps(obj) + "\n")
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    logger.info("Evaluation complete", threshold=threshold, output=output_path)
