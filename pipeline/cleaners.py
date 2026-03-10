import json
from pathlib import Path
from typing import Set
import structlog

logger = structlog.get_logger(__name__)

REFUSALS = [
    "as an ai language model",
    "i cannot assist",
    "i'm unable to",
    "sorry, i can't",
    "as an ai i cannot",
]

def is_refusal(text: str) -> bool:
    """Fast refusal filter."""
    t = text.lower()
    return any(r in t for r in REFUSALS)

def clean_dataset(input_path: str | Path, output_path: str | Path) -> None:
    """Clean refusals and deduplicate by instruction."""
    seen: Set[str] = set()
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                obj = json.loads(line.strip())
                if is_refusal(obj.get("output", "")):
                    continue
                key = obj.get("instruction", "")
                if key and key not in seen:
                    seen.add(key)
                    fout.write(json.dumps(obj) + "\n")
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Skipped malformed line")
                continue

    logger.info("Dataset cleaned", input=input_path, output=output_path, samples=len(seen))
            seen.add(key)

            fout.write(json.dumps(obj)+"\n")
