import json
from pathlib import Path
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

def dataset_stats(path: str | Path) -> dict:
    """Compute dataset statistics."""
    path = Path(path)
    lengths = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                lengths.append(len(obj.get("output", "").split()))
            except (json.JSONDecodeError, KeyError):
                continue

    if not lengths:
        return {"samples": 0, "avg_tokens": 0.0, "max_tokens": 0, "min_tokens": 0}

    stats = {
        "samples": len(lengths),
        "avg_tokens": float(np.mean(lengths)),
        "max_tokens": int(np.max(lengths)),
        "min_tokens": int(np.min(lengths)),
        "median_tokens": float(np.median(lengths)),
    }
    logger.info("Dataset stats", **stats)
    return stats
