import random
from typing import List
import structlog

logger = structlog.get_logger(__name__)

TEMPLATES = [
    "Rewrite this prompt to require deeper reasoning and multi-step thinking:",
    "Transform this into a multi-step task with clear constraints:",
    "Increase the complexity and specificity of the following question:",
    "Make this prompt more challenging by adding edge cases and requirements:",
]

def evolve(prompts: List[str], multiplier: int = 3) -> List[str]:
    """Proper per-prompt evolution (original version only evolved the last prompt)."""
    if multiplier < 1:
        return prompts[:]

    evolved: List[str] = []
    random.seed(42)  # reproducible

    for p in prompts:
        evolved.append(p)  # keep original
        for _ in range(multiplier - 1):
            t = random.choice(TEMPLATES)
            evolved.append(f"{t}\n\nOriginal prompt:\n{p}")

    logger.info("Evolution complete", original=len(prompts), evolved=len(evolved))
    return evolved
