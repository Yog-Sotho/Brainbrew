"""
Custom distilabel steps.

No `from __future__ import annotations` in this module, on purpose: distilabel
finds a step's input parameter by its *runtime* `StepInput` type hint, and
postponed (string) annotations make DAG validation fail with "should have a
parameter with type hint `StepInput`".
"""
from typing import Any

from distilabel.steps import StepInput
from distilabel.steps.base import Step
from distilabel.typing import StepOutput

MIN_OUTPUT_CHARS: int = 100


class ToCanonical(Step):
    """Turn a generated row into canonical record columns.

    The *evolved* instruction becomes the instruction, because that is the
    question TextGeneration answered; the seed prompt is kept as provenance.
    Rows whose evolution failed or whose answer is shorter than `min_length`
    characters are dropped.
    """

    min_length: int = MIN_OUTPUT_CHARS

    @property
    def inputs(self) -> list[str]:
        return ["instruction", "evolved_instruction", "generation"]

    @property
    def outputs(self) -> list[str]:
        return ["instruction", "output", "seed"]

    def process(self, inputs: StepInput) -> StepOutput:  # type: ignore[override]
        kept: list[dict[str, Any]] = []
        for row in inputs:
            evolved = row.get("evolved_instruction")
            generation = row.get("generation")
            if not isinstance(evolved, str) or not evolved.strip():
                continue
            if not isinstance(generation, str) or len(generation.strip()) < self.min_length:
                continue
            kept.append({
                **row,
                "seed": row.get("instruction"),
                "instruction": evolved.strip(),
                "output": generation.strip(),
            })
        yield kept
