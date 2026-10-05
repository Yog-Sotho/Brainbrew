"""Deterministic offline LLM for running the *real* distilabel pipeline in tests.

It recognises distilabel's Evol-Instruct mutation prompts and returns a tagged
"evolved" instruction, and answers any other prompt with a tagged answer that
echoes the question. Tests can then assert that every exported pair is aligned:
the kept instruction is the evolved one, and the output answers that instruction.

Kept in its own importable module because distilabel pickles steps (and their
LLMs) into worker processes.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

from distilabel.models.llms.base import LLM

_GIVEN_PROMPT_RE = re.compile(
    r"#(?:The )?Given Prompt#:\s*\n(?P<prompt>.*?)\n#(?:Rewritten|Created) Prompt#:",
    re.DOTALL,
)

EVOLVED_PREFIX = "EVOLVED"
ANSWER_PREFIX = "ANSWER"


def _tag(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def evolve(prompt: str) -> str:
    """The instruction FakeLLM produces when asked to evolve *prompt*."""
    return f"{EVOLVED_PREFIX}[{_tag(prompt)}] {prompt}"


def answer(instruction: str) -> str:
    """The answer FakeLLM produces for *instruction* (always > 100 chars)."""
    return (
        f"{ANSWER_PREFIX}[{_tag(instruction)}] This is a detailed explanation that "
        "covers the concept thoroughly, gives an example, and notes common pitfalls "
        "so the response is long enough to pass the length filter."
    )


class FakeLLM(LLM):
    """Offline, deterministic distilabel LLM.

    Args:
        calls_file: optional path; one line is appended per `generate` call so a
            test can count how much work a (resumed) run actually did.
        crash_after_calls: if set, hard-exit the worker process (no cleanup, like
            a power loss or OOM kill) once this many calls were recorded in
            `calls_file`.
        delay_seconds: sleep this long per call, so a test can interrupt a run
            while it is in progress.
    """

    calls_file: str | None = None
    crash_after_calls: int | None = None
    delay_seconds: float = 0.0

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def generate(  # type: ignore[override]
        self,
        inputs: list[Any],
        num_generations: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        self._record_call()
        outputs = []
        for conversation in inputs:
            prompt = conversation[-1]["content"]
            match = _GIVEN_PROMPT_RE.search(prompt)
            text = evolve(match.group("prompt")) if match else answer(prompt)
            outputs.append({
                "generations": [text] * num_generations,
                "statistics": {"input_tokens": [0] * num_generations,
                               "output_tokens": [0] * num_generations},
            })
        return outputs

    def _record_call(self) -> None:
        if not self.calls_file:
            return
        path = Path(self.calls_file)
        with path.open("a", encoding="utf-8") as fh:
            fh.write("call\n")
        if self.crash_after_calls is not None:
            calls = len(path.read_text(encoding="utf-8").splitlines())
            if calls >= self.crash_after_calls:
                os._exit(1)
