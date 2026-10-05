"""
Prompts and structured-output schemas for grounded dataset generation.

Kept in one module so the tests' fake OpenAI server, the generator and the
docs all agree on the exact wording.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

QuestionType = Literal["factual", "conceptual", "procedural", "comparative", "multi-hop"]
QUESTION_TYPES: tuple[QuestionType, ...] = ("factual", "conceptual", "procedural", "comparative", "multi-hop")

UNANSWERABLE = "UNANSWERABLE"


# ── schemas ──────────────────────────────────────────────────────────────────

class GeneratedQuestion(BaseModel):
    type: QuestionType
    question: str


class QuestionSet(BaseModel):
    questions: list[GeneratedQuestion]


class EvolvedQuestion(BaseModel):
    question: str
    answerable_from_source: bool


class JudgeScores(BaseModel):
    faithfulness: int
    helpfulness: int
    correctness: int
    reason: str


# ── question generation ──────────────────────────────────────────────────────

QUESTION_SYSTEM = (
    "You write questions for a training dataset that teaches a language model the "
    "knowledge in a source document. Every question must be fully answerable from "
    "the source passage alone, must make sense without seeing the passage (never say "
    "'the passage', 'the text', 'the author' or 'according to'), and must ask about "
    "substance, not formatting or page layout. Questions must be different from each "
    "other and from any listed as already asked."
)


def question_messages(
    passage: str,
    k: int,
    types: tuple[QuestionType, ...],
    next_passage: str | None = None,
    avoid: list[str] | None = None,
) -> list[dict[str, str]]:
    parts = [f"Source passage A:\n<<<\n{passage}\n>>>"]
    if next_passage and "multi-hop" in types:
        parts.append(f"Source passage B (the text that follows A):\n<<<\n{next_passage}\n>>>")
    wanted = ", ".join(t for t in types if t != "multi-hop" or next_passage)
    parts.append(
        f"Write {k} questions. Spread them across these types: {wanted}. "
        + ("A multi-hop question must need facts from both passage A and passage B. " if next_passage else "")
        + "Return JSON: {\"questions\": [{\"type\": ..., \"question\": ...}]}."
    )
    if avoid:
        parts.append("Already asked (do not repeat or rephrase these):\n" + "\n".join(f"- {q}" for q in avoid))
    return [{"role": "system", "content": QUESTION_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


# ── evolution (Research mode) ────────────────────────────────────────────────

EVOLVE_SYSTEM = (
    "You make a question harder so that answering it takes more reasoning (comparison, "
    "cause and effect, a worked example, or combining two facts), while it stays fully "
    "answerable from the source passage. Keep it one clear question with no preamble. "
    "Set answerable_from_source to false if the harder question cannot be answered from "
    "the passage."
)


def evolve_messages(question: str, passage: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EVOLVE_SYSTEM},
        {"role": "user", "content": f"Source passage:\n<<<\n{passage}\n>>>\n\nQuestion: {question}\n\n"
                                    "Return JSON: {\"question\": ..., \"answerable_from_source\": true|false}."},
    ]


# ── answering ────────────────────────────────────────────────────────────────

ANSWER_SYSTEM = (
    "You are an expert writing the reference answer for a training dataset. Use only "
    "facts stated in the source passage. Write a complete, well-organised answer that "
    "stands on its own: do not mention 'the passage', 'the text', 'the context' or 'the "
    "document', and do not add a preamble or closing offer. If the passage does not "
    f"contain the answer, reply with exactly {UNANSWERABLE}."
)


def answer_messages(question: str, passage: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Source passage:\n<<<\n{passage}\n>>>\n\nQuestion: {question}"},
    ]


# ── judging ──────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You grade one question/answer pair from a synthetic training dataset against the "
    "source passage it was written from. Score each criterion from 1 (very poor) to 5 "
    "(excellent):\n"
    "- faithfulness: every claim in the answer is supported by the passage (5 = fully "
    "supported, 1 = contradicts it or invents facts);\n"
    "- helpfulness: the answer fully addresses the question, clearly, without filler, "
    "refusals or references to 'the passage';\n"
    "- correctness: the answer is accurate and the question is sensible on its own.\n"
    "Be strict. Give a one-sentence reason."
)


def judge_messages(question: str, answer: str, passage: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Source passage:\n<<<\n{passage}\n>>>\n\nQuestion: {question}\n\n"
                                    f"Answer:\n<<<\n{answer}\n>>>\n\n"
                                    "Return JSON: {\"faithfulness\": 1-5, \"helpfulness\": 1-5, "
                                    "\"correctness\": 1-5, \"reason\": ...}."},
    ]
