"""
Pattern-based filters for generated questions and answers (Phase 2.4).

These catch the failure modes seen in real runs before spending judge calls:
rewrite preambles in questions ("Sure! Here's a more complex version of the
prompt: ..."), non-answers ("Understood! If you have any specific questions,
feel free to ask"), refusals ("As an AI ..."), and answers that talk about
"the passage" instead of standing on their own.
"""
from __future__ import annotations

import re

from pipeline.prompts import UNANSWERABLE

MIN_QUESTION_CHARS = 12
MIN_ANSWER_CHARS = 40

# A leading chatty line or clause before the real question.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:(?:sure|certainly|of course|okay|ok|absolutely|great)[!.,]?\s*)?"
    r"(?:here(?:'s| is| are)\b[^\n:]*[:\n]|(?:the )?(?:rewritten|revised|new|harder|more complex)"
    r"(?: version of the)? (?:prompt|question)\s*[:\n])",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^\s*(?:#+\s*[^\n]*\n|\*\*[^*\n]+\*\*\s*\n)")
_LABEL_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*]\s*|(?:question|q)\s*\d*\s*[:.-]\s*)", re.IGNORECASE)
_PROMPT_LEAK_RE = re.compile(r"#?\s*(?:the )?(?:given|rewritten|created) prompt#?", re.IGNORECASE)

_REFUSAL_RE = re.compile(
    r"\b(?:as an ai\b|as a language model|i(?:'m| am) (?:sorry|unable|not able)|i can(?:no|')t (?:help|answer|provide|assist)"
    r"|i do not have (?:access|enough information)|i don't have (?:access|enough information))",
    re.IGNORECASE,
)
_DEFLECTION_RE = re.compile(
    r"^\s*(?:understood|sure|certainly|of course|okay|got it)[!.,]|feel free to ask|let me know if you"
    r"|if you have any (?:other |specific |further )?questions|i hope this helps|happy to help",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"\b(?:according to|based on|as (?:stated|described|mentioned) in|in) the (?:passage|text|context|document|source|excerpt)\b"
    r"|\bthe (?:passage|text|context|excerpt) (?:states|says|mentions|describes|does not)\b",
    re.IGNORECASE,
)


def clean_question(text: str) -> str | None:
    """Strip preambles, headings and list labels; None if no usable question remains."""
    q = text.strip().strip('"').strip()
    for _ in range(3):  # preambles and headings can stack
        before = q
        q = _PREAMBLE_RE.sub("", q, count=1).lstrip()
        q = _HEADING_RE.sub("", q, count=1).lstrip()
        if q == before:
            break
    q = _LABEL_RE.sub("", q, count=1).strip().strip('"').strip()
    if len(q) < MIN_QUESTION_CHARS or _PROMPT_LEAK_RE.search(q):
        return None
    return q


def answer_problem(answer: str) -> str | None:
    """Why an answer must be dropped, or None if it passes."""
    a = answer.strip()
    if not a:
        return "empty"
    if UNANSWERABLE in a[:40].upper():
        return "unanswerable"
    if len(a) < MIN_ANSWER_CHARS:
        return "too short"
    if _REFUSAL_RE.search(a[:300]):
        return "refusal"
    if _DEFLECTION_RE.search(a[:200]) and len(a) < 400:
        return "non-answer"
    if _META_RE.search(a):
        return "mentions the source"
    return None
