"""
Grounded synthetic dataset generation (Phase 2.2–2.4, 2.8).

For each source chunk the teacher writes k diverse questions (factual,
conceptual, procedural, comparative, multi-hop across adjacent chunks), and
answers each one with the source text in context. Optionally (Research mode)
each question is first made harder, with a check that it is still answerable
from the source. Every candidate pair then passes:

    question cleaning → answer filters (refusals, non-answers, "the passage …")
    → LLM judge (faithfulness / helpfulness / correctness ≥ threshold)
    → MinHash near-duplicate check

Generation runs in rounds until `target` pairs are accepted or every chunk
stops producing new questions, so "dataset size" means what it says.
"""
from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from openai import AuthenticationError, NotFoundError, PermissionDeniedError

from engine import ChatClient
from pipeline.dedup import Deduplicator, SemanticDeduplicator, normalise
from pipeline.filters import answer_problem, clean_question
from pipeline.prompts import (
    QUESTION_TYPES,
    EvolvedQuestion,
    JudgeScores,
    QuestionSet,
    QuestionType,
    answer_messages,
    evolve_messages,
    judge_messages,
    question_messages,
)
from pipeline.records import Record

logger = structlog.get_logger(__name__)

MAX_QUESTIONS_PER_CHUNK = 8
# Rough number of pairs a ~1,600-character chunk supports before questions
# start repeating; used for the yield estimate shown before a run.
PAIRS_PER_CHUNK_ESTIMATE = 6
MAX_ROUNDS = 4
AVOID_LIST_SIZE = 40
MAX_REJECTED_KEPT = 1000
ERRORS_LOGGED = 5  # failed requests logged in full; later ones are only counted

# Errors that no retry or other chunk can fix: stop the whole run.
FATAL_ERRORS = (AuthenticationError, PermissionDeniedError, NotFoundError)


@dataclass
class SynthStats:
    chunks: int = 0
    rounds: int = 0
    questions_generated: int = 0
    questions_dropped: int = 0  # malformed or not self-contained
    questions_repeated: int = 0
    evolved: int = 0
    evolve_unanswerable: int = 0
    answers: int = 0
    answers_filtered: Counter[str] = field(default_factory=Counter)
    judged: int = 0
    judge_rejected: int = 0
    duplicates: int = 0
    semantic_duplicates: int = 0
    errors: int = 0
    last_error: str | None = None
    accepted: int = 0
    exhausted_chunks: int = 0

    # Rejected pairs with the reason in meta["rejected"], for inspection (capped).
    rejected: list[Record] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("answers_filtered", "rejected")}
        d["answers_filtered"] = dict(self.answers_filtered)
        return d


@dataclass(frozen=True)
class SynthSettings:
    target: int
    evolve: bool = False
    judge_threshold: int = 4
    dedup_threshold: float | None = 0.85  # None: keep near-duplicates
    semantic_threshold: float = 0.92  # cosine; used only when an embedder is given
    question_types: tuple[QuestionType, ...] = QUESTION_TYPES
    max_rounds: int = MAX_ROUNDS


def judge_passes(scores: JudgeScores, threshold: int) -> bool:
    return min(_clamp(scores.faithfulness), _clamp(scores.helpfulness), _clamp(scores.correctness)) >= threshold


def _clamp(score: int) -> int:
    return score if 1 <= score <= 5 else 1


def questions_per_chunk(remaining: int, active_chunks: int, acceptance: float) -> int:
    """How many questions to ask each chunk this round."""
    if active_chunks <= 0 or remaining <= 0:
        return 0
    wanted = remaining / max(acceptance, 0.2) / active_chunks
    return max(1, min(MAX_QUESTIONS_PER_CHUNK, math.ceil(wanted)))


class _Run:
    def __init__(
        self,
        chunks: list[str],
        teachers: list[ChatClient],
        judge: ChatClient | None,
        settings: SynthSettings,
        progress: Callable[[float], None] | None,
        embedder: ChatClient | None = None,
    ) -> None:
        self.chunks = chunks
        self.teachers = teachers
        self.judge = judge
        self.s = settings
        self.progress = progress
        self.stats = SynthStats(chunks=len(chunks))
        self.dedup = Deduplicator(settings.dedup_threshold) if settings.dedup_threshold else None
        self.embedder = embedder
        self.semantic = SemanticDeduplicator(settings.semantic_threshold) if embedder else None
        self.asked: dict[int, list[str]] = {i: [] for i in range(len(chunks))}
        self.exhausted: set[int] = set()
        self.empty_rounds: dict[int, int] = dict.fromkeys(range(len(chunks)), 0)
        self.accepted: list[Record] = []
        # Pairs that passed filters and judge this round, keyed for a stable order.
        self.candidates: list[tuple[tuple[int, int, int], Record]] = []

    # ── one chunk in one round ───────────────────────────────────────────
    async def chunk_round(self, rnd: int, i: int, k: int) -> None:
        teacher = self.teachers[i % len(self.teachers)]
        nxt = self.chunks[i + 1] if i + 1 < len(self.chunks) else None
        try:
            qset = await teacher.chat_json(
                question_messages(self.chunks[i], k, self.s.question_types, nxt, self.asked[i][-AVOID_LIST_SIZE:]),
                QuestionSet,
            )
        except FATAL_ERRORS:
            raise
        except Exception as exc:
            self._error("questions", exc)
            return

        seen = {normalise(q) for q in self.asked[i]}
        fresh: list[tuple[QuestionType, str]] = []
        repeated = 0
        for gq in qset.questions:
            self.stats.questions_generated += 1
            q = clean_question(gq.question)
            if q is None:
                self.stats.questions_dropped += 1
                continue
            if normalise(q) in seen:
                repeated += 1
                continue
            seen.add(normalise(q))
            self.asked[i].append(q)
            fresh.append((gq.type, q))
        self.stats.questions_repeated += repeated
        if not fresh:
            # Only repeats left: the chunk is used up. Malformed questions get one more round.
            self.empty_rounds[i] += 1
            if (repeated and repeated == len(qset.questions)) or self.empty_rounds[i] >= 2:
                self.exhausted.add(i)
            return
        self.empty_rounds[i] = 0

        await asyncio.gather(*(
            self.pair(rnd, i, n, qtype, q, teacher, nxt) for n, (qtype, q) in enumerate(fresh)
        ))

    # ── one question → one accepted (or rejected) pair ──────────────────
    async def pair(
        self, rnd: int, i: int, n: int, qtype: QuestionType, question: str,
        teacher: ChatClient, nxt: str | None,
    ) -> None:
        passage = self.chunks[i] + (f"\n\n{nxt}" if qtype == "multi-hop" and nxt else "")
        try:
            evolved = False
            if self.s.evolve:
                ev = await teacher.chat_json(evolve_messages(question, passage), EvolvedQuestion)
                self.stats.evolved += 1
                harder = clean_question(ev.question) if ev.answerable_from_source else None
                if harder is None:
                    self.stats.evolve_unanswerable += 1
                else:
                    question, evolved = harder, True

            answer = await teacher.chat(answer_messages(question, passage))
            self.stats.answers += 1
            problem = answer_problem(answer)
            if problem:
                self.stats.answers_filtered[problem] += 1
                self._reject(rnd, i, qtype, question, answer, teacher, f"filter: {problem}")
                return

            scores: JudgeScores | None = None
            if self.judge is not None:
                scores = await self.judge.chat_json(judge_messages(question, answer, passage), JudgeScores, temperature=0.0)
                self.stats.judged += 1
                if not judge_passes(scores, self.s.judge_threshold):
                    self.stats.judge_rejected += 1
                    self._reject(rnd, i, qtype, question, answer, teacher, "judge", scores)
                    return
        except FATAL_ERRORS:
            raise
        except Exception as exc:  # StructuredOutputError, timeouts after retries, 4xx/5xx
            self._error("pair", exc)
            return

        rec = Record(
            instruction=question,
            output=answer,
            meta={
                "chunk": i,
                "type": qtype,
                "round": rnd,
                "evolved": evolved,
                "model": teacher.settings.model,
                "judge": scores.model_dump() if scores else None,
            },
        )
        self.candidates.append(((rnd, i, n), rec))

    def _error(self, step: str, exc: Exception) -> None:
        """Count a failed request; log the first few in full and remember the last one."""
        self.stats.errors += 1
        self.stats.last_error = f"{type(exc).__name__}: {exc}"[:500]
        if self.stats.errors <= ERRORS_LOGGED:
            logger.warning("Request failed", step=step, error=self.stats.last_error)
        elif self.stats.errors == ERRORS_LOGGED + 1:
            logger.warning("Further request failures are only counted", logged=ERRORS_LOGGED)

    def _reject(
        self, rnd: int, i: int, qtype: QuestionType, question: str, answer: str,
        teacher: ChatClient, reason: str, scores: JudgeScores | None = None,
    ) -> None:
        if len(self.stats.rejected) >= MAX_REJECTED_KEPT or not answer.strip():
            return
        self.stats.rejected.append(Record(
            instruction=question, output=answer,
            meta={"chunk": i, "type": qtype, "round": rnd, "model": teacher.settings.model,
                  "rejected": reason, "judge": scores.model_dump() if scores else None},
        ))

    async def accept_round(self) -> None:
        """Dedup and accept this round's candidates in a stable order, up to the target.

        Done after the round (not as replies arrive) so the result does not
        depend on which request happened to finish first. MinHash catches
        near-identical wording; the optional embedding pass catches paraphrases.
        """
        def text(rec: Record) -> str:
            return normalise(f"{rec.instruction} {rec.output}")

        ordered = [rec for _, rec in sorted(self.candidates, key=lambda t: t[0])]
        self.candidates.clear()
        # Only accepted records enter the indexes, so a rejected pair never blocks a later one.
        vectors: list[list[float]] | None = None
        if self.embedder is not None and self.semantic is not None and ordered:
            fresh = [r for r in ordered if not (self.dedup and self.dedup.is_duplicate(text(r)))]
            self.stats.duplicates += len(ordered) - len(fresh)
            # Embed only as many as could still be accepted, with headroom for rejects.
            ordered = fresh[:2 * (self.s.target - len(self.accepted))]
            vectors = await self.embedder.embed([f"{r.instruction}\n{r.output}" for r in ordered])
        for idx, rec in enumerate(ordered):
            if len(self.accepted) >= self.s.target:
                break
            key = text(rec)
            if self.dedup and self.dedup.is_duplicate(key):
                self.stats.duplicates += 1
                continue
            if vectors is not None and self.semantic is not None and not self.semantic.add_if_new(vectors[idx]):
                self.stats.semantic_duplicates += 1
                continue
            if self.dedup:
                self.dedup.add_if_new(key)
            self.accepted.append(rec)
        self.stats.accepted = len(self.accepted)

    # ── rounds ───────────────────────────────────────────────────────────
    async def run(self) -> list[Record]:
        acceptance = 0.7  # first-round guess; later rounds use what was observed
        for rnd in range(1, self.s.max_rounds + 1):
            active = [i for i in range(len(self.chunks)) if i not in self.exhausted]
            remaining = self.s.target - len(self.accepted)
            k = questions_per_chunk(remaining, len(active), acceptance)
            if k == 0:
                break
            self.stats.rounds = rnd
            before_q, before_acc = self.stats.questions_generated, len(self.accepted)

            tasks = [asyncio.create_task(self.chunk_round(rnd, i, k)) for i in active]
            for finished, fut in enumerate(asyncio.as_completed(tasks), 1):
                try:
                    await fut
                except BaseException:
                    for t in tasks:
                        t.cancel()
                    raise
                self._report(rnd, finished / len(tasks))

            await self.accept_round()
            if len(self.accepted) >= self.s.target:
                break
            produced = self.stats.questions_generated - before_q
            if produced:
                acceptance = max((len(self.accepted) - before_acc) / produced, 0.05)
        self.stats.exhausted_chunks = len(self.exhausted)
        if self.progress:
            self.progress(1.0)
        return self.accepted

    def _report(self, rnd: int, round_fraction: float) -> None:
        if not self.progress:
            return
        by_target = len(self.accepted) / self.s.target if self.s.target else 1.0
        by_rounds = (rnd - 1 + round_fraction) / self.s.max_rounds
        self.progress(min(0.99, max(by_target, by_rounds)))


async def synthesize(
    chunks: list[str],
    teachers: list[ChatClient],
    settings: SynthSettings,
    judge: ChatClient | None = None,
    progress: Callable[[float], None] | None = None,
    embedder: ChatClient | None = None,
) -> tuple[list[Record], SynthStats]:
    """Generate up to `settings.target` accepted pairs from *chunks*."""
    if not chunks:
        raise ValueError("No source chunks to generate from.")
    if not teachers:
        raise ValueError("At least one teacher model is required.")
    run = _Run(chunks, teachers, judge, settings, progress, embedder)
    records = await run.run()
    return records, run.stats
