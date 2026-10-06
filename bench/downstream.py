"""
Downstream evaluation: does training on a Brainbrew dataset make a model better?

Three steps, each a subcommand (the full procedure is in docs/downstream-eval.md):

  testset   Write a held-out exam from the source documents: closed-book
            questions with short reference answers, in a different style from
            the training questions, with near-copies of any training question
            removed.
  train     Train a LoRA adapter on a run's records (wraps training.lora_trainer).
  evaluate  Ask every model the exam questions without the documents and have a
            judge grade each answer against the reference (1-5). The first model
            is the baseline; every other model is compared with it question by
            question, with a bootstrap 95% confidence interval.

    python bench/downstream.py testset docs/*.pdf --train runs/<id>/records.jsonl \\
        --model gpt-4o-mini --size 100 --out testset.jsonl
    python bench/downstream.py train runs/<id>/records.jsonl \\
        --base-model Qwen/Qwen2.5-1.5B-Instruct --out adapter --epochs 3
    python bench/downstream.py evaluate testset.jsonl \\
        --student-base-url http://127.0.0.1:8000/v1 \\
        --models Qwen/Qwen2.5-1.5B-Instruct brainbrew --judge-model gpt-4o-mini

API keys come from the environment only. OPENAI_API_KEY is sent only to
OpenAI itself (when no base URL is given); a custom endpoint gets
ENDPOINT_API_KEY, if set. A local vLLM server needs no key.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python bench/downstream.py` from anywhere
    sys.path.insert(0, str(ROOT))

from engine import ChatClient, EndpointSettings  # noqa: E402
from pipeline.dedup import Deduplicator, normalise  # noqa: E402
from pipeline.document_loader import source_chunks  # noqa: E402
from pipeline.filters import clean_question  # noqa: E402
from pipeline.records import read_records  # noqa: E402
from pipeline.service import read_documents  # noqa: E402

ClientFactory = Callable[[EndpointSettings], ChatClient]

OVERLAP_THRESHOLD = 0.5   # MinHash similarity at which a test question counts as a training question
PER_CHUNK = 2             # exam questions asked per passage and round
MAX_ROUNDS = 3
PASS_SCORE = 4            # a grade of 4 or 5 counts as correct
BOOTSTRAP_SAMPLES = 2000


def _make_client(settings: EndpointSettings) -> ChatClient:
    return ChatClient(settings)


def api_key_for(base_url: str | None) -> str | None:
    """OPENAI_API_KEY only for OpenAI itself; ENDPOINT_API_KEY for any other endpoint."""
    if base_url is None:
        return os.getenv("OPENAI_API_KEY") or None
    return os.getenv("ENDPOINT_API_KEY") or None


# ── prompts ──────────────────────────────────────────────────────────────────

class ExamItem(BaseModel):
    question: str
    answer: str


class ExamSet(BaseModel):
    items: list[ExamItem]


class Grade(BaseModel):
    score: int
    reason: str


EXAM_SYSTEM = (
    "You write a closed-book exam. The student studied a document earlier and must now "
    "answer from memory, without seeing it. Each question asks for one specific fact, "
    "reason or step stated in the excerpt, names what it is about, and makes sense on "
    "its own (never 'the passage', 'the text', 'the author', 'according to'). Avoid "
    "questions anyone could answer from general knowledge, and questions about wording "
    "or layout. Give each a reference answer of one or two sentences, taken from the excerpt."
)


def exam_messages(passage: str, n: int, avoid: Sequence[str]) -> list[dict[str, str]]:
    asked = "\n".join(f"- {q}" for q in avoid[-20:]) or "(none)"
    return [
        {"role": "system", "content": EXAM_SYSTEM},
        {"role": "user", "content": f"Excerpt:\n<<<\n{passage}\n>>>\n\nAlready on the exam:\n{asked}\n\n"
                                    f"Write {n} new questions. Return JSON: "
                                    "{\"items\": [{\"question\": ..., \"answer\": ...}]}."},
    ]


STUDENT_SYSTEM = (
    "Answer the question directly and concisely, in at most three sentences. "
    "If you do not know, say that you do not know."
)

GRADER_SYSTEM = (
    "You grade a closed-book exam answer against a reference answer and the source "
    "excerpt it was written from. Score correctness only, not style or length:\n"
    "5 = fully correct and consistent with the reference;\n"
    "4 = correct, with a minor omission or imprecision;\n"
    "3 = partly correct, missing or blurring something important;\n"
    "2 = mostly wrong, with a small correct element;\n"
    "1 = wrong, contradicts the source, or does not answer (including 'I don't know')."
)


def grade_messages(question: str, reference: str, passage: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": GRADER_SYSTEM},
        {"role": "user", "content": f"Source excerpt:\n<<<\n{passage}\n>>>\n\nQuestion: {question}\n"
                                    f"Reference answer: {reference}\n\nStudent answer:\n<<<\n{answer}\n>>>\n\n"
                                    "Return JSON: {\"score\": 1-5, \"reason\": ...}."},
    ]


# ── testset ──────────────────────────────────────────────────────────────────

@dataclass
class ExamQuestion:
    id: int
    question: str
    reference: str
    passage: str
    chunk: int


def training_index(records_path: Path | None, threshold: float) -> Deduplicator:
    index = Deduplicator(threshold)
    if records_path is not None:
        for rec in read_records(records_path):
            index.add_if_new(normalise(rec.instruction))
    return index


async def build_testset(
    chunks: list[str],
    client: ChatClient,
    size: int,
    train_index: Deduplicator,
    seed: int,
    threshold: float = OVERLAP_THRESHOLD,
) -> tuple[list[ExamQuestion], dict[str, int]]:
    """Exam questions from passages in a seeded random order, until *size* survive."""
    order = list(range(len(chunks)))
    random.Random(seed).shuffle(order)
    kept: list[ExamQuestion] = []
    own = Deduplicator(threshold)
    stats = {"proposed": 0, "malformed": 0, "overlaps_training": 0, "duplicate": 0, "errors": 0}

    async def ask(i: int, avoid: list[str]) -> tuple[int, ExamSet | None]:
        try:
            return i, await client.chat_json(exam_messages(chunks[i], PER_CHUNK, avoid), ExamSet)
        except Exception:
            stats["errors"] += 1
            return i, None

    for _ in range(MAX_ROUNDS):
        # Enough passages for the remaining questions, plus a margin for the filters.
        need = size - len(kept)
        batch = order[: max(1, -(-need * 3 // (2 * PER_CHUNK)))]
        order = order[len(batch):] + batch  # later rounds move on to other passages first
        results = await asyncio.gather(*(ask(i, [t.question for t in kept]) for i in batch))
        for i, exam in results:  # gather keeps the batch order, so the result is deterministic
            for item in exam.items if exam else []:
                stats["proposed"] += 1
                question = clean_question(item.question)
                if question is None or not item.answer.strip():
                    stats["malformed"] += 1
                    continue
                key = normalise(question)
                if train_index.is_duplicate(key):
                    stats["overlaps_training"] += 1
                    continue
                if not own.add_if_new(key):
                    stats["duplicate"] += 1
                    continue
                kept.append(ExamQuestion(len(kept), question, item.answer.strip(), chunks[i], i))
                if len(kept) >= size:
                    return kept, stats
    return kept, stats


def cmd_testset(args: argparse.Namespace, make_client: ClientFactory = _make_client) -> int:
    text, errors = read_documents((Path(p).name, Path(p).read_bytes()) for p in args.documents)
    for err in errors:
        print(err, file=sys.stderr)
    chunks = source_chunks(text, args.semantic) if text.strip() else []
    if not chunks:
        print("No text could be extracted from the documents.", file=sys.stderr)
        return 2
    settings = EndpointSettings(model=args.model, base_url=args.base_url, api_key=api_key_for(args.base_url),
                                temperature=0.4, max_tokens=1024, concurrency=args.concurrency,
                                timeout_s=float(args.timeout), seed=args.seed)

    async def run() -> tuple[list[ExamQuestion], dict[str, int]]:
        client = make_client(settings)
        try:
            return await build_testset(chunks, client, args.size,
                                       training_index(args.train, args.overlap_threshold), args.seed,
                                       args.overlap_threshold)
        finally:
            await client.close()

    items, stats = asyncio.run(run())
    with open(args.out, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    print(f"{len(items)}/{args.size} questions from {len({t.chunk for t in items})} of {len(chunks)} passages "
          f"-> {args.out}  ({', '.join(f'{k} {v}' for k, v in stats.items())})")
    if args.train is None:
        print("Warning: no --train records given, so overlap with the training set was not checked.",
              file=sys.stderr)
    return 0 if len(items) >= args.size else 1


# ── train ────────────────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace) -> int:
    from training.lora_trainer import train_lora

    start = time.perf_counter()
    out = train_lora(args.records, args.base_model, args.out, args.rank,
                     num_train_epochs=args.epochs, max_length=args.max_length)
    print(f"Adapter saved to {out} in {time.perf_counter() - start:.0f}s. Serve it next to the base model with:\n"
          f"  vllm serve {args.base_model} --enable-lora --lora-modules brainbrew={out} --max-lora-rank {args.rank}")
    return 0


# ── evaluate ─────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    model: str
    mean_score: float = 0.0
    accuracy: float = 0.0       # share of answers graded >= PASS_SCORE
    graded: int = 0
    errors: int = 0
    seconds: float = 0.0
    vs_baseline: dict[str, Any] = field(default_factory=dict)


def load_testset(path: Path) -> list[ExamQuestion]:
    with open(path, encoding="utf-8") as fh:
        return [ExamQuestion(**json.loads(line)) for line in fh if line.strip()]


def paired_bootstrap(diffs: Sequence[float], seed: int, samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    """95% confidence interval for the mean of *diffs* (percentile bootstrap)."""
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def compare(base: dict[int, int], other: dict[int, int], seed: int) -> dict[str, Any]:
    shared = sorted(base.keys() & other.keys())
    diffs = [other[i] - base[i] for i in shared]
    low, high = paired_bootstrap(diffs, seed)
    return {
        "questions": len(shared),
        "mean_gain": round(sum(diffs) / len(diffs), 3) if diffs else 0.0,
        "ci95": [round(low, 3), round(high, 3)],
        "significant": low > 0 or high < 0,
        "wins": sum(d > 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
    }


async def answer_and_grade(
    items: list[ExamQuestion], student: ChatClient, judge: ChatClient,
) -> tuple[dict[int, int], dict[int, str], int]:
    """Scores and answers by question id, plus the number of failed questions."""

    async def one(item: ExamQuestion) -> tuple[int, str, int | None]:
        try:
            answer = await student.chat([{"role": "system", "content": STUDENT_SYSTEM},
                                         {"role": "user", "content": item.question}])
            grade = await judge.chat_json(grade_messages(item.question, item.reference, item.passage, answer),
                                          Grade, temperature=0.0)
        except Exception as exc:
            return item.id, f"[error] {type(exc).__name__}: {exc}"[:300], None
        return item.id, answer, min(5, max(1, grade.score))

    results = await asyncio.gather(*(one(i) for i in items))
    scores = {qid: s for qid, _, s in results if s is not None}
    answers = {qid: a for qid, a, _ in results}
    return scores, answers, sum(s is None for _, _, s in results)


def cmd_evaluate(args: argparse.Namespace, make_client: ClientFactory = _make_client) -> int:
    items = load_testset(args.testset)
    if not items:
        print(f"{args.testset} holds no questions.", file=sys.stderr)
        return 2
    judge_settings = EndpointSettings(model=args.judge_model, base_url=args.judge_base_url,
                                      api_key=api_key_for(args.judge_base_url), temperature=0.0,
                                      max_tokens=300, concurrency=args.concurrency, timeout_s=float(args.timeout))

    async def run_model(model: str) -> tuple[dict[int, int], dict[int, str], int]:
        student = make_client(EndpointSettings(
            model=model, base_url=args.student_base_url, api_key=api_key_for(args.student_base_url),
            temperature=0.0, max_tokens=args.max_tokens, concurrency=args.concurrency,
            timeout_s=float(args.timeout), seed=args.seed))
        judge = make_client(judge_settings)
        try:
            return await answer_and_grade(items, student, judge)
        finally:
            await student.close()
            await judge.close()

    results: list[ModelResult] = []
    all_scores: list[dict[int, int]] = []
    all_answers: list[dict[int, str]] = []
    for model in args.models:
        print(f"→ {model}", flush=True)
        start = time.perf_counter()
        scores, answers, errors = asyncio.run(run_model(model))
        res = ModelResult(model=model, graded=len(scores), errors=errors,
                          seconds=round(time.perf_counter() - start, 1))
        if scores:
            res.mean_score = round(sum(scores.values()) / len(scores), 3)
            res.accuracy = round(sum(s >= PASS_SCORE for s in scores.values()) / len(scores), 3)
        if results:
            res.vs_baseline = compare(all_scores[0], scores, args.seed)
        print(f"  mean {res.mean_score:.2f}, correct {res.accuracy:.0%}, {res.graded} graded, "
              f"{res.errors} failed, {res.seconds:.0f}s", flush=True)
        results.append(res)
        all_scores.append(scores)
        all_answers.append(answers)

    report = {
        "testset": str(args.testset),
        "questions": len(items),
        "baseline": args.models[0],
        "judge_model": args.judge_model,
        "student_base_url": args.student_base_url,
        "models": [asdict(r) for r in results],
        "answers": [
            {"id": item.id, "question": item.question, "reference": item.reference,
             **{m: {"answer": a.get(item.id), "score": s.get(item.id)}
                for m, a, s in zip(args.models, all_answers, all_scores, strict=True)}}
            for item in items
        ],
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_markdown(report))
    return 0 if all(r.errors < len(items) for r in results) else 1


def summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"## Downstream evaluation: {report['questions']} held-out questions, closed book",
        "",
        f"Judge `{report['judge_model']}`; a grade of {PASS_SCORE} or 5 counts as correct.",
        "",
        "| Model | Mean grade (1-5) | Correct | Gain vs baseline [95% CI] | Wins / ties / losses |",
        "|---|---|---|---|---|",
    ]
    for m in report["models"]:
        vs = m["vs_baseline"]
        gain = (f"{vs['mean_gain']:+.2f} [{vs['ci95'][0]:+.2f}, {vs['ci95'][1]:+.2f}]"
                + (" *" if vs["significant"] else "")) if vs else "baseline"
        wtl = f"{vs['wins']} / {vs['ties']} / {vs['losses']}" if vs else "–"
        lines.append(f"| `{m['model']}` | {m['mean_score']:.2f} | {m['accuracy']:.0%} | {gain} | {wtl} |")
    lines += ["", "\\* the 95% interval excludes zero.", ""]
    return "\n".join(lines)


# ── command line ─────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("testset", help="Write a held-out exam from the source documents.")
    t.add_argument("documents", nargs="+", type=Path, help="The documents the training set was made from (PDF/TXT)")
    t.add_argument("--train", type=Path, default=None,
                   help="The training run's records.jsonl; exam questions too close to a training question are dropped")
    t.add_argument("--model", required=True, help="Model that writes the exam (ideally a strong one)")
    t.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (default: OpenAI)")
    t.add_argument("--size", type=int, default=100)
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--semantic", action="store_true", help="Chunk the way a run with semantic chunking did")
    t.add_argument("--overlap-threshold", type=float, default=OVERLAP_THRESHOLD)
    t.add_argument("--concurrency", type=int, default=8)
    t.add_argument("--timeout", type=int, default=120)
    t.add_argument("--out", type=Path, default=Path("testset.jsonl"))

    r = sub.add_parser("train", help="Train a LoRA adapter on a run's records.")
    r.add_argument("records", type=Path, help="runs/<id>/records.jsonl")
    r.add_argument("--base-model", required=True)
    r.add_argument("--out", type=Path, default=Path("adapter"))
    r.add_argument("--epochs", type=float, default=3.0)
    r.add_argument("--rank", type=int, default=16)
    r.add_argument("--max-length", type=int, default=2048)

    e = sub.add_parser("evaluate", help="Grade models on the exam, the first one being the baseline.")
    e.add_argument("testset", type=Path)
    e.add_argument("--models", nargs="+", required=True,
                   help="Model names as the student endpoint knows them; the first is the baseline")
    e.add_argument("--student-base-url", required=True, help="Endpoint serving the models (e.g. vllm with --enable-lora)")
    e.add_argument("--judge-model", required=True)
    e.add_argument("--judge-base-url", default=None, help="Judge endpoint (default: OpenAI)")
    e.add_argument("--max-tokens", type=int, default=300)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--concurrency", type=int, default=8)
    e.add_argument("--timeout", type=int, default=120)
    e.add_argument("--out", type=Path, default=Path("downstream-report.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None, make_client: ClientFactory = _make_client) -> int:
    args = parse_args(argv)
    if args.command == "testset":
        return cmd_testset(args, make_client)
    if args.command == "train":
        return cmd_train(args)
    return cmd_evaluate(args, make_client)


if __name__ == "__main__":
    raise SystemExit(main())
