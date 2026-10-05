"""
Phase 2 quality benchmark: generate a dataset from each PDF in the fixed
corpus (tests/fixtures/bench/) and check the generation-quality gate.

    OPENAI_API_KEY=... python bench/run_bench.py --model gpt-4o-mini
    python bench/run_bench.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3-4B-Instruct-2507

Gate (per document):
  * judge faithfulness >= 4.0 on average, from an independent judge pass over
    the final records (not the in-pipeline scores, which only kept pairs >= the
    threshold);
  * near-duplicate rate < 2 % (questions that are MinHash near-duplicates of an
    earlier question, a stricter check than the pipeline's question+answer dedup);
  * refusal rate 0 (answers the refusal/non-answer filters would flag);
  * yield within 10 % of the target.

The API key is read from OPENAI_API_KEY only, never from the command line.
Exit status is 1 when any gate fails; the JSON report has every number.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python bench/run_bench.py` from anywhere
    sys.path.insert(0, str(ROOT))

from config import DistillationConfig, QualityMode  # noqa: E402
from engine import ChatClient, EndpointSettings  # noqa: E402
from pipeline.dedup import Deduplicator, normalise  # noqa: E402
from pipeline.document_loader import read_document, source_chunks  # noqa: E402
from pipeline.filters import answer_problem  # noqa: E402
from pipeline.prompts import JudgeScores, judge_messages  # noqa: E402
from pipeline.records import Record, read_records  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "bench"
# Targets sit well below what each document can support (about 6 pairs per chunk).
TARGETS = {"elements_of_style.pdf": 15, "federalist_10.pdf": 25, "origin_of_species.pdf": 40}

MIN_FAITHFULNESS = 4.0
MAX_NEAR_DUP_RATE = 0.02
MAX_REFUSAL_RATE = 0.0
YIELD_TOLERANCE = 0.10
NEAR_DUP_THRESHOLD = 0.8
REFUSALS = {"empty", "unanswerable", "refusal", "non-answer"}


@dataclass
class DocResult:
    document: str
    target: int
    records: int = 0
    faithfulness: float = 0.0
    judged: int = 0
    judge_errors: int = 0
    near_dup_rate: float = 0.0
    refusal_rate: float = 0.0
    yield_ratio: float = 0.0
    seconds: float = 0.0
    run_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    error: str | None = None


# ── metrics ──────────────────────────────────────────────────────────────────

def near_dup_rate(records: list[Record], threshold: float = NEAR_DUP_THRESHOLD) -> float:
    if not records:
        return 0.0
    index = Deduplicator(threshold)
    dups = sum(not index.add_if_new(normalise(r.instruction)) for r in records)
    return dups / len(records)


def refusal_rate(records: list[Record]) -> float:
    if not records:
        return 0.0
    return sum(answer_problem(r.output) in REFUSALS for r in records) / len(records)


def passage_for(record: Record, chunks: list[str]) -> str:
    """The passage the pair was generated from (same rule as pipeline/synth.py)."""
    i = int(record.meta.get("chunk", 0))
    nxt = chunks[i + 1] if i + 1 < len(chunks) else None
    return chunks[i] + (f"\n\n{nxt}" if record.meta.get("type") == "multi-hop" and nxt else "")


async def rejudge(judge: ChatClient, records: list[Record], chunks: list[str]) -> tuple[list[int], int]:
    """Independent faithfulness scores; returns (scores, errors)."""

    async def one(rec: Record) -> int | None:
        try:
            s = await judge.chat_json(judge_messages(rec.instruction, rec.output, passage_for(rec, chunks)),
                                      JudgeScores, temperature=0.0)
        except Exception:
            return None
        return s.faithfulness if 1 <= s.faithfulness <= 5 else 1

    results = await asyncio.gather(*(one(r) for r in records))
    scores = [s for s in results if s is not None]
    return scores, len(results) - len(scores)


def check_gate(res: DocResult) -> list[str]:
    failures = []
    if res.faithfulness < MIN_FAITHFULNESS:
        failures.append(f"faithfulness {res.faithfulness:.2f} < {MIN_FAITHFULNESS}")
    if res.near_dup_rate >= MAX_NEAR_DUP_RATE:
        failures.append(f"near-dup rate {res.near_dup_rate:.1%} >= {MAX_NEAR_DUP_RATE:.0%}")
    if res.refusal_rate > MAX_REFUSAL_RATE:
        failures.append(f"refusal rate {res.refusal_rate:.1%} > 0")
    if abs(res.yield_ratio - 1.0) > YIELD_TOLERANCE:
        failures.append(f"yield {res.records}/{res.target} outside ±{YIELD_TOLERANCE:.0%}")
    return failures


# ── running ──────────────────────────────────────────────────────────────────

def _make_client(settings: EndpointSettings) -> ChatClient:
    return ChatClient(settings)


def bench_document(
    pdf: Path,
    target: int,
    args: argparse.Namespace,
    make_client: Callable[[EndpointSettings], ChatClient] = _make_client,
) -> DocResult:
    import orchestrator

    res = DocResult(document=pdf.name, target=target)
    text = read_document(pdf.name, pdf.read_bytes())
    cfg = DistillationConfig(
        teacher_model=args.model,
        judge_model=args.judge_model,
        base_url=args.base_url,
        api_key=os.getenv("OPENAI_API_KEY") or None,
        dataset_size=target,
        quality_mode=QualityMode(args.mode),
        concurrency=args.concurrency,
        request_timeout=args.timeout,
        max_new_tokens=args.max_tokens,
    )
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"{pdf.stem}.txt"
        src.write_text(text, encoding="utf-8")
        try:
            result = orchestrator.run_distillation(cfg, src, client_factory=make_client)
        except Exception as exc:  # report and move on to the next document
            res.error = f"{type(exc).__name__}: {exc}"
            res.failures = [f"run failed: {res.error}"]
            res.seconds = round(time.perf_counter() - start, 1)
            return res

    records = read_records(result.run.records)
    res.run_id = result.run.run_id
    res.usage = result.run.read_manifest().get("usage", {})
    res.records = len(records)
    res.yield_ratio = len(records) / target
    res.near_dup_rate = near_dup_rate(records)
    res.refusal_rate = refusal_rate(records)

    judge_settings = EndpointSettings(
        model=args.judge_model or args.model.split(",")[0].strip(), base_url=args.base_url,
        api_key=os.getenv("OPENAI_API_KEY") or None, temperature=0.0, max_tokens=512,
        timeout_s=float(args.timeout), concurrency=args.concurrency,
    )

    async def _judge() -> tuple[list[int], int]:
        judge = make_client(judge_settings)
        try:
            return await rejudge(judge, records, source_chunks(text, cfg.use_semantic_chunking))
        finally:
            await judge.close()

    scores, res.judge_errors = asyncio.run(_judge())
    res.judged = len(scores)
    res.faithfulness = round(sum(scores) / len(scores), 3) if scores else 0.0
    res.seconds = round(time.perf_counter() - start, 1)
    res.failures = check_gate(res)
    return res


def summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"## Brainbrew generation benchmark — {'PASS' if report['passed'] else 'FAIL'}",
        "",
        f"Model `{report['model']}`, judge `{report['judge_model']}`, mode `{report['mode']}`.",
        "",
        "| Document | Pairs / target | Faithfulness | Near-dup | Refusals | Time | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in report["documents"]:
        verdict = "✅" if not d["failures"] else "❌ " + "; ".join(d["failures"])
        lines.append(f"| {d['document']} | {d['records']} / {d['target']} | {d['faithfulness']:.2f} | "
                     f"{d['near_dup_rate']:.1%} | {d['refusal_rate']:.1%} | {d['seconds']:.0f}s | {verdict} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Teacher model (comma-separate for an ensemble)")
    p.add_argument("--judge-model", default=None, help="Judge model (default: the first teacher)")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or None,
                   help="OpenAI-compatible endpoint (default: OPENAI_BASE_URL, else OpenAI)")
    p.add_argument("--mode", default="balanced", choices=[m.value for m in QualityMode])
    p.add_argument("--scale", type=float, default=1.0, help="Multiply every target (e.g. 0.3 for slow local models)")
    p.add_argument("--only", nargs="*", default=None, help="Benchmark only these fixture files")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--fixtures", type=Path, default=FIXTURES)
    p.add_argument("--out", type=Path, default=Path("bench-report.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None, make_client: Callable[[EndpointSettings], ChatClient] = _make_client) -> int:
    args = parse_args(argv)
    docs = [args.fixtures / name for name in TARGETS if not args.only or name in args.only]
    results = []
    for pdf in docs:
        target = max(10, round(TARGETS[pdf.name] * args.scale))  # 10 is the config minimum
        print(f"→ {pdf.name}: target {target}", flush=True)
        res = bench_document(pdf, target, args, make_client)
        print(f"  {res.records}/{target} pairs, faithfulness {res.faithfulness:.2f}, "
              f"near-dup {res.near_dup_rate:.1%}, refusals {res.refusal_rate:.1%}, {res.seconds:.0f}s"
              + (f"  FAIL: {'; '.join(res.failures)}" if res.failures else "  ok"), flush=True)
        results.append(res)

    report = {
        "passed": all(not r.failures for r in results) and bool(results),
        "model": args.model,
        "judge_model": args.judge_model or args.model.split(",")[0].strip(),
        "base_url": args.base_url,
        "mode": args.mode,
        "gate": {"min_faithfulness": MIN_FAITHFULNESS, "max_near_dup_rate": MAX_NEAR_DUP_RATE,
                 "max_refusal_rate": MAX_REFUSAL_RATE, "yield_tolerance": YIELD_TOLERANCE},
        "documents": [asdict(r) for r in results],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = summary_markdown(report)
    print(md)
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
