# Testing

```bash
uv run pytest                     # everything, with the locked environment
uv run pytest tests/test_app.py   # one area
uvx pre-commit run --all-files    # ruff, mypy --strict, lock check
```

No GPU and no API key are needed.

## What the tests run

- **The real pipeline.** Generation tests use the real async engine and openai
  SDK against `tests/fake_openai.py`, an in-memory OpenAI-compatible server that
  can misbehave on request: rate limits, server errors, broken JSON, refusals,
  missing structured-output support, slow responses.
- **The real app.** `tests/test_app.py` drives `app.py` and the run-history page
  with Streamlit's `AppTest`: upload, generate, reload, cancel, results.
- **The real CLI** through Typer's runner, and **real LoRA training** on tiny Hub
  models in the `train-contract` CI job (CPU torch).
- **Property tests** (`tests/test_properties.py`, Hypothesis) for chunkers,
  formatters, record I/O, dedup and the sanitizer.
- **Security tests**: keys never reach logs, run folders, the browser or a URL
  the visitor chose.

## Mutation testing

```bash
uv run mutmut run       # mutates the sanitizer, PII detection and dedup
uv run mutmut results
```

Surviving mutants point at behaviour no test pins down.

## Quality benchmark

`bench/run_bench.py` generates datasets from three public-domain PDFs in
`tests/fixtures/bench/` and checks faithfulness, near-duplicates, refusals and
yield. `.github/workflows/bench.yml` runs it nightly and on demand. The
`BENCH_PROVIDER` repository variable (or the provider input of a manual run)
picks the model provider:

| Provider | Setup | Default model |
|---|---|---|
| `openai` (default) | secret `OPENAI_API_KEY` (paid) | `gpt-4o-mini` |
| `gemini` | secret `GEMINI_API_KEY`, a free key from [Google AI Studio](https://aistudio.google.com/apikey) | `gemma-4-31b-it`, 4 parallel requests (about three hours per full run) |
| `custom` | variables `BENCH_BASE_URL` and `BENCH_MODEL`, optional secret `BENCH_API_KEY` | – |

- `BENCH_MODEL` overrides the model, `BENCH_JUDGE_MODEL` sets a different judge
  (a more independent faithfulness score), and `BENCH_CONCURRENCY` sets the number
  of parallel requests; lower it if free-tier rate limits bite.
- Without the provider's key, the job is skipped with a notice.
- Before running, one tiny request checks the key, the model and the account, and
  names the fix if any of them is refused.
- Gemini's free tier is small and differs per model; check yours at
  https://aistudio.google.com/rate-limit. On 2026-10-06, `gemini-3.8-flash`
  allowed 5 requests per minute and 20 per day, while one full benchmark run
  needs a few hundred; the Flash-Lite models allowed 500 a day and Gemma 4 14,400.
  Gemma is therefore the default. On the free tier it answers in 30–60 s per
  request and sometimes returns 500/503 errors (retried), so a run takes hours;
  your own GPU (below) is much faster.
- A provider that asks for a short wait (a per-minute limit) is waited out. One
  that asks for hours (a daily quota) stops the run at once with its message.
- On Gemini's free tier, Google may use the requests to improve its products. The
  benchmark corpus is public domain, but do not send private documents through a
  free tier.

To run the benchmark locally for free against your own GPU:

```bash
uv run vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --max-model-len 8192   # terminal 1
uv run python bench/run_bench.py --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ --concurrency 4                # terminal 2
```

## Downstream evaluation

`bench/downstream.py` checks whether a model trained on a Brainbrew dataset gets
better. It writes a held-out, closed-book exam from the source documents and
wraps LoRA training. It then grades the base model, the Brainbrew adapter and a
control adapter, with confidence intervals. It needs a GPU; the procedure is in
[Downstream evaluation](downstream-eval.md).
