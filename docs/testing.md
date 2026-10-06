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
yield. It runs nightly in CI when the `OPENAI_API_KEY` secret is set; see the
README for running it locally.
