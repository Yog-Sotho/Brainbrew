# Changelog

All notable changes to Brainbrew. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Change notes live here, not in code
comments. The phases refer to
[the roadmap](https://github.com/Yog-Sotho/Brainbrew/blob/main/docs/project/sota_roadmap.md).

## [Unreleased]: Phase 4, hardening, and audit follow-ups

### Added
- `mypy --strict` for the whole codebase (settings and file list in
  `pyproject.toml`; run `uv run mypy`); ruff's security (`S`),
  pytest (`PT`) and `RUF` rule sets; a pre-commit configuration that runs the
  locked tools.
- Property-based tests (Hypothesis) for the chunkers, formatters, record I/O,
  dedup and the sanitizer; mutation testing (mutmut) for the sanitizer, PII
  detection and dedup.
- CycloneDX SBOMs and a Trivy image scan in CI; signed releases (Sigstore) with
  SBOMs through `.github/workflows/release.yml`.
- Markdown documentation in `docs/` with an MkDocs site (built strictly in CI),
  replacing the 13 PDF manuals.
- This changelog.
- `bench/downstream.py` and the [downstream evaluation](https://github.com/Yog-Sotho/Brainbrew/blob/main/docs/downstream-eval.md)
  runbook: a held-out, closed-book exam written from the source documents (with
  near-copies of training questions removed), LoRA training, and paired grading
  of the base model, the Brainbrew adapter and a control adapter, with bootstrap
  confidence intervals. Sized for one 8 GB GPU.
- Running jobs are cancelled when the server shuts down, so they end as
  "cancelled" instead of holding the process open; finished jobs beyond the
  last 200 are dropped from memory (their run folders stay).

### Security
- Endpoint URLs pointing at link-local or cloud metadata addresses are
  rejected (SSRF), including numeric and IPv4-mapped spellings. At connect
  time the client also refuses hosts that *resolve* to such an address, and
  redirects to one, and connects only to the address it checked (no DNS
  rebinding).
- With login on, custom endpoints are off unless the operator sets
  `BRAINBREW_ALLOW_CUSTOM_ENDPOINTS=1` (fail closed on shared servers).
- With login on, the server's Hugging Face token only publishes to the
  operator's namespace (`HF_USERNAME`); other repos need the user's own token.
- Base image bumped: the pinned Python image had 8 fixable HIGH/CRITICAL
  Debian vulnerabilities (found by the new Trivy scan).

### Fixed
- An empty `OPENAI_BASE_URL` (what an undefined CI variable expands to) made
  the SDK send every request to a URL without a scheme, so the first real
  benchmark run failed all 252 requests. The engine now treats an empty value
  as unset and uses OpenAI, as the web app already did.
- Failed requests log their cause chain; "Connection error." alone did not say
  why.
- The benchmark workflow checks the API key and model with one request before
  running, and fails with a single clear error if OpenAI rejects them. The
  benchmark itself stops at the first fatal error (rejected key, no access,
  unknown model) instead of repeating it for every document.
- JSONL files could split one record across two lines when a text contained
  U+0085, U+2028 or U+2029, which JSON leaves unescaped but many readers treat
  as line breaks. Both writers now escape them (found by a property test).
- Failed generation requests were only counted. The first five are now logged
  with their error, and a run that produces nothing reports the last error.
- An invalid `BRAINBREW_MAX_JOBS` gave a bare `int()` error; it now says what
  the variable must be.

### Changed
- Internal structure only, with no change in behaviour: the pipeline stages,
  dataset card sections and LoRA trainer setup are separate functions; the
  Generate page's sidebar and widget-free logic moved to `ui/sidebar.py` and
  `ui/generate.py`.

### Removed
- Unused code: `exporter.deduplicate_records` (use `pipeline.dedup.deduplicate`),
  `pii.URL_POLICIES` and `quality.GRADES`.

## [2.0.0] - 2026-10-06: Phases 2 and 3

### Changed (breaking)
- distilabel is gone. Generation talks to any OpenAI-compatible endpoint
  (OpenAI, `vllm serve`, Ollama, llama.cpp, a custom URL) through an async
  client with bounded concurrency, retries and structured outputs.
- Config fields `use_vllm` and `batch_size` are removed; `concurrency`,
  `request_timeout`, `base_url`, `judge_model`, `seed` and the publishing and
  data-cleaning options are new.

### Added
- Grounded generation: typed questions per chunk (factual, conceptual,
  procedural, comparative, multi-hop), each answered from its passage; an LLM
  judge in Balanced and Research mode; refusal, non-answer and prompt-leak
  filters; a real target size with a yield estimate before the run.
- Near-duplicate removal with MinHash-LSH, optional paraphrase removal with
  embeddings, benchmark decontamination (GSM8K, MMLU, ARC, TruthfulQA,
  HumanEval), and PII detection v2 (Luhn, IBAN, IP context, URL policy,
  optional Presidio).
- Background job runner with live progress, cancel and a run-history page;
  runs survive page reloads. One GPU slot shared by the app and the CLI.
- Per-run `run.log`, `rejected.jsonl` and a manifest with models, seed, token
  usage, actual cost and stage timings.
- `brainbrew` command line (`run`, `runs list`, `runs show`).
- Hugging Face dataset and model cards; private repos by default; optional
  adapter upload.
- `compose.yaml` with a vLLM server; a benchmark corpus, `bench/run_bench.py`
  and a nightly benchmark workflow.

### Security
- The server's API key is only ever sent to the server's own endpoint, never
  to a URL a visitor chooses; `BRAINBREW_ALLOW_CUSTOM_ENDPOINTS=0` locks the
  endpoint.
- With login on, the gate runs on every page and users only see their own runs.

## [1.3.0] - 2026-10-05: Phases 0 and 1

### Security
- Server-side API keys are never sent to the browser; the app binds to
  127.0.0.1; optional OIDC login (`BRAINBREW_REQUIRE_LOGIN`); upload size limit.

### Added
- Locked dependencies (`uv.lock`, hash-pinned exports), Docker images, CI
  (lint, types, tests, audit, secrets, Docker), Dependabot.
- Canonical records with formatting only at export; persistent run folders
  with a manifest; LoRA training on TRL + PEFT.

### Removed
- Run resume, which lost data in every interrupted-run test.

## [1.2.0] and earlier

- `DistillationConfig.safe_dict()` redacts the API key and HF token, and
  `repr`/`str` never show them (formerly marked C-01, C-02 in the code).
- One shared Hugging Face repo-name rule for the UI, the config and the
  publisher (M-10).
- The installer reads API keys without echoing them (H-06).
- Exact and near-duplicate dedup (Enhancement 5), paragraph-aware semantic
  chunking (Enhancement 9) and the dataset quality report (Enhancement 10).
- Sanitizer and dedup fast paths: skip work when the text has no `<`, no
  non-ASCII characters or no repeated whitespace (2026-07 to 2026-08).
