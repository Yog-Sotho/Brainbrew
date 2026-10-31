# Changelog

All notable changes to Brainbrew. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Change notes live here, not in code
comments. The phases refer to
[the roadmap](https://github.com/Yog-Sotho/Brainbrew/blob/main/docs/project/sota_roadmap.md).

## [Unreleased]

### Security
- Locked dependencies raised past newly published advisories that turned the `audit`
  and `docker` (Trivy) CI jobs red: pillow 13.0.0 (39 advisories, several critical; needs
  streamlit 1.66.1, as 1.65 caps pillow below 13), authlib 1.9.1, certifi 2026.10.23,
  filelock 4.2.4, joserfc 1.7.7 and, for the `train` extra, setuptools 84.2.1.
- The CPU image's `python:3.12-slim-bookworm` base is re-pinned to a digest with the fixed
  OpenSSL (3.0.22-1~deb12u2, CVE-2026-84782).

## [3.0.2] - 2026-10-20

### Added
- Container image `ghcr.io/yog-sotho/brainbrew:{version,latest}` for `linux/amd64`
  and `linux/arm64`, published by the release workflow: the CPU (`api`) target,
  signed with Sigstore (keyless) and with build provenance. A `Docker` workflow
  builds and smoke-tests the arm64 image for pull requests that change it (ci.yml
  already covers amd64).

## [3.0.1] - 2026-10-20

### Added
- PyPI publishing via OIDC Trusted Publishing. The distribution is named
  `brainbrew-llm` (the bare name `brainbrew` is unavailable on PyPI); `pip
  install brainbrew-llm` still provides the `brainbrew` command.

### Security
- aiohttp raised to 3.14.5 (CVE-2026-111063, CVE-2026-111064,
  GHSA-2g87-pp6c-29x5).

## [3.0.0] - 2026-10-18: Phase 4, hardening, and audit follow-ups

A major version because two things are removed (see *Removed*): the `vllm` extra
(vLLM now runs in its own environment) and three unused functions.

### Added
- The documentation is published to GitHub Pages at
  <https://yog-sotho.github.io/Brainbrew/> by a new `docs.yml` workflow on every push to
  `main` that changes it.
- The benchmark workflow takes a model provider (`BENCH_PROVIDER`): `openai`,
  `gemini` (free tier, secret `GEMINI_API_KEY`, `gemma-4-31b-it`: the only
  model there whose free daily quota fits a run) or `custom` (any OpenAI-compatible endpoint). Each has its own key secret,
  default model and parallel-request limit, and the key check works for all of
  them.
- `reasoning_effort` setting (config, CLI config files, `bench/run_bench.py
  --reasoning-effort`): `none` turns thinking off where the server allows it, so
  thinking does not use up the answer's token budget.
- A server that rejects the `seed` field gets requests without it, and one that
  rejects system messages (Gemma on Google's API) gets their text at the start
  of the user turn. Each is learned once and logged, instead of failing every
  request.
- Rate limits that name their wait are followed: a short one (Gemini's "retry in
  36s") becomes the retry delay, and one of hours (a used-up daily quota) stops
  the run at once instead of retrying every request.
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
- setuptools 80.10.2 -> 84.0.0, fixing CVE-2026-59890 / GHSA-h35f-9h28-mq5c
  (`MANIFEST.in` exclusions bypassed in sdists; the four Dependabot alerts). It was
  held back only by vLLM's `setuptools<81` cap; with vLLM out of the lock (see
  Removed) nothing caps it. The package itself is built with setuptools >= 83, and
  CI's pip-audit no longer ignores the advisory.
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
- `narwhals` 2.27.0 (a Streamlit dependency) was yanked upstream; the lock moves to 2.27.1.
- The nightly benchmark no longer goes red when the model provider is down. On
  2026-10-13 and 14 Google answered `500 Internal error` to many or all
  `gemma-4-31b-it` requests: one night a document fell short of its yield, the
  next nothing was generated. A shortfall the provider's failed requests account
  for is now a warning, and a document with no pairs because every request failed
  stops the run as "not measured". Quality gates still fail as before.
- Dependabot's uv updates opened no pull requests: its updater could not recompute
  the hashes in the exported `requirements*.txt` and failed on every dependency.
  It now skips those files; regenerate them with `uv export` on the PR branch.
- An empty `OPENAI_BASE_URL` (what an undefined CI variable expands to) made
  the SDK send every request to a URL without a scheme, so the first real
  benchmark run failed all 252 requests. The engine now treats an empty value
  as unset and uses OpenAI, as the web app already did.
- Failed requests log their cause chain; "Connection error." alone did not say
  why.
- The benchmark workflow checks the API key, the model and the account with a
  1-token request before running, and fails with a single clear error if
  OpenAI refuses them. The benchmark itself stops at the first fatal error
  (rejected key, no access, unknown model, no credits) instead of repeating it
  for every document.
- An OpenAI account without credits answers 429, which looked like a rate
  limit: every request was retried with backoff and every chunk tried (252
  requests over five minutes in the benchmark). "No credits" is now fatal: it
  is not retried and stops the run at once, in the app, the CLI and the
  benchmark. Ordinary rate limits are still retried.
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
- The `vllm` extra and `requirements-vllm.txt` / `requirements-gpu.txt`. Brainbrew
  never imported vLLM; it talks to `vllm serve` over HTTP like any endpoint, so vLLM
  now lives in its own environment with its own pins: `uv tool install vllm==0.31.0`,
  `install.sh`'s `.venv-vllm` (used by `run_vllm.sh`), or the official image in
  `compose.yaml`. The GPU Docker image is Brainbrew + LoRA training (`GPU_EXTRAS=train`),
  and the release's GPU SBOM is now the training-stack SBOM (`-train-sbom.cdx.json`).
  The lock drops from 288 to 182 packages.
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
