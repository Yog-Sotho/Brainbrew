# Codebase Audit Report

**Date:** 2026-10-06
**Project:** Brainbrew v2.0.0 + Phase 4 and the audit follow-ups (branch `ccr-68d77136-hi00bp`)
**Language / Framework:** Python 3.12 · Streamlit · OpenAI-compatible async client (OpenAI, vLLM, Ollama, llama.cpp) · TRL + PEFT · Hugging Face Hub · Typer CLI
**Project Type:** Web app (Streamlit UI and CLI driving an LLM and GPU batch pipeline). All 9 dimensions apply.
**Audit Mode:** Global. About 5.4k LOC of application code and 5.3k LOC of tests.
**Revision:** updated after the follow-ups to the Phase 4 gate audit (8.3): every LOW finding with a code fix is closed.
**Baseline:** [codebase_audit_baseline.md](./codebase_audit_baseline.md), 3.2 / 10 on 2026-10-05 (v1.2.0).

---

## Executive Summary

Brainbrew now installs from a lock file, builds a clean Docker image and runs end to end against real OpenAI-compatible servers. Every baseline CRITICAL and HIGH finding is closed, and so are the four this audit raised:

- an SSRF path to cloud metadata services;
- the server's Hugging Face token being usable for any repo under login;
- generation failures that were counted but never explained;
- jobs that held the process open at shutdown.

The checks behind this:

- 643 tests (96% line coverage), with property-based tests and an 87.5% mutation score on the sanitizer, PII detection and dedup;
- `mypy --strict` on the whole codebase, and ruff with the bandit rules;
- in CI: pip-audit, gitleaks, CodeQL, Trivy and CycloneDX SBOMs, plus Sigstore-signed releases.

The follow-ups closed the gate audit's LOW findings:

- metadata addresses are now also refused at connect time, after DNS resolution and on redirects;
- the longest functions are split;
- the Generate page's sidebar and logic moved into `ui/`;
- strict typing covers every module.

Two documented limits remain, both deployment choices rather than code defects:

- behind an HTTP proxy, only the URL is checked;
- an operator who opts in to custom endpoints on a shared server lets users reach private-network hosts.

**8.7 overall, with no CRITICAL, HIGH or MEDIUM findings open** (the Phase 4 gate needed ≥ 8.0).

## Overall Score: 8.7 / 10

| Dimension | Score | Baseline | Priority | Status |
|---|---|---|---|---|
| Security | 9/10 | 3 | CRITICAL | PASS |
| Build & Types | 10/10 | 1 | CRITICAL | PASS |
| Code Principles (correctness and design) | 9/10 | 2 | HIGH | PASS |
| Code Quality | 9/10 | 5 | MEDIUM | PASS |
| Dependencies | 8/10 | 2 | MEDIUM | PASS |
| Dead Code | 9/10 | 5 | LOW | PASS |
| Observability | 8/10 | 4 | MEDIUM | PASS |
| Concurrency | 8/10 | 4 | HIGH | PASS |
| Lifecycle | 8/10 | 3 | MEDIUM | PASS |

Average = 78 / 9 = 8.7 (the gate audit scored 75 / 9 = 8.3). No CRITICAL findings, so neither hard cap applies. Open findings: 0 CRITICAL, 0 HIGH, 0 MEDIUM, 2 LOW, 3 INFO.

### Closed by the follow-ups (after the 8.3 gate audit)

| Was | Location | Fix |
|---|---|---|
| LOW (Security) | `config.py` | The metadata check looked at the URL only. `engine/netguard.py` now resolves the host itself, refuses if any answer is a metadata or link-local address, and connects to the checked address. That covers names that resolve there, DNS rebinding and redirects. 23 tests, including a real local server that redirects to `169.254.169.254`. |
| LOW (Build & Types) | `app.py`, `ui/`, `pages/`, `publish/`, `training/` | `strict = true` now applies to all 40 files (`pyproject.toml`); CI and pre-commit run one `mypy`. |
| LOW (Code Principles) | `orchestrator._run`, `dataset_card`, `train_lora` | `_run` is a 40-line sequence of stage functions. The dataset card is built from section builders, with byte-identical output checked against a snapshot. The trainer's model loading and arguments are helpers, and the real TRL + PEFT tests pass. |
| LOW (Code Quality) | `app.py` | The sidebar moved to `ui/sidebar.py` (typed `SidebarSettings`) and the widget-free logic to `ui/generate.py` (11 unit tests). `app.py` went from 462 to 268 lines; the AppTest suite is unchanged. |

### Findings raised and fixed during the gate audit

These were found while auditing and fixed before scoring, each with tests. They are listed so the score is not read as "nothing was found".

| Severity | Location | Issue | Fix |
|---|---|---|---|
| HIGH | `config.py:51` `check_base_url` | A visitor could set the endpoint to `http://169.254.169.254/...`, so the server would query its own cloud metadata service and could reflect credentials back in error messages (SSRF). | Link-local and metadata hosts are rejected, including numeric (`2852039166`, `0xa9fea9fe`, `169.254.43518`) and IPv4-mapped IPv6 spellings. 17 tests in `tests/test_security.py`. |
| HIGH | `app.py` (publish), `ui/common.py` | With login on, any user could publish with the server's `HF_TOKEN` to any repo that token can write to, including overwriting the operator's other datasets. | With login on, the server token only publishes under `HF_USERNAME`; anything else needs the user's own token. |
| MEDIUM | `app.py`, `ui/common.py` | Custom endpoints were on by default even with login on, so any signed-in user could make the server send requests into its private network. | With login on, custom endpoints are off unless `BRAINBREW_ALLOW_CUSTOM_ENDPOINTS=1` (fail closed). |
| MEDIUM | `pipeline/synth.py` | Failed generation requests were only counted. A run that produced nothing said "no usable pairs" with no cause. | The first five failures are logged with their error, then a single "only counted" note. The failure message includes the last error. |
| MEDIUM | `pipeline/jobs.py` | At shutdown Python waited for running jobs to finish, which could take hours. A killed container left runs to be marked "interrupted" later. `BRAINBREW_MAX_JOBS=abc` failed with a bare `int()` error. The job registry grew without bound. | A `threading` exit hook (it runs before the executor joins its workers) cancels active jobs, so they end as "cancelled". The variable is validated with a clear message. Only the last 200 finished jobs are kept in memory. A subprocess test proves the exit path. |
| LOW | `pipeline/exporter.py`, `pii.py`, `quality.py` | `deduplicate_records`, `URL_POLICIES` and `GRADES` had no production callers. | Removed; the tests use `pipeline.dedup.deduplicate`. |

---

## Strengths

- **Secrets never leave the server.** Server keys are never widget values. `DistillationConfig` keeps secrets out of `repr`, `str`, manifests and logs. Tests scan every file in a run folder and the run log for the key. The server's API key is only sent to the server's own endpoint, and the CLI refuses config files that contain secrets.
- **Correctness is tested against real behaviour, not mocks of it.** Contract tests use the real `openai` SDK against an HTTP fake server. AppTest drives the real Streamlit script, and the CLI is tested end to end. Hypothesis checks invariants for chunkers, formatters, record I/O, dedup and the sanitizer. A property test found a real bug: unescaped U+2028 split JSONL records across lines. Mutation testing measures how strong the sanitizer, PII and dedup tests are.
- **Supply chain.** `uv.lock` with hash-pinned exports. CI runs pip-audit, gitleaks, CodeQL, a Trivy image scan (which caught 8 fixable HIGH/CRITICAL Debian CVEs in the old base image), CycloneDX SBOMs and a strict docs build. Releases are built, SBOM'd and Sigstore-signed, and the version must match the tag. Dependabot covers pip, Actions and Docker, and Actions and base images are pinned by digest.
- **Operable runs.** Every run has its own folder with:
  - a manifest: config without secrets, models, seed, token usage, actual cost, stage timings, owner and status;
  - `run.log` and `rejected.jsonl`.

  Runs survive reloads, can be cancelled mid-request, and show as "interrupted" or "running elsewhere" when the owning process is gone or is another process. One GPU slot is shared across processes.
- **Clear module boundaries.** The engine (HTTP and structured output), pipeline stages (chunk → synthesise → judge → filter → dedup → decontaminate → sanitise → export), publishing and the UI are separate packages. Records stay canonical until export.

---

## Findings by Dimension

### Security [9/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| LOW | `engine/netguard.py` | Behind an HTTP proxy (`HTTPS_PROXY`), the proxy resolves the target, so only the URL check applies there. | Documented in `docs/security.md`: block the metadata address at the proxy or in the network policy as well. |
| LOW | `app.py` (custom endpoint), `pipeline/synth.py` | Once an operator opts in to custom endpoints with login on, a signed-in user can reach private-network hosts, and the last error (up to 500 characters) is shown to them. | This is by design for local vLLM and Ollama. The opt-in is documented. If such servers are exposed, prefer a server-side allow-list of endpoints over `=1`. |
| INFO | `pyproject.toml` (gpu extra) | PYSEC-2026-3447 (setuptools) is accepted: vLLM caps setuptools below the fixed version. The base `requirements.txt` audits clean. | Lift the ignore when vLLM relaxes its cap; Dependabot will propose it. |

The checks: ruff `S` passes with documented `noqa`s only. There is no `eval`, `exec`, `pickle`, `yaml.load` or `subprocess` in production code. Repo names, model names, run ids and URLs are validated against strict patterns. The app binds to `127.0.0.1`, the OIDC gate fails closed and runs on every page, runs are owner-scoped, uploads are size-limited and file names are never used as paths. Containers run as non-root with no secrets baked in.

### Build & Types [10/10]

No open findings. `mypy --strict` covers all 40 source files, and a probe confirmed it rejects an untyped function. `uv lock --check` passes and the exports are in sync. ruff (E, F, W, I, UP, B, SIM, S, PT, RUF), both mypy runs and pre-commit are clean. The Docker image builds, passes its healthcheck and scans clean. `mkdocs build --strict` passes.

### Code Principles [9/10]

No open findings. The remaining functions over 70 lines are advisory (see below). Earlier correctness gaps are closed. There is one validation source for the UI, the CLI and the config, and records stay canonical until export. The engine degrades structured output (`json_schema` → `json_object` → schema in the prompt) and remembers what each server supports.

### Code Quality [9/10]

No open findings. The largest file is now `pipeline/sanitizer.py` (424 lines, one responsibility). Coverage is 96% (2,934 statements, 113 missed; most of the gap is GPU-only training code). There are 643 tests plus 4 skipped GPU tests. The mutation score is 87.5% (dedup 95%, PII 92%, sanitizer 79%; the remaining survivors are mostly equivalent mutants in hash and encoding formatting). There are no TODO or FIXME markers and no `print` in production code, and the change-log comments have moved to `CHANGELOG.md`.

### Dependencies [8/10]

No open findings. The lock file holds 289 packages. Optional extras keep vLLM and training out of the base install. pip-audit is clean apart from the INFO item above. One item to watch: Material for MkDocs warns that MkDocs 2.0 will break plugins, so the `docs` group is pinned to `mkdocs>=1.6,<2`.

### Dead Code [9/10]

No open findings. The three unused symbols were removed (see above). Vulture's other hits at 60% confidence are false positives: pydantic validators, Typer commands, dataclass and TypedDict fields, and `model_config`.

### Observability [8/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| INFO | n/a | There are no metrics or tracing endpoints. Run state, token usage, cost and timings are in each manifest and in structured logs. | If Brainbrew is run as a shared service, export manifest fields to Prometheus or OpenTelemetry. |

structlog and stdlib logging are configured once, with JSON in containers. Each run has its own `run.log`. Failed requests now carry their cause. Both images have a Docker healthcheck, and compose waits on vLLM's health.

### Concurrency [8/10]

| Severity | Location | Issue | Recommendation |
|---|---|---|---|
| INFO | `pipeline/jobs.py:63`, `:98` | A worker thread updates `Job.status`, `progress` and `stage`, and the UI thread reads them without a lock. Each is a single attribute assignment from one writer, which is atomic under CPython, and a reader can at worst see a stage one poll late. | No change needed. If a free-threaded build is targeted, group the fields in an immutable snapshot that is swapped atomically. |

The runner is a bounded thread pool. Model requests run under an asyncio semaphore. The GPU slot is a cross-process file lock that can be cancelled while waiting. Manifest writes are atomic (temp file plus `os.replace`, under a lock), and per-process temp names avoid collisions. Phase 3's gate ran two browser sessions and a CLI run at the same time without any data mixing between them.

### Lifecycle [8/10]

No open findings. Jobs are cancelled at interpreter exit, and Streamlit turns SIGTERM into a normal shutdown, so `docker stop` ends running jobs as "cancelled". Runs whose process died show as "interrupted". The GPU slot's file lock is released by the OS when its holder dies. Configuration errors fail fast with clear messages. Resume was removed in Phase 1 because it lost data; a run is restarted rather than resumed, which is documented.

---

## Advisory Findings

- `[High cohesion module]` `pipeline/document_loader.py:50` `semantic_chunk` (78 lines) and `pipeline/sanitizer.py:353` `sanitize_dataset` (72 lines) each do one thing over one data model at one level of abstraction. Property and mutation tests pin their behaviour.
- `[High cohesion module]` `cli.py:92` `run` (81 lines) is mostly Typer option declarations for one command.
- `[High cohesion module]` `training/lora_trainer.py` `train_lora` (70 lines, 24 of them docstring) and `orchestrator.run_distillation` (63, mostly the manifest header and the status transitions) read top to bottom at one level of abstraction.
- `[Single consumer, locality correct]` `ui/common.server_hf_token_allowed` and `custom_endpoints_allowed` have one caller each (`app.py`), as do the `ui/generate.py` helpers. They live in `ui/` so they can be unit-tested without AppTest.
- `[Private API, guarded]` `engine/netguard.guard_client` sets the private `_network_backend` of httpx2's connection pools; there is no public hook. It raises if the attribute disappears, so an upgrade fails loudly instead of sending requests unguarded, and a test checks that every pool is guarded, proxies included.

---

## Comparison with the Baseline (2026-10-05, 3.2 / 10)

| Baseline finding | Now |
|---|---|
| `requirements.txt` unresolvable; Docker image would not build; CI never ran | `uv.lock` and hash-pinned exports; the image builds, passes its healthcheck and scans clean; CI runs lint, types, tests, audit, secrets, CodeQL, Docker, Trivy, SBOM and docs |
| Pipeline failed on real distilabel 1.5.2; LoRA called a non-existent Unsloth API | distilabel replaced by our own OpenAI-compatible engine with contract tests against the real SDK; training on TRL + PEFT |
| Server API key and HF token sent to every browser | Secrets stay server-side; the server key only goes to the server endpoint; the HF token is scoped under login |
| Tests passed only because mocks encoded wrong APIs | Contract tests over HTTP, AppTest, CLI end-to-end, property and mutation tests |
| Lost results on reload; no cancel; no run history | Background jobs, URL re-attach, cancel, history page, per-run folders and logs |
| PDFs as documentation; change notes in code comments | Markdown docs with a strict MkDocs build; `CHANGELOG.md` |

---

## Recommended Actions (Priority Order)

1. [LOW] Offer a server-side endpoint allow-list (for example, `BRAINBREW_ENDPOINTS`) as a middle ground between "locked" and "any URL" for shared servers.
2. [LOW] Behind a proxy, block metadata addresses at the proxy as well (operator action, documented).
3. [INFO] Export run metrics if Brainbrew is run as a shared service.
4. [INFO] Lift the setuptools advisory ignore when vLLM relaxes its cap.

---

## Sources Consulted

- The repository at this commit: ruff, `mypy` and `mypy --strict`, pytest with coverage, mutmut 3.8, Hypothesis, `uv lock --check`, pip-audit, vulture, Trivy 0.75 (image scan), and `mkdocs build --strict`.
- The baseline report, [codebase_audit_baseline.md](./codebase_audit_baseline.md), and the roadmap, [sota_roadmap.md](./sota_roadmap.md).
- Python `threading` and `concurrent.futures` shutdown order (`threading._register_atexit`), and the Streamlit signal handling for SIGTERM.
- AWS, GCP and Azure instance metadata endpoints (`169.254.169.254`, `fd00:ec2::254`, `metadata.google.internal`, `metadata.azure.com`).
