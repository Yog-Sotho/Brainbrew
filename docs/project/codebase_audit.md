# Codebase Audit Report

**Date:** 2026-10-05
**Project:** Brainbrew v1.2.0 (`main` @ `54f8632`)
**Language / Framework:** Python 3.12 · Streamlit · distilabel 1.5.2 · vLLM / OpenAI · Unsloth + TRL · Hugging Face Hub
**Project Type:** Web app (Streamlit UI driving a GPU / LLM batch pipeline). All 9 dimensions apply.
**Audit Mode:** Global. About 2.1k LOC of application code and 2.4k LOC of tests.

---

## Executive Summary

Brainbrew has clean, readable modules, careful input validation and a large test suite (255 tests, all passing). **The product does not work end to end, and it cannot be installed.** Checked against the real libraries:

1. `requirements.txt` cannot be resolved (3 version conflicts).
2. The Docker image cannot be built.
3. The CI workflow never runs.
4. The generation pipeline fails with real distilabel 1.5.2 at two points: LLM construction and validation of the custom Step.
5. LoRA training calls an Unsloth API that does not exist.

The tests pass only because every heavy dependency is mocked, and the mocks encode the wrong APIs. The UI also sends the server's OpenAI and HF secrets to every browser that opens the page.

The top priority is to restore a working, installable and secure baseline (Phases 0–1 of the roadmap) before adding any feature.

## Overall Score: 3.2 / 10 (SOTA score: **3 / 10**)

| Dimension | Score | Priority | Status |
|---|---|---|---|
| Security | 3/10 | CRITICAL | FAIL |
| Build & Types | 1/10 | CRITICAL | FAIL |
| Code Principles (correctness and design) | 2/10 | HIGH | FAIL |
| Code Quality | 5/10 | MEDIUM | WARN |
| Dependencies | 2/10 | MEDIUM | FAIL |
| Dead Code | 5/10 | LOW | WARN |
| Observability | 4/10 | MEDIUM | WARN |
| Concurrency | 4/10 | HIGH | WARN |
| Lifecycle | 3/10 | MEDIUM | FAIL |

Average = 29 / 9 = 3.2. Hard caps for CRITICAL Security (≤ 4.0) and CRITICAL Build (≤ 5.0) are below the average and do not change it.

**Methodology aspect (outside the 9 dimensions):** the generation approach is 2023-era. It produces one Evol-Instruct pass per character chunk. It has no grounding or faithfulness check, no LLM-as-judge (`judge_model` exists but is never used), no semantic dedup and no decontamination. The 2026 practice is knowledge-base-anchored generation with judge filtering (see Sources).

---

## Strengths

- **Defensive config layer.** `config.py` validates every model name, repo id, path and secret: path traversal, absolute paths, control characters and length limits. It also redacts secrets in `safe_dict`, `__repr__` and `__str__`.
- **Fast, well-reasoned sanitizer and dedup code.** The fast-path pre-checks, the Jaccard length-ratio pruning and the set-arithmetic union are correct optimizations, each backed by a benchmark (`.jules/bolt.md`).
- **Lazy heavy imports.** `train_lora` and `publish_dataset` import GPU and Hub packages only when they are called, so the core modules import on CPU-only hosts.
- **Large, organized test suite.** 255 tests pass in about 2 s. Coverage is 84 % excluding `app.py`, with clear security regression tests.
- **Safe defaults.** HF repos are created private, the container runs as a non-root user and has a HEALTHCHECK, and the installer reads secrets silently.

---

## Findings by Dimension

Every CRITICAL and HIGH finding below was **reproduced or verified live**: by running real distilabel 1.5.2, by resolving dependencies with `uv`, by inspecting the downloaded unsloth and trl wheels, with pip-audit, and from official docs and advisories.

### Security [3/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| **CRITICAL** | `app.py:44-62`, `Dockerfile:38` | `OPENAI_API_KEY` and `HF_TOKEN` from the server environment are pre-filled as the `value=` of `type="password"` widgets. Streamlit sends widget values to the browser, where they can be read in the DOM ([Streamlit forum](https://discuss.streamlit.io/t/text-input-type-password-is-leaking-in-the-dom/52322)). The app binds `0.0.0.0` with no authentication, and the README recommends RunPod/Modal. Anyone who can reach the port can take both credentials. | Never echo server secrets into widgets. Use the env or `st.secrets` value server-side and show only "key configured ✓". Add authentication (`st.login` OIDC) or bind to `127.0.0.1` by default. |
| HIGH | `Dockerfile:27`, missing `.dockerignore` | `COPY . .` with no `.dockerignore` copies `.env` (which `install.sh` creates) into the image layers, along with `.git`, `old/` and `assets/`. | Add `.dockerignore` (`.env*`, `.git`, `old/`, `docs/`, `tests/`, `*.jsonl`, `.venv`). |
| HIGH | `requirements.txt:12` (pdfminer.six 20240706) | CVE-2025-64512 (CVSS 8.6): arbitrary code execution through pickle deserialization when parsing a crafted PDF. The app parses **untrusted uploaded PDFs**. Also CVE-2025-70559. | Upgrade to ≥ 20251107 (latest is 20260107). |
| HIGH | `requirements.txt:25,30,31` | pip-audit reports **52** advisories on vllm 0.19.0 (fixes up to 0.30.0), 7 on transformers 5.0.0rc3 and 1 on accelerate 1.2.1. Also python-dotenv (CVE-2026-28684), langchain-text-splitters (CVE-2026-41481) and datasets (CVE-2026-66007). | Re-pin to patched versions in a resolved lock file, and add `pip-audit` to CI. |
| HIGH | `orchestrator.py:233-300`, `pipeline/sanitizer.py:263` | **The PII scrubbing promise is silently broken for 3 of 4 formats.** `_extract_text_for_quality` reads only top-level strings, so ShareGPT, ChatML and OpenAI records score as "empty text" and 100 % are rejected (reproduced: 0/30 kept). The orchestrator then **keeps the unsanitized file** and logs only a warning. A user who enabled "Clean & sanitize" gets PII in the output. | Sanitize the canonical `{instruction, input, output}` records *before* formatting. Treat an empty sanitizer result as an error. |
| MEDIUM | `pipeline/sanitizer.py:93-113` | The PII regexes are both over- and under-inclusive. Reproduced: `version 1.2.3.4` → `[PII_IP]` and `1234567890 rows` → `[PII_PHONE]`. Card numbers have no Luhn check. There is no name or address detection. | Add Luhn and IP context checks, or use Presidio, as a pluggable redactor. Make URL redaction optional, because technical datasets need URLs. |
| MEDIUM | `app.py:134-145` | The 50 MB limit is checked *after* Streamlit has buffered the upload (the server limit defaults to 200 MB). There is no rate limiting or job quota. | Set `server.maxUploadSize` in `.streamlit/config.toml`, and add a per-session job lock. |
| LOW | `app.py:142` | The filename allow-list blocks legitimate files such as `Report (1).pdf` and non-ASCII names, yet the filename is never used as a path. | Drop the check, or sanitize names for display only. |
| INFO | bandit B105/B107/B615 | All are false positives: a redaction placeholder, an EOS token, and `load_dataset("json")` on a local file. | Suppress with `# nosec` and a reason. |

### Build & Types [1/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| **CRITICAL** | `requirements.txt` | **Unresolvable.** `uv pip compile` (py3.12, linux x86_64) fails because vllm 0.19.0 requires `pydantic>=2.12` (pinned 2.10.6) and `openai>=2.0` (pinned <2.0). Wheel metadata also shows unsloth 2024.12.7 requires `transformers<=4.46.3` and trl 0.12.2 requires `transformers<4.47`, but `transformers==5.0.0rc3` (a release candidate) is pinned. `pip install -r requirements.txt` cannot succeed. | Split dependencies into `core`, `[vllm]` and `[train]` extras in `pyproject.toml`. Resolve and lock them with `uv lock`, including hashes. |
| **CRITICAL** | `Dockerfile:1-4,14-15` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` (jammy) has **no `python3.12` package** ([packages.ubuntu.com](https://packages.ubuntu.com/jammy/python3.12): "not available in this suite"). `apt-get install python3.12` fails. Even with deadsnakes, `python3-pip` would install for 3.10 and `--prefix=/install` would target the wrong interpreter. | Use an Ubuntu 24.04 CUDA base (python3.12 is native) with a uv-managed venv. Use a slim `python:3.12` image for API-only mode. |
| **CRITICAL** | `ci.yml` (repo root) | GitHub Actions reads workflows only from `.github/workflows/`, and `.github/` does not exist, so **CI has never run**. The README's "full CI/CD" claim is false, and the Dependabot config is also missing. | Move the file to `.github/workflows/ci.yml` and add `dependabot.yml`. |
| HIGH | `setup.py:36` | The console script `brainbrew=app:main` points to a `main()` that does not exist. `app.py` is a top-level Streamlit script. | Add a Typer CLI `main()` (also useful for headless runs) or remove the entry point. |
| HIGH | `pyproject.toml`, `setup.py`, `requirements.txt`, `poetry.lock` | Four packaging sources drift apart. `pyproject.toml` declares no dependencies, and `poetry.lock` is an empty stub. | Make `pyproject.toml` (PEP 621) plus `uv.lock` the single source of truth, and delete `setup.py` and `poetry.lock`. |
| MEDIUM | `app.py:303` / mypy | mypy reports 5 errors, including missing `temperature`, `max_new_tokens`, `batch_size` and `lora_rank` arguments (no pydantic mypy plugin) and `no-any-return` at `orchestrator.py:166`. CI type-checks only `config.py pipeline/ publish/`. | Enable `plugins = ["pydantic.mypy"]`, type-check every module, and run `--strict` on `pipeline/` and `config.py`. |
| LOW | ruff | 12 errors, e.g. `B905` zip without `strict=`, `F841`, `B017`, `SIM*` and `UP042` (use `StrEnum`). | `ruff check --fix` and `ruff format`, enforced with pre-commit. |

### Code Principles: Correctness & Design [2/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| **CRITICAL** | `orchestrator.py:185-199` | `vLLM(... max_new_tokens=, temperature=)` and `OpenAILLM(...)` both raise `ValidationError: Extra inputs are not permitted` in distilabel 1.5.2 (reproduced). In distilabel these are `generation_kwargs`, not constructor fields. **Every generation run crashes before any LLM call.** | `OpenAILLM(model=..., api_key=..., generation_kwargs={"max_new_tokens":..., "temperature":...})`, and the same for `vLLM`. |
| **CRITICAL** | `orchestrator.py:48-68` | `FilterAndRenameOutputs.process(self, inputs: list[dict])` fails distilabel DAG validation: *"should have a parameter with type hint `StepInput`"* (reproduced with a real `Pipeline` and a dummy LLM). | Annotate with `StepInput` / `StepOutput`, and add a contract test that builds the real DAG on CPU. |
| HIGH | `orchestrator.py:216-221` | **Instruction/response mismatch.** `TextGeneration` answers `evolved_instruction`, but `KeepColumns(["instruction","output"])` keeps the *original seed prompt*. Reproduced: the output row pairs the seed prompt with an answer to a different question. Every training pair is semantically misaligned. | Keep `evolved_instruction` as the instruction, using `output_mappings` or a rename step. |
| HIGH | `training/lora_trainer.py:98` | `FastLanguageModel.get_peft_config` does not exist in unsloth 2024.12.7 or in 2026.9.14 (checked in the wheels; the API is `get_peft_model(model, r=..., ...)`), so LoRA training always crashes. The tests mock this non-existent API. The `tokenizer=`, `dataset_text_field=` and `max_seq_length=` kwargs are deprecated in TRL 0.12 and removed in TRL 1.x. `fp16=True` is hard-coded, but Ampere+ GPUs load in bf16. | `model = FastLanguageModel.get_peft_model(model, r=rank, ...)`, `SFTTrainer(processing_class=tok, args=SFTConfig(...))`, and auto-select bf16/fp16. |
| HIGH | `training/lora_trainer.py:79`, `orchestrator.py:479` | LoRA training always uses the Alpaca formatter. With ShareGPT, ChatML or OpenAI output, `examples["instruction"]` raises KeyError. The adapter is written to a CWD-relative `trained_adapter/` and is never offered to the user. | Train from canonical records with `tokenizer.apply_chat_template`, write to the run directory and offer it for download. |
| HIGH | `orchestrator.py:82-153` | `score_dataset` reads only `instruction`/`output`. Every non-Alpaca dataset is graded **BAD with 0 avg length** (reproduced). | Score canonical records, not formatted output. |
| HIGH | `orchestrator.py:356-374,467-475`, `app.py:317` | **"Resume support" does not work.** (a) The checkpoint lives in a `TemporaryDirectory` that is deleted after every run. (b) It is written only *after* the whole pipeline succeeds, so a crash saves nothing. (c) A partial resume overwrites `final_path` with only the new rows, losing earlier data. (d) The all-done path returns a path that may no longer exist. | Use a persistent run directory and per-batch append with a hash ledger, or remove the feature and the README claim. distilabel's own `use_cache=True` already provides resumable runs. |
| MEDIUM | `orchestrator.py:347-352`, `app.py:120` | "Target Dataset Size" only truncates: one pair per 800-char chunk, so a 20-page PDF produces about 60 rows whatever the slider says. | Generate N questions per chunk (a taxonomy of question types) until the target is reached, and show the expected yield before the run. |
| MEDIUM | `app.py:232-255` vs `config.py:84-148` vs `publish/hf_publisher.py:13` | Validation logic and `_REPO_NAME_RE` are duplicated three times and have already drifted: the app does not split comma lists or check Windows drive letters. | Build `DistillationConfig` in the UI and render `ValidationError`s, so there is one source of truth. |
| MEDIUM | `config.py:50,59-62` | `judge_model`, `temperature`, `max_new_tokens`, `batch_size` and `lora_rank` are never exposed in the UI, although the README says they are "tweakable". `judge_model` is never used anywhere. | Expose them in an "Advanced" expander and implement judge filtering. |
| MEDIUM | README | Claims "Automatic refusal cleaning" (no code exists), "Resume support" (broken) and "full CI/CD" (CI never runs). | Implement the features or correct the claims. |

### Code Quality [5/10]

| Sev | Location | Issue | Recommendation |
|---|---|---|---|
| MEDIUM | `app.py` (414 lines) | A single top-level script with no functions mixes UI, validation, PDF parsing, the pricing table and the result rendering. Coverage is 0 % and it cannot be unit-tested. | Split into `ui/` components plus a pure `services/` layer, and test with `streamlit.testing.v1.AppTest`. |
| MEDIUM | many files | Change-log comments such as `FIX C-01`, `FIX M-08` and `Enhancement 7`, plus the ⚡ notes, refer to tickets that are not in the repo. They are noise for readers. | Move the history to CHANGELOG and the PRs, and keep comments about *why*. |
| LOW | `app.py:160-167` | A hard-coded "March 2026" price table with substring matching (`"gpt-4o"` matches `gpt-4o-mini` first only by dict order). | Use exact-match lookup from a dated data file and mark estimates as approximate. |
| LOW | `pipeline/document_loader.py` | Chunking counts characters, not tokens, and semantic overlap can split mid-word. | Use token-aware splitting (`from_tiktoken_encoder`). |

### Dependencies [2/10]

| Sev | Issue | Recommendation |
|---|---|---|
| CRITICAL | The dependency set cannot be installed (see Build). | Lock with uv. |
| HIGH | 60+ known advisories (see Security). The release candidate `transformers==5.0.0rc3` is pinned in production. | Upgrade, then run pip-audit in CI. |
| HIGH | **distilabel is effectively unmaintained upstream.** The last PyPI release is 1.5.3 (2025-01-28), and the GitHub README says the original authors have moved on and community collaborators maintain it. `distilabel.llms` is deprecated, with removal planned for 1.7.0. | Isolate distilabel behind an interface. Consider replacing it with an async OpenAI-compatible client (vLLM, OpenAI, Ollama and most providers expose this API) or NVIDIA NeMo Data Designer (`pip install data-designer`). |
| MEDIUM | Declared but never imported: `tiktoken`, `tenacity`, `tqdm`, `pandas`, `numpy`, `rich`, `openai`, `sentencepiece`, `accelerate`, `bitsandbytes`. Some are transitive; declaring them only adds pin conflicts. | Declare only direct imports, and use extras for GPU and training packages. |
| MEDIUM | All GPU packages (about 10 GB) are mandatory even in OpenAI-only mode, and macOS cannot install vllm. | Make `[vllm]` and `[train]` optional extras. |

### Dead Code [5/10]

| Sev | Location | Issue |
|---|---|---|
| LOW | `old/*.orig` | Tracked in git even though `.gitignore` lists `old/`. Delete it. |
| LOW | `config.py:50` | `judge_model` is never used. |
| LOW | `pipeline/exporter.py:270` | `export_alpaca` legacy alias with no callers. |
| LOW | `tests/conftest.py:52,60` | Unused `MockStep`, plus stubs for `FilterRows` and `RenameColumns`, which do not exist. |
| LOW | `orchestrator.py:33-41` | `_SANITIZER_REQUIRE_FIELDS` entries for 3 formats never work (see Security). |

### Observability [4/10]

| Sev | Issue | Recommendation |
|---|---|---|
| MEDIUM | `pipeline/exporter.py` and `pipeline/sanitizer.py` use stdlib `logging`, which is never configured, so dedup and sanitizer INFO logs are dropped. structlog is configured only in `app.py`. | Configure logging once (structlog's stdlib integration) in a `logging_setup.py`. |
| MEDIUM | No per-run report: no token usage, actual cost, duration per stage, filter and rejection counts, or model/seed provenance. | Write `run_manifest.json` per run, show it in the UI and embed it in the HF dataset card. |
| LOW | Failures surface as `st.error(f"Generation failed: {e}")` with raw exception text. | Map errors to user-actionable messages, with a correlation id that points to the log. |

### Concurrency [4/10]

| Sev | Issue | Recommendation |
|---|---|---|
| HIGH | Each Streamlit session runs the whole pipeline in its script thread. Two users in vLLM mode load two copies of the model, causing GPU OOM. | Add a single-GPU job queue or lock. Better: run vLLM as a separate OpenAI-compatible server shared by all jobs. |
| MEDIUM | `trained_adapter/` is a CWD-relative path shared by all sessions and runs, so concurrent or successive runs overwrite each other. | Use per-run directories (`runs/<uuid>/`). |
| LOW | `Pipeline(name="brainbrew")` uses a fixed name, so all runs share one distilabel cache namespace. | Include the run id in the name. |

### Lifecycle [3/10]

| Sev | Issue | Recommendation |
|---|---|---|
| HIGH | Results exist only inside the `with TemporaryDirectory()` block in `app.py:275`. Any rerun (any widget interaction; the download button reruns by default) erases the quality report and preview. A multi-hour run's dataset survives only as bytes in one download button. | Persist the run directory, keep results in `st.session_state`, and use `download_button(on_click="ignore")`. |
| MEDIUM | Running jobs cannot be cancelled or survive a page refresh. | Background worker plus `@st.fragment(run_every=...)` progress polling, a cancel button and a run history page. |
| LOW | vLLM GPU memory is not released between runs in the same process. | Running vLLM as a separate server process fixes this. |

### Testing (cross-cutting)

- **Mock drift (HIGH).** All 255 tests pass, yet the core pipeline, the LoRA trainer and 3 of 4 sanitizer paths are broken. `conftest.py` replaces distilabel, unsloth and trl with `MagicMock`, and `test_lora_trainer.py` asserts the non-existent `get_peft_config`. Add **contract tests** that build the real distilabel DAG with a dummy `LLM` on CPU (shown to be feasible during this audit), plus format-matrix tests (4 formats × sanitize/score/train).
- `app.py` has 0 % coverage. Use `streamlit.testing.v1.AppTest`.
- Replace the `pytest.raises(Exception)` blind asserts with specific exceptions.

---

## Advisory Findings

- `[High cohesion module]` `pipeline/sanitizer.py` (464 lines) has a single responsibility, a unified data model and a consistent abstraction level. Not a god-module.
- `[Planned: README roadmap]` RAG retrieval and multi-modal support are listed as ideas, so they are not scored as gaps.
- Near-dup dedup is O(N²) with good pruning. It is acceptable up to about 20k records. Above that, MinHash-LSH is the standard approach (advisory).

---

## Recommended Actions (Priority Order)

The phased execution plan is in [`sota_roadmap.md`](./sota_roadmap.md).

1. [CRITICAL] Stop echoing server secrets to the browser, and add authentication or bind to localhost.
2. [CRITICAL] Make the project installable: `pyproject.toml` extras plus `uv.lock`, patched versions.
3. [CRITICAL] Fix the Dockerfile (24.04 base, uv venv, `.dockerignore`).
4. [CRITICAL] Move CI to `.github/workflows/` so it runs.
5. [CRITICAL] Fix distilabel integration (`generation_kwargs`, `StepInput`, keep `evolved_instruction`).
6. [HIGH] Fix LoRA (`get_peft_model`, `SFTConfig`, chat templates for all formats).
7. [HIGH] Canonical-record architecture: sanitize, score and train on `{instruction,input,output}`, and format only at export.
8. [HIGH] Persistent run directories, session state, a job queue, and a real or removed resume feature.
9. [HIGH] Contract tests against real libraries, a format-matrix test suite, `AppTest` for the UI.
10. [MEDIUM] SOTA generation: grounded multi-question generation, LLM-as-judge filtering, semantic dedup, dataset card with provenance.

---

## Sources Consulted

- Local reproduction: pytest + coverage, ruff, mypy, bandit, pip-audit, `uv pip compile`, real `distilabel[openai]==1.5.2` with a dummy LLM, wheel inspection of unsloth 2024.12.7 / 2026.9.14 and trl 0.12.2 / 1.14.1.
- PyPI JSON API (latest versions and `requires_dist` for every dependency).
- [distilabel GitHub (maintenance notice)](https://github.com/argilla-io/distilabel)
- [GHSA-wf5f-4jwr-ppcp / CVE-2025-64512 (pdfminer.six)](https://github.com/advisories/GHSA-wf5f-4jwr-ppcp)
- [packages.ubuntu.com: python3.12 not in jammy](https://packages.ubuntu.com/jammy/python3.12)
- [Streamlit forum: password text_input value visible in DOM](https://discuss.streamlit.io/t/text-input-type-password-is-leaking-in-the-dom/52322)
- Streamlit docs via Context7: `download_button(on_click="ignore")`, `st.session_state`, `st.fragment(run_every=)`, `st.login` OIDC.
- [NVIDIA NeMo Data Designer docs](https://docs.nvidia.com/nemo/datadesigner/latest/getting-started/welcome.md)
- [Synthetic data generation with LLMs, 2026 guide (knowledge-base-anchored generation, judge filtering)](https://futureagi.com/blog/definitive-guide-synthetic-data-generation-2026/)
- [LLM-Synthetic-Data reading list](https://github.com/pengr/LLM-Synthetic-Data)
