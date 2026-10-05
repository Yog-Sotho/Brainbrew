# Brainbrew: Roadmap from 3/10 to ≥ 8/10

Source: [`codebase_audit.md`](./codebase_audit.md) (2026-10-05). The phases are ordered by dependency. Each phase ends with acceptance criteria that can be checked mechanically. **Do not start a later phase until the current phase's gate passes.**

## Target scores

| Dimension | Now | After P0 | After P1 | After P2 | After P3–P4 |
|---|---|---|---|---|---|
| Security | 3 | 6 | 7 | 7 | **8.5** |
| Build & Types | 1 | 7 | 8 | 8 | **9** |
| Code Principles | 2 | 2 | 7 | 8 | **8.5** |
| Code Quality | 5 | 5 | 6 | 7 | **8** |
| Dependencies | 2 | 7 | 7 | 8 | **8.5** |
| Dead Code | 5 | 7 | 8 | 8 | **9** |
| Observability | 4 | 4 | 5 | 6 | **8** |
| Concurrency | 4 | 4 | 5 | 6 | **8** |
| Lifecycle | 3 | 3 | 5 | 6 | **8** |
| **Overall** | **3.2** | ~5.0 | ~6.4 | ~7.1 | **≈ 8.4** |

## Key decision: the generation engine

distilabel's last release was 1.5.3 (Jan 2025), and its README says the original authors have moved on. Options:

- **A. Keep distilabel 1.5.x and fix the integration.** Least work, but it pins the core to an aging library, and `distilabel.llms` is due for removal.
- **B (recommended). A thin internal async engine over the OpenAI-compatible API.** vLLM (`vllm serve`), OpenAI, Ollama and most hosted providers expose this API. That gives one code path, real concurrency, retries (`tenacity`, already a dependency) and no mandatory GPU packages for API users. vLLM runs as a separate server process, which also fixes the concurrency and GPU-memory findings.
- **C. NVIDIA NeMo Data Designer** (`pip install data-designer`). Mature column and judge orchestration, but a heavier dependency with its own concepts.

Phase 1 fixes distilabel *in place* (option A) so the product works quickly. Phase 2 moves to option B behind the same `Generator` interface.

---

## Phase 0: Installable, secure baseline (about 1–2 days)

| # | Task | Files |
|---|---|---|
| 0.1 | Stop sending server secrets to the browser. Resolve keys server-side (env / `st.secrets`), show "configured ✓", and do not pre-fill password widgets. | `app.py` |
| 0.2 | Bind to `127.0.0.1` by default. Add an optional `st.login` (OIDC) gate when `BRAINBREW_AUTH=1`. Add `.streamlit/config.toml` with `maxUploadSize=50`. | `app.py`, `.streamlit/config.toml`, `Dockerfile` |
| 0.3 | Packaging: put PEP 621 `dependencies` (core only) in `pyproject.toml`, add `[project.optional-dependencies] vllm=[...]`, `train=[...]`, `dev=[...]`, and run `uv lock`. Delete `setup.py`, `poetry.lock` and `old/`. Turn `requirements.txt` into an exported lock (`uv export --format requirements-txt`) or remove it. | packaging |
| 0.4 | Upgrade to patched versions, resolved together: pdfminer.six ≥ 20251107, python-dotenv ≥ 1.2.2, langchain-text-splitters ≥ 1.1.2, datasets ≥ 5.0.1, transformers ≥ 5.10 (stable), vllm ≥ 0.30, and pydantic and openai as vllm requires. Choose unsloth and trl versions compatible with that transformers. | lock |
| 0.5 | Dockerfile: CUDA **Ubuntu 24.04** base, a uv-managed venv, multi-stage build, `.dockerignore` (`.env*`, `.git`, `old`, `docs`, `tests`, `*.jsonl`, `.venv`), a separate slim `python:3.12` target for API-only use, and no secrets in layers. | `Dockerfile`, `.dockerignore` |
| 0.6 | Move CI to `.github/workflows/ci.yml`. Jobs: ruff (lint + format check), mypy (pydantic plugin, all modules), pytest with a coverage gate, `pip-audit`, gitleaks, and `docker build` (API target). Add `.github/dependabot.yml` (pip + actions + docker). | `.github/` |
| 0.7 | Fix the existing 12 ruff and 5 mypy errors. Use `StrEnum`. | various |

**Gate:** `uv sync` succeeds on a clean py3.12 box; `docker build` succeeds; CI is green on a PR; pip-audit reports 0 known vulns with fixes; there is no secret in the page HTML (checked with an `AppTest`).

### Phase 0 status (2026-10-05): done, pending first CI run on GitHub

| Gate item | Result |
|---|---|
| Resolvable, locked deps | `uv lock`: 290 packages; `vllm` ⟂ `train` declared as conflicting extras. Core: streamlit 1.65, pydantic 2.13, distilabel 1.5.3, pdfminer.six 20260107. vLLM: vllm 0.31.0 / torch 2.13.0+cu130. |
| Runtime compatibility | Real distilabel 1.5.3 pipeline (EvolInstruct → TextGeneration) run against a fake OpenAI-compatible server with openai 3.24 / datasets 5.0.1. |
| Docker | `api` target built, runs as uid 10001, `/_stcore/health` ok and Docker health `healthy`; decoy `.env` and `secrets.toml` in the build dir were absent from the image. `gpu-builder` stage built on Ubuntu 24.04 and `import vllm` works. The final `gpu` stage was not built locally because of sandbox disk limits; it is the same `COPY` pattern as `api`. |
| Secrets to browser | `tests/test_app_secrets.py` (AppTest) fails on the old `app.py` and passes on the fix. |
| Network exposure | Server binds 127.0.0.1 (verified: loopback ok, non-loopback refused). Optional OIDC gate (`BRAINBREW_REQUIRE_LOGIN=1`) fails closed without `[auth]` config. |
| CI | `.github/workflows/ci.yml` passes actionlint, with SHA-pinned actions. Locally: ruff ✓, mypy ✓ (pydantic plugin), 260 tests ✓ (cov 76 %, gate 75 %), lock/exports in sync ✓, pip-audit ✓, gitleaks ✓ (history + tree). |

**Accepted risk, carried into Phase 1 (resolved in Phase 1, see below):** the `train` extra is capped by Unsloth (torch < 2.13, transformers ≤ 5.5, datasets < 4.4, trl ≤ 0.24). Four advisories therefore remain in that stack only, and CI ignores them for that stack: PYSEC-2025-194, PYSEC-2026-3929, PYSEC-2026-4174 and PYSEC-2026-3716. setuptools CVE-2026-59890 (build-time sdist handling) is capped by vllm (< 81) and ignored everywhere. **Add to Phase 1.3:** move LoRA training to plain TRL + PEFT (no Unsloth pin), which lifts these caps.

## Phase 1: Make it correct (about 3–4 days)

| # | Task |
|---|---|
| 1.1 | **Canonical record model.** Add `Record(instruction, input, output, meta)` (pydantic). Every stage (dedup, sanitize, score, train) works on canonical JSONL. Formatting to Alpaca, ShareGPT, ChatML or OpenAI happens only in the final `export`. This fixes the sanitizer, scorer and LoRA format bugs at the root. |
| 1.2 | distilabel fixes: import from `distilabel.models`, use `generation_kwargs={...}`, annotate the custom step with `StepInput`/`StepOutput`, keep **`evolved_instruction`** as the instruction, and use a per-run `Pipeline(name=f"brainbrew-{run_id}")`. |
| 1.3 | LoRA: `FastLanguageModel.get_peft_model(model, r=rank, lora_alpha=rank, target_modules=[...])`, `SFTTrainer(processing_class=tok, args=SFTConfig(...))`, `tokenizer.apply_chat_template` for messages, auto bf16/fp16, and output to `runs/<id>/adapter/` with a zip download. |
| 1.4 | Persistent **run directory** `runs/<uuid>/` (`source.txt`, `raw.jsonl`, `dataset.<fmt>.jsonl`, `manifest.json`, `adapter/`). Keep the UI state in `st.session_state`. Use `download_button(on_click="ignore")`. |
| 1.5 | Resume: enable distilabel `use_cache=True` on the persistent run directory with a "Resume run" button, **or** remove the feature and the README claim. No half-working version. |
| 1.6 | One validation source: the UI builds `DistillationConfig` and renders `ValidationError`s. Remove the duplicated regexes in `app.py` and `hf_publisher.py` (import from `config`). |
| 1.7 | Expose `temperature`, `max_new_tokens`, `batch_size` and `lora_rank` in an "Advanced" expander. Hide `judge_model` until Phase 2 implements it. |
| 1.8 | **Tests that would have caught these bugs:** (a) a contract test that builds and runs the *real* distilabel DAG with a dummy `LLM` on CPU, asserting the instruction equals the evolved instruction; (b) a 4-format matrix for export → sanitize → score → train-format; (c) `streamlit.testing.v1.AppTest` smoke tests for `app.py`; (d) delete the mocks of non-existent APIs. |
| 1.9 | README: remove false claims (refusal cleaning, resume, CI), fix the test count and document the extras-based install. |

**Gate:** an end-to-end run against a real OpenAI-compatible endpoint (e.g. `vllm serve` with a small model, or OpenAI `gpt-4o-mini`) on a 5-page PDF produces aligned pairs in all 4 formats. The sanitizer keeps more than 0 rows in every format, the scorer gives the same grade across formats, and coverage including `app.py` is ≥ 80 %.

### Phase 1 status (2026-10-05): done

| # | Result |
|---|---|
| 1.1 | `pipeline/records.py`: `Record(instruction, input, output, meta)`. Dedup, sanitize, scoring and training work on canonical records; `pipeline/exporter.py` only formats. Sanitize now keeps records in all 4 formats (the audit reproduced 0/30 for ShareGPT/ChatML/OpenAI), and an empty sanitize result is an error instead of a silent fallback to unsanitized data. |
| 1.2 | `distilabel.models`, `generation_kwargs`, `StepInput` step (in `pipeline/steps.py`, which must not use postponed annotations or distilabel cannot see the hint), evolved instruction kept with the seed in `meta`, per-run pipeline name and cache dir. |
| 1.3 | Unsloth replaced by TRL 1.14 + PEFT 0.21: conversational prompt-completion data with the model's chat template (Alpaca text for base models), loss on answers only, `target_modules="all-linear"`, auto bf16/fp16, 4-bit QLoRA on CUDA. Default base model `Qwen/Qwen3-4B-Instruct-2507`. With Unsloth gone, `vllm` and `train` resolve together (one torch 2.13), and the four training-stack advisories accepted in Phase 0 are fixed. |
| 1.4 | `runs/<run-id>/` with `source.txt`, `raw.jsonl`, `records.jsonl`, `dataset.<fmt>.jsonl`, `manifest.json` (config without secrets, status, counts, sanitizer stats, quality, error), `adapter/` + `adapter.zip`. Results render from `st.session_state["run_id"]` on every rerun; downloads use `on_click="ignore"`. |
| 1.5 | **Resume removed.** distilabel 1.5.3's `use_cache=True` lost rows in every interrupted-run test (12/20 kept after SIGTERM, 16/20 after SIGKILL) and hangs forever if a worker process dies. Checkpoint code, `checkpoint_dir` and the README claim are gone. |
| 1.6 | The UI builds `DistillationConfig` and renders its `ValidationError`s; cross-field rules (API key without vLLM, HF token + repo when publishing) live in the config. The HF repo rule is shared with the publisher. The filename allow-list is gone (names were never used as paths). |
| 1.7 | Sidebar "Generation settings" (temperature, max answer length, batch size); base model and LoRA rank appear when auto-train is on. `judge_model` removed until Phase 2. |
| 1.8 | Tests run the real libraries: the distilabel DAG with an offline `FakeLLM`, a 4-format matrix, `AppTest` flows through the real `app.py` (upload → generate → results survive reruns), and real TRL + PEFT training on tiny Hub models (new `train-contract` CI job, CPU torch). The test that copied `app.py`'s validation logic was deleted. Coverage 91 % (gate: 80 %). |
| 1.9 | README: no resume / refusal-cleaning claims, TRL + PEFT, run folders, `OPENAI_BASE_URL`, the one-pair-per-chunk cap. |

**End-to-end gate:** in progress at the time of this commit (real UI in headless Chromium → Streamlit → OpenAI-compatible server running `Qwen/Qwen2.5-0.5B-Instruct` on CPU, 5-page PDF); results follow in the next commit.

**Found during Phase 1 (not in the audit):**

- **Generation could never run from the UI.** `Pipeline.run` installs a SIGINT handler, which Python allows only on the main thread, and Streamlit runs scripts on worker threads. Generation now runs in a spawned child process (`pipeline/generation.py`), which also keeps distilabel's process-global side effects (it replaces the root logger's handlers and leaves a closed `QueueHandler` behind) out of the server.
- The sanitizer turned every newline into a space, flattening lists, paragraphs and code in answers. Line structure is now preserved.
- `re.match(r"^...$")` accepted a trailing newline in run ids and HF repo names; `fullmatch` everywhere.
- transformers 5 removed `warmup_ratio` (`warmup_steps` takes a ratio).

**Carried forward:**

- **2.1:** with a slow OpenAI-compatible server (e.g. a CPU-only local model), distilabel's 120 s timeout plus the default batch size of 64 makes most requests time out. A batch size of 1–4 works today; the new engine should expose timeout and concurrency.
- **2.1:** a dead distilabel worker hangs the pipeline forever; ~50k Pydantic deprecation warnings per run come from distilabel (silenced in tests only).
- **2.8:** one pair per chunk, so "Target dataset size" is an upper bound.
- **3.1:** interrupted runs cannot be resumed; the job runner should add a per-batch ledger in the run directory. A run keeps going if the browser is closed (the script thread waits for the child).

## Phase 2: State-of-the-art generation quality (about 1–2 weeks)

Each of these is supported by current practice (see Sources in the audit):

| # | Task |
|---|---|
| 2.1 | **Engine swap (option B):** an async OpenAI-compatible client with bounded concurrency, `tenacity` retries and backoff, and structured outputs (JSON schema). vLLM becomes `vllm serve` (a sidecar container or `install.sh` launcher). distilabel becomes an optional extra or is removed. |
| 2.2 | **Knowledge-base-anchored generation:** for each token-aware chunk, generate *k* diverse questions across a taxonomy (factual, conceptual, procedural, comparative, multi-hop across adjacent chunks). Answer each with the source chunk in context. Evol-Instruct becomes an optional complexity step with an "is it still answerable from the source?" check. |
| 2.3 | **LLM-as-judge filtering** (finally using `judge_model`): score faithfulness to the source, helpfulness and correctness on a 1–5 rubric with a structured output, and keep rows at or above a configurable threshold. Store scores in `meta`. |
| 2.4 | **Refusal and boilerplate filter:** pattern-based and judge-based removal of "As an AI…" answers and empty answers (this fulfils the README claim). |
| 2.5 | **Dedup v2:** MinHash-LSH for near-duplicates (O(N)), plus optional embedding-based semantic dedup. |
| 2.6 | **Decontamination (optional):** n-gram overlap check against common eval sets selected by the user. |
| 2.7 | **PII v2:** Luhn and IP-context validation, a configurable URL policy, and an optional Presidio backend. |
| 2.8 | Make "Target dataset size" real: generate until the target is reached or the source is exhausted, and show the expected yield before the run. |

**Gate:** on a fixed benchmark corpus checked into `tests/fixtures/` (e.g. 3 public-domain PDFs), judge-faithfulness is ≥ 4.0 on average, the near-dup rate is < 2 %, the refusal rate is 0 and yield is within ±10 % of the target. Track these in a `bench/` script that CI runs nightly with a cheap model.

## Phase 3: Operability (about 1 week)

| # | Task |
|---|---|
| 3.1 | A background job runner (thread or process pool with one GPU slot), progress through `@st.fragment(run_every="2s")`, a cancel button and a run-history page. |
| 3.2 | Logging configured once (structlog plus stdlib integration, JSON in containers). Per-run `manifest.json` with config (secrets redacted), models, seed, token usage, actual cost, stage timings and filter counts. |
| 3.3 | HF publish: an auto-generated **dataset card** (provenance, models, license, filters, stats), `private=True` by default, and an optional upload of the adapter as a model repo. |
| 3.4 | A headless CLI (`brainbrew run --config cfg.yaml docs/*.pdf`) through Typer, sharing the services layer. This makes the `pyproject` console script real. |

**Gate:** two concurrent UI sessions plus a CLI run complete without GPU OOM or file clobbering; a page refresh does not lose a running job; the manifest is present for every run.

## Phase 4: Hardening and polish (about 3–5 days)

- `mypy --strict` on `config`, `pipeline/` and `engine/`; ruff with the `S` (bandit), `PT` and `RUF` rule sets; pre-commit.
- Mutation testing (`mutmut`) on the sanitizer and dedup. Property tests (Hypothesis) for chunkers and formatters.
- SBOM (`cyclonedx-py`) and image scanning (Trivy) in CI. Signed releases.
- Replace the 13 binary PDFs in `docs/` with Markdown and a docs site, so documentation is reviewable in diffs.
- Remove the change-log comments (`FIX C-xx`, `Enhancement N`) and move them to `CHANGELOG.md`.

**Gate:** a re-run of the codebase audit scores **≥ 8.0 overall with no CRITICAL or HIGH findings open**.

## Suggested PR breakdown

1. `sec/secrets-auth`: 0.1 and 0.2
2. `build/packaging-lock`: 0.3, 0.4 and 0.7
3. `build/docker`: 0.5
4. `ci/actions`: 0.6
5. `fix/distilabel-integration`: 1.2 plus the contract test from 1.8a
6. `refactor/canonical-records`: 1.1 plus the format matrix from 1.8b
7. `fix/lora-trainer`: 1.3
8. `feat/run-dirs-session-state`: 1.4, 1.5 and 1.7
9. `refactor/single-validation` and `test/apptest`: 1.6 and 1.8c
10. Phase 2 onward: one PR per numbered item.
