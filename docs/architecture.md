# Architecture

```
documents ─► service.read_documents ─► run folder (source.txt)
                                           │
             web app ──► JobRunner ────────┤        CLI ──► worker thread
                                           ▼
                                orchestrator.run_distillation
   chunk ─► synth (questions ─► answers ─► filters ─► judge ─► dedup) ─► decontaminate
         ─► sanitize ─► export ─► [train in the GPU slot] ─► [publish with cards]
```

## Modules

| Module | Role |
|---|---|
| `app.py`, `pages/`, `ui/` | Streamlit pages. `ui/common.py` holds page setup, the login gate and per-user run visibility; `ui/sidebar.py` the Generate page's sidebar; `ui/generate.py` its widget-free logic (endpoints, estimates, validation messages); `ui/results.py` the run panel. |
| `cli.py` | The `brainbrew` command line. |
| `config.py` | `DistillationConfig`, the single source of validation for the app and the CLI. |
| `orchestrator.py` | Runs one pipeline in a run folder: stages, progress, cancellation, manifest. |
| `engine/client.py` | Async client for any OpenAI-compatible API: bounded concurrency, retries, structured output that degrades `json_schema` → `json_object` → schema in the prompt and remembers what the server supports. Connections go through `engine/netguard.py`, which refuses cloud metadata addresses after DNS resolution and on redirects. |
| `pipeline/synth.py` | Grounded generation in rounds, with filters, judge and dedup, until the target or the source is exhausted. Selection is deterministic. |
| `pipeline/prompts.py`, `pipeline/filters.py` | Prompts and schemas; pattern filters for questions and answers. |
| `pipeline/dedup.py` | MinHash-LSH near-duplicates and embedding-based paraphrase dedup. |
| `pipeline/decontam.py` | 13-gram overlap with public benchmarks. |
| `pipeline/pii.py`, `pipeline/sanitizer.py` | PII detection and redaction; text cleaning and quality gates. |
| `pipeline/records.py`, `pipeline/exporter.py` | The canonical `Record` and the four export formats. |
| `pipeline/jobs.py`, `pipeline/gpu.py` | The background job runner and the single GPU slot (a file lock shared across processes). |
| `pipeline/runs.py`, `pipeline/logs.py`, `pipeline/pricing.py` | Run folders and manifests, logging, cost. |
| `publish/` | Hugging Face upload and dataset/model cards. |
| `training/lora_trainer.py` | LoRA / QLoRA training with TRL + PEFT, loss on answers only. |
| `bench/` | The generation-quality benchmark and its fixture builder. |

## Design rules

- **Canonical records everywhere.** Every stage works on `Record(instruction,
  input, output, meta)`; formatting happens only at export.
- **The run folder is the source of truth.** The manifest is written at every
  stage, so a run can be followed, audited or reported on from its folder alone.
- **One validation source.** The UI and the CLI build `DistillationConfig` and
  show its errors.
- **Secrets never leave the process.** They are excluded from manifests, logs,
  `repr` and widget state, and only sent to the endpoint they belong to.
