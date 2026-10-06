# Configuration reference

Every run is described by `DistillationConfig` (`config.py`). The web app builds it from the
form; the CLI from a YAML/JSON file plus options. Invalid values are reported with the field name.

This page is checked against the model by `tests/test_docs.py`.

| Field | Type | Default | Allowed | Meaning |
|---|---|---|---|---|
| `teacher_model` | `str` | `required` |  | Model name as the endpoint knows it; comma-separate several for an ensemble. |
| `judge_model` | `str \| None` | `None` |  | Grades pairs in Balanced and Research mode. Default: the first teacher. |
| `judge_threshold` | `int` | `4` | ≥ 1 ≤ 5 | Minimum judge score (1-5) that faithfulness, helpfulness and correctness must all reach. |
| `embedding_model` | `str \| None` | `None` |  | Embedding model on the same endpoint; enables paraphrase removal. |
| `semantic_dedup_threshold` | `float` | `0.92` | ≥ 0.5 ≤ 0.999 | Cosine similarity at which a pair counts as a paraphrase of an accepted one. |
| `base_url` | `str \| None` | `None` |  | OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1`. Unset: OpenAI. |
| `dataset_size` | `int` | `500` | ≥ 10 ≤ 50000 | Target number of pairs. |
| `quality_mode` | `QualityMode` | `'balanced'` |  | `fast` (filters), `balanced` (+ judge) or `research` (+ evolved questions). |
| `output_format` | `OutputFormat` | `'alpaca'` |  | `alpaca`, `sharegpt`, `chatml` or `openai`. |
| `train_model` | `bool` | `False` |  | Train a LoRA adapter after export (needs the `train` extra). |
| `base_model` | `str` | `'Qwen/Qwen3-4B-Instruct-2507'` |  | Hugging Face model id to fine-tune. |
| `publish_dataset` | `bool` | `False` |  | Upload the dataset to the Hugging Face Hub with a dataset card. |
| `hf_repo` | `str \| None` | `None` |  | Dataset repo, `user/name`. |
| `hf_private` | `bool` | `True` |  | Create repos as private. |
| `dataset_license` | `one of ['other', 'unknown', 'cc-by-4.0', 'cc-by-sa-4.0', 'cc-by-nc-4.0', 'cc0-1.0', 'odc-by', 'mit', 'apache-2.0']` | `'other'` |  | License written on the dataset and model cards. |
| `publish_adapter` | `bool` | `False` |  | Also upload the LoRA adapter as a model repo (needs `train_model`). |
| `hf_model_repo` | `str \| None` | `None` |  | Adapter repo. Default: `<hf_repo>-lora`. |
| `temperature` | `float` | `0.7` | ≥ 0.0 ≤ 2.0 | Sampling temperature for questions and answers (the judge always uses 0). |
| `max_new_tokens` | `int` | `2048` | ≥ 128 ≤ 32768 | Maximum tokens per answer. |
| `concurrency` | `int` | `8` | ≥ 1 ≤ 64 | Parallel requests per model client. Use 1-2 for slow CPU servers. |
| `request_timeout` | `int` | `120` | ≥ 10 ≤ 1800 | Seconds before a request is retried. |
| `seed` | `int \| None` | `None` | ≥ 0 ≤ 2147483647 | Sampling seed sent with every request. Unset: a random seed is chosen and recorded. A server that rejects the field gets requests without it (logged). |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high` or `None` | `None` | | Thinking effort for models that think. `none` turns thinking off where the server allows it (Gemini 2.5 Flash), so it does not use up the answer's token budget. Unset: the server's default. |
| `lora_rank` | `int` | `16` | ≥ 4 ≤ 256 | LoRA rank `r` (alpha is set equal). |
| `api_key` | `str \| None` | `None` |  | Key for the endpoint. Never written to manifests or logs. CLI: environment only. |
| `hf_token` | `str \| None` | `None` |  | Hugging Face write token. Same rules as `api_key`. |
| `use_semantic_chunking` | `bool` | `False` |  | Paragraph and sentence-aware chunks instead of fixed windows. |
| `enable_dedup` | `bool` | `True` |  | Remove near-duplicates (and paraphrases when an embedding model is set). |
| `sanitize_dataset` | `bool` | `False` |  | Redact PII, strip HTML and apply quality gates before export. |
| `pii_url_policy` | `one of ['domain', 'redact', 'keep']` | `'domain'` |  | `domain` keeps scheme and host, `redact` removes links, `keep` keeps them. Credentials and secret query values are always removed. |
| `pii_presidio` | `bool` | `False` |  | Also detect names and ID numbers with Presidio (`pii` extra). |
| `decontaminate` | `list[str]` | `[]` |  | Benchmarks to remove overlap with: `gsm8k`, `mmlu`, `arc`, `truthfulqa`, `humaneval`. |

## Rules across fields

- The OpenAI API (no `base_url`) needs an `api_key`; local servers do not.
- Publishing needs `hf_repo` and `hf_token`.
- `publish_adapter` needs `train_model`, a repo (`hf_model_repo` or `hf_repo`) and a token.

## Environment

See the operator table in the [README](https://github.com/Yog-Sotho/Brainbrew#operator-settings-environment):
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `HF_TOKEN`, `BRAINBREW_ALLOW_CUSTOM_ENDPOINTS`,
`BRAINBREW_DEFAULT_MODEL`, `BRAINBREW_MAX_JOBS`, `BRAINBREW_RUNS_DIR`,
`BRAINBREW_LOG_FORMAT`, `BRAINBREW_LOG_LEVEL`, `BRAINBREW_REQUIRE_LOGIN`.

