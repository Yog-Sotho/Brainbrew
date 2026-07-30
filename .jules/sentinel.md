## 2026-07-28 - Hugging Face Token Leakage & Repository Misconfiguration
**Vulnerability:** Hugging Face API tokens entered via UI were ignored and resolved from environment variables. Additionally, there was no protection against logging `hf_token` if added to `DistillationConfig`, and no configuration-time validation of repository name formats.
**Learning:** Credentials and repository name validation must occur early at initialization (configuration-time) in `DistillationConfig` to prevent late-stage execution failures and accidental logging or leakage of sensitive API tokens in stdout, logs, or error stack traces.
**Prevention:** Always validate format and presence of all credentials and target repository name schemas in Pydantic config schemas, and apply robust `safe_dict` / redaction filters so that all credentials are consistently redacted.

## 2026-07-28 - Incomplete Input Validation on Model/Path Configs
**Vulnerability:** Unvalidated configuration parameters (`judge_model` and `checkpoint_dir`) allowed potential directory traversal, log pollution, or arbitrary local path references when using orchestrator configs directly.
**Learning:** Validation decorators must explicitly cover all string settings in Pydantic configurations that represent model names or directory paths, as omitted fields do not inherit validation automatically.
**Prevention:** Ensure any new string fields representing paths, models, or system-level identifiers are validated against path traversal (`..`), control character patterns, and length limits.
