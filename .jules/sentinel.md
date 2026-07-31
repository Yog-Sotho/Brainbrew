## 2026-07-28 - Hugging Face Token Leakage & Repository Misconfiguration
**Vulnerability:** Hugging Face API tokens entered via UI were ignored and resolved from environment variables. Additionally, there was no protection against logging `hf_token` if added to `DistillationConfig`, and no configuration-time validation of repository name formats.
**Learning:** Credentials and repository name validation must occur early at initialization (configuration-time) in `DistillationConfig` to prevent late-stage execution failures and accidental logging or leakage of sensitive API tokens in stdout, logs, or error stack traces.
**Prevention:** Always validate format and presence of all credentials and target repository name schemas in Pydantic config schemas, and apply robust `safe_dict` / redaction filters so that all credentials are consistently redacted.

## 2026-07-28 - Incomplete Input Validation on Model/Path Configs
**Vulnerability:** Unvalidated configuration parameters (`judge_model` and `checkpoint_dir`) allowed potential directory traversal, log pollution, or arbitrary local path references when using orchestrator configs directly.
**Learning:** Validation decorators must explicitly cover all string settings in Pydantic configurations that represent model names or directory paths, as omitted fields do not inherit validation automatically.
**Prevention:** Ensure any new string fields representing paths, models, or system-level identifiers are validated against path traversal (`..`), control character patterns, and length limits.

## 2026-07-29 - Path Traversal Vulnerability via Loose Repository Validation Regex
**Vulnerability:** Loose validation regex for Hugging Face repository names (`^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$`) successfully matched sequences containing double dots (`..`), such as `user/..` or `user/../repo`. If resolved or used inside caching/file system interactions, this would allow malicious actors to perform directory traversal attacks.
**Learning:** Standard regex validation of structured repository/package identifiers often forgets to exclude directory traversal patterns like `..` when dots are permissible in names.
**Prevention:** Explicitly block `..` sequences in repository identifiers during both configuration-time validation and publication execution steps, as defense in depth.
