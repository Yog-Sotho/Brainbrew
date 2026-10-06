# Security model

## Secrets

- API keys and Hugging Face tokens from the environment stay on the server.
  They are never used as widget values (Streamlit sends widget state to the
  browser); the sidebar only says that a server key is configured.
- `DistillationConfig` excludes secrets from `repr`, `str`, manifests
  (`public_dict`) and logs (`safe_dict`). Tests scan every file of a run folder
  and the run log for the key.
- **The server's key only goes to the server's endpoint.** If a visitor picks
  another endpoint, only a key they typed themselves is sent. Otherwise a
  visitor could point the app at their own URL and receive the server's key.
  Operators can lock the endpoint with `BRAINBREW_ALLOW_CUSTOM_ENDPOINTS=0`.
  With login on, it is locked unless the operator sets the variable to `1`:
  on a shared server, a user-chosen URL makes the server send requests into its
  own network.
- The CLI reads secrets only from the environment and refuses config files that
  contain them.
- With login on, the server's Hugging Face token only publishes to the
  operator's own namespace (`HF_USERNAME`). Publishing anywhere else needs the
  user's own token, so one user cannot write to another's repos with the
  server's credentials.

## Network exposure

- The app listens on `127.0.0.1` (`.streamlit/config.toml`). The Docker examples
  publish the port on localhost only; in `compose.yaml` vLLM's port is not
  published at all.
- Before exposing Brainbrew to a network, set `BRAINBREW_REQUIRE_LOGIN=1` and
  configure an OIDC provider under `[auth]` in `.streamlit/secrets.toml`. The
  gate runs on every page, and without the configuration the app refuses to
  start the UI (fail closed).
- Model requests never reach link-local or cloud metadata addresses
  (`169.254.169.254`, `fd00:ec2::254`, `metadata.google.internal` and similar).
  Endpoint URLs naming one are rejected up front. At connect time
  (`engine/netguard.py`) the client resolves the host itself, refuses if any
  answer is such an address, and connects to the address it checked. That
  covers host names that resolve there, DNS answers that change between checks
  (rebinding) and redirects.
- Behind an HTTP proxy (`HTTPS_PROXY`), the proxy resolves the target, so only
  the URL check applies. Block the metadata address at the proxy or in the
  network policy as well.
- With login on, each user sees only their own runs. A run id in a URL is not
  enough to open someone else's documents or dataset.

## Uploaded documents

- Upload size is limited (`maxUploadSize`, 50 MB per file); source text over
  100 MB is rejected.
- Uploaded file names are never used as paths: documents are parsed from
  memory and stored as `runs/<run-id>/source.txt`. Run ids are generated and
  validated against a strict pattern before any path is built.

## Data you publish

- Hugging Face repos are created private unless *Make it public* is ticked.
- Dataset cards never include source document names or text; a SHA-256 of the
  source text identifies it.
- *Clean & sanitize* redacts personal data before export. Redaction is
  pattern-based (plus Presidio, if installed) and cannot catch everything:
  review data from sensitive documents before publishing it.

## Supply chain

- Dependencies are locked in `uv.lock`; the `requirements*.txt` exports are
  hash-pinned. CI runs pip-audit, gitleaks and CodeQL; GitHub Actions and base
  images are pinned to digests and kept current by Dependabot.
- CI scans the Docker image with Trivy and fails on any fixable HIGH or CRITICAL
  vulnerability, and generates CycloneDX SBOMs of the locked dependencies.
- Releases (`vX.Y.Z` tags) ship the wheel, sdist and SBOMs, each signed with
  Sigstore by the release workflow's GitHub identity. Verify a download with:

  ```bash
  uvx sigstore verify github \
    --cert-identity "https://github.com/Yog-Sotho/Brainbrew/.github/workflows/release.yml@refs/tags/vX.Y.Z" \
    brainbrew-X.Y.Z-py3-none-any.whl
  ```
- Images run as a non-root user and never contain `.env` or
  `.streamlit/secrets.toml`.

Report a vulnerability privately through GitHub's security advisories for this
repository rather than in a public issue.
