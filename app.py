"""
Brainbrew — Streamlit UI for synthetic dataset generation.

This is the main entry point for the application. Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlsplit

import streamlit as st
import structlog
from dotenv import load_dotenv
from pydantic import ValidationError

from config import (
    DEFAULT_BASE_MODEL,
    OUTPUT_FORMAT_LABELS,
    QUALITY_MODE_LABELS,
    DistillationConfig,
)
from orchestrator import run_distillation
from pipeline.document_loader import read_document, source_chunks
from pipeline.records import Record
from pipeline.runs import RunDir, create_run, open_run
from pipeline.synth import PAIRS_PER_CHUNK_ESTIMATE

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger("INFO"))
logger = structlog.get_logger(__name__)

MAX_WARN_BYTES: int = 10 * 1024 * 1024   # warn at 10 MB
MAX_HARD_BYTES: int = 50 * 1024 * 1024   # hard limit at 50 MB per file

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Brainbrew", page_icon="🧠", layout="wide")

# ── Optional login gate ──────────────────────────────────────────────────────
# Set BRAINBREW_REQUIRE_LOGIN=1 and an [auth] block in .streamlit/secrets.toml
# (OIDC provider) before exposing the app beyond localhost.
def _auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except FileNotFoundError:  # no secrets.toml at all
        return False


if os.getenv("BRAINBREW_REQUIRE_LOGIN", "").strip().lower() in {"1", "true", "yes"}:
    if not _auth_configured():
        # Fail closed: never fall through to an unauthenticated app.
        st.error(
            "BRAINBREW_REQUIRE_LOGIN is set but no [auth] section was found in "
            ".streamlit/secrets.toml. Configure an OIDC provider to continue."
        )
        st.stop()
    if not st.user.is_logged_in:
        st.title("🧠 Brainbrew")
        st.info("This Brainbrew instance is private. Please log in.")
        st.button("Log in", on_click=st.login, type="primary")
        st.stop()
    st.sidebar.button("Log out", on_click=st.logout)

st.title("🧠 Brainbrew v2.0.0")
st.caption("Grounded synthetic dataset generator for any OpenAI-compatible model")

# ── Model endpoint ───────────────────────────────────────────────────────────
# The server's own API key (OPENAI_API_KEY) is only ever sent to the server's
# own endpoint (OPENAI_BASE_URL, or OpenAI when unset). Any other endpoint a
# visitor picks gets only the key that visitor typed, so the server key cannot
# be redirected to a URL the visitor controls.
SERVER_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL", "").strip() or None
DEFAULT_MODEL = os.getenv("BRAINBREW_DEFAULT_MODEL", "").strip() or "gpt-4o-mini"
ALLOW_CUSTOM_ENDPOINTS = os.getenv("BRAINBREW_ALLOW_CUSTOM_ENDPOINTS", "1").strip().lower() not in {"0", "false", "no"}
_CUSTOM = "custom"
_SERVER_ENDPOINT_LABEL = "Server default" if SERVER_BASE_URL else "OpenAI API"
ENDPOINTS: dict[str, str | None] = {_SERVER_ENDPOINT_LABEL: SERVER_BASE_URL}
if ALLOW_CUSTOM_ENDPOINTS:
    ENDPOINTS.update({
        "Local vLLM server (localhost:8000)": "http://localhost:8000/v1",
        "Ollama (localhost:11434)": "http://localhost:11434/v1",
        "Custom URL…": _CUSTOM,
    })

# ── Sidebar: settings ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    endpoint_label: str = st.selectbox(
        "Model endpoint",
        options=list(ENDPOINTS),
        help="Any OpenAI-compatible API: OpenAI, `vllm serve`, Ollama, llama.cpp, or a hosted provider.",
    )
    base_url: str | None = ENDPOINTS[endpoint_label]
    if base_url == _CUSTOM:
        base_url = st.text_input("Endpoint URL", placeholder="https://my-server.example.com/v1") or None
    on_server_endpoint = endpoint_label == _SERVER_ENDPOINT_LABEL

    # Server-side secrets are never used as widget values: Streamlit sends widget
    # state to the browser, so a pre-filled password field discloses the key to
    # every visitor. The env value is applied server-side as a fallback instead.
    openai_env_key = os.getenv("OPENAI_API_KEY", "")
    use_server_key = bool(openai_env_key) and on_server_endpoint
    openai_key: str = st.text_input(
        "API Key",
        type="password",
        placeholder="Using server key" if use_server_key else "sk-...",
        help="Your key for this endpoint. Local servers (vLLM, Ollama) usually need none.",
    )
    if use_server_key:
        st.caption("🔑 *Server API key configured; leave blank to use it*")
    elif openai_env_key:
        st.caption("🔒 *The server's key is only used with its own endpoint. Enter a key if this one needs it.*")
    elif base_url is None:
        st.caption("⚠️ *API Key required for the OpenAI API*")

    hf_env_token = os.getenv("HF_TOKEN", "")
    hf_token: str = st.text_input(
        "Hugging Face Token",
        type="password",
        placeholder="Using server token" if hf_env_token else "hf_...",
        help="Enter your Hugging Face write token. Create one at the [Hugging Face Settings page](https://huggingface.co/settings/tokens).",
    )
    if hf_env_token:
        st.caption("🔑 *Server HF token configured; leave blank to use it*")

    st.divider()
    st.subheader("🧪 Data cleaning")
    use_semantic_chunking: bool = st.checkbox(
        "Semantic chunking",
        value=False,
        help="Split documents by paragraph + sentence boundaries instead of fixed character windows.",
    )
    enable_dedup: bool = st.checkbox(
        "Deduplicate dataset",
        value=True,
        help="Drop exact and near-duplicate question/answer pairs (MinHash).",
    )
    sanitize_dataset: bool = st.checkbox(
        "Clean & sanitize dataset",
        value=False,
        help=(
            "Remove PII (emails, phone numbers, URLs, IPs, card numbers), strip HTML "
            "artifacts, and drop low-quality pairs before export."
        ),
    )

    with st.expander("🔧 Generation settings"):
        temperature: float = st.slider(
            "Temperature", 0.0, 2.0, 0.7, 0.1,
            help="Higher values give more varied questions and answers.",
        )
        max_new_tokens: int = st.number_input(
            "Max answer length (tokens)", min_value=128, max_value=32768, value=2048, step=128,
        )
        concurrency: int = st.number_input(
            "Parallel requests", min_value=1, max_value=64, value=8,
            help="Requests in flight at once. Lower it for a slow local server.",
        )
        request_timeout: int = st.number_input(
            "Request timeout (seconds)", min_value=10, max_value=1800, value=120, step=10,
        )
        judge_model: str = st.text_input(
            "Judge model", value="",
            help="Grades every pair in Balanced and Research mode. Blank: the (first) teacher model.",
        )
        judge_threshold: int = st.select_slider(
            "Minimum judge score", options=[1, 2, 3, 4, 5], value=4,
            help="A pair is kept only if faithfulness, helpfulness and correctness all reach this score.",
        )

# ── Main panel ───────────────────────────────────────────────────────────────

teacher_model: str = st.text_input(
    "Teacher Model(s)",
    value=DEFAULT_MODEL,
    help="The model name as your endpoint knows it. Comma-separate several for an ensemble.",
)
st.caption(
    "💡 **Examples:** `gpt-4o-mini` · `gpt-4.1` (OpenAI) · `Qwen/Qwen2.5-72B-Instruct` "
    "(the name `vllm serve` was started with) · `llama3.1:8b` (Ollama)"
)

quality_label: str = st.selectbox(
    "Quality Mode",
    options=list(QUALITY_MODE_LABELS.values()),
    index=1,
)
quality_mode = next(k for k, v in QUALITY_MODE_LABELS.items() if v == quality_label)

format_label: str = st.selectbox(
    "Output Format",
    options=list(OUTPUT_FORMAT_LABELS.values()),
    index=0,
    help="Choose the dataset format your training framework expects.",
)
output_format = next(k for k, v in OUTPUT_FORMAT_LABELS.items() if v == format_label)

dataset_size: int = st.slider(
    "Target Dataset Size", 10, 5000, 200,
    help="Generation stops when this many pairs pass every check, or when the documents run out of new questions.",
)

train_model: bool = st.checkbox("Auto-train LoRA adapter", value=False)
base_model: str = DEFAULT_BASE_MODEL
lora_rank: int = 16
if train_model:
    st.caption("Needs the training extra (`uv sync --extra train`) and an NVIDIA GPU for real models.")
    col_model, col_rank = st.columns([3, 1])
    base_model = col_model.text_input(
        "Base model to fine-tune",
        value=DEFAULT_BASE_MODEL,
        help="A Hugging Face model id. Instruct models with a chat template work best.",
    )
    lora_rank = col_rank.select_slider("LoRA rank", options=[4, 8, 16, 32, 64, 128], value=16)

publish: bool = st.checkbox("Publish to Hugging Face", value=False)
hf_repo_name: str | None = None
if publish:
    default_repo: str = f"{os.getenv('HF_USERNAME', 'yourusername')}/brainbrew-dataset"
    hf_repo_name = st.text_input(
        "Hugging Face Repo",
        value=default_repo,
        help="Format: username/repo-slug. Created as private if it does not exist.",
    )

uploaded_files = st.file_uploader(
    "Upload documents (PDF/TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    total_bytes: int = sum(getattr(f, "size", 0) or 0 for f in uploaded_files)
    if total_bytes > MAX_WARN_BYTES:
        st.warning(
            f"⚠️ Total upload is **{total_bytes / 1e6:.1f} MB**. "
            "Large documents produce more chunks and will take longer to process. "
            "Consider splitting into smaller files for faster iteration."
        )


@st.cache_data(show_spinner="Reading documents…", max_entries=8)
def _read_uploads(files: tuple[tuple[str, bytes], ...]) -> tuple[str, list[str]]:
    """Extracted text plus per-file read errors (cached per upload content)."""
    parts, errors = [], []
    for name, data in files:
        try:
            parts.append(read_document(name, data))
        except Exception as e:
            errors.append(f"Could not parse '{name}': {e} — skipping.")
    return "\n\n".join(parts), errors


source_text: str = ""
read_errors: list[str] = []
if uploaded_files:
    source_text, read_errors = _read_uploads(tuple((f.name, f.getvalue()) for f in uploaded_files))
chunk_count = len(source_chunks(source_text, use_semantic_chunking)) if source_text.strip() else 0


# ── Cost / time / yield estimate ─────────────────────────────────────────────

# Pricing (USD per 1M tokens, blended input+output estimate).
# Source: https://openai.com/api/pricing/
_MODEL_PRICING: dict[str, float] = {
    "gpt-4o-mini":    0.50,    # $0.15 input + $0.60 output per 1M
    "gpt-4o":         8.00,    # $2.50 input + $10 output per 1M
    "gpt-4.1-mini":   0.35,    # $0.10 input + $0.40 output per 1M
    "gpt-4.1":        6.50,    # $2.00 input + $8.00 output per 1M
    "gpt-3.5-turbo":  1.00,    # legacy pricing estimate
}
_DEFAULT_COST_PER_M: float = 8.00  # conservative default for unknown hosted models
# Tokens per *accepted* pair, including over-generation, question writing,
# answering, and (Balanced/Research) judging and evolving.
_TOKENS_PER_PAIR: dict[str, int] = {"fast": 1500, "balanced": 2600, "research": 3800}


def _estimate(model: str, size: int, mode: str, local: bool) -> tuple[str, str]:
    """Return (cost_str, time_str) estimates for the UI info bar."""
    total_tokens = size * _TOKENS_PER_PAIR.get(mode, 2600)
    minutes = max(1, round(total_tokens / 60_000))  # ~1k tokens/s across parallel requests
    if local:
        return "Free (your server)", f"~{minutes} min"
    first_model = model.split(",")[0].strip().lower()
    cost_per_m = next((price for key, price in _MODEL_PRICING.items() if key in first_model), _DEFAULT_COST_PER_M)
    return f"~${total_tokens * cost_per_m / 1_000_000:.2f}", f"~{minutes} min"


is_local = base_url is not None and urlsplit(base_url).hostname in {"localhost", "127.0.0.1", "::1"}
est_cost, est_time = _estimate(teacher_model, dataset_size, quality_mode.value, is_local)
yield_note = ""
if chunk_count:
    expected = chunk_count * PAIRS_PER_CHUNK_ESTIMATE
    yield_note = f"  ·  📄 {chunk_count} chunks (≈{expected} pairs possible)"
st.info(
    f"💰 Estimated cost: **{est_cost}**  ·  ⏱️ Estimated time: **{est_time}**  "
    f"·  🎯 Target **{dataset_size}** pairs{yield_note}  ·  Mode: **{quality_label}**"
)
if chunk_count and dataset_size > chunk_count * PAIRS_PER_CHUNK_ESTIMATE:
    st.warning(
        f"These documents will likely support about {chunk_count * PAIRS_PER_CHUNK_ESTIMATE} good pairs, "
        f"fewer than the target of {dataset_size}. The run stops when the chunks stop producing new questions."
    )

# ── Validation: DistillationConfig is the single source of truth ─────────────

_FIELD_LABELS: dict[str, str] = {
    "teacher_model": "Teacher model",
    "judge_model": "Judge model",
    "base_model": "Base model",
    "base_url": "Endpoint URL",
    "hf_repo": "Hugging Face repo",
    "api_key": "API key",
    "hf_token": "Hugging Face token",
    "dataset_size": "Dataset size",
    "temperature": "Temperature",
    "max_new_tokens": "Max answer length",
    "concurrency": "Parallel requests",
    "request_timeout": "Request timeout",
    "lora_rank": "LoRA rank",
}


def _friendly_errors(exc: ValidationError) -> list[str]:
    """Turn pydantic errors into one readable line each."""
    messages = []
    for err in exc.errors():
        field = str(err["loc"][0]) if err["loc"] else ""
        label = _FIELD_LABELS.get(field)
        for msg in str(err["msg"]).removeprefix("Value error, ").splitlines():
            messages.append(f"{label}: {msg}" if label else msg)
    return messages


validation_errors: list[str] = []
if not uploaded_files:
    validation_errors.append("Upload at least one document (PDF/TXT) to begin.")
else:
    for uploaded in uploaded_files:
        if (getattr(uploaded, "size", 0) or 0) > MAX_HARD_BYTES:
            validation_errors.append(
                f"File '{uploaded.name}' exceeds the 50 MB hard size limit "
                f"({uploaded.size / 1e6:.1f} MB)."
            )
    if not read_errors and not source_text.strip():
        validation_errors.append("No text could be extracted from the uploaded documents.")
if ENDPOINTS[endpoint_label] == _CUSTOM and not base_url:
    validation_errors.append("Enter the endpoint URL.")

cfg: DistillationConfig | None = None
try:
    cfg = DistillationConfig(
        teacher_model=teacher_model,
        judge_model=judge_model or None,
        judge_threshold=judge_threshold,
        base_url=base_url,
        quality_mode=quality_mode,
        output_format=output_format,
        dataset_size=dataset_size,
        train_model=train_model,
        base_model=base_model,
        lora_rank=lora_rank,
        publish_dataset=publish,
        hf_repo=hf_repo_name if publish else None,
        api_key=openai_key or (openai_env_key if use_server_key else None),
        hf_token=hf_token or os.getenv("HF_TOKEN"),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        concurrency=concurrency,
        request_timeout=request_timeout,
        use_semantic_chunking=use_semantic_chunking,
        enable_dedup=enable_dedup,
        sanitize_dataset=sanitize_dataset,
    )
except ValidationError as exc:
    validation_errors.extend(_friendly_errors(exc))

if validation_errors:
    st.error(
        "⚠️ **Please resolve the following issues to enable dataset generation:**\n\n"
        + "\n".join(f"- {err}" for err in validation_errors)
    )
    button_help = "Solve the validation errors listed above to enable dataset generation."
else:
    button_help = "Click to start generating the dataset."


# ── Results (rendered on every rerun while this session has a run) ──────────

_GRADE_EMOJI = {"SUPER": "🟢", "GOOD": "🔵", "NORMAL": "🟡", "BAD": "🟠", "DISASTER": "🔴"}


def _preview(run: RunDir, limit: int = 5) -> list[Record]:
    rows: list[Record] = []
    try:
        with open(run.records, encoding="utf-8") as fh:
            for line in fh:
                if len(rows) >= limit:
                    break
                rows.append(Record.model_validate(json.loads(line)))
    except (OSError, ValueError):
        logger.debug("Preview unavailable", run_id=run.run_id, exc_info=True)
    return rows


def _render_results(run_id: str) -> None:
    try:
        run = open_run(run_id)
    except (ValueError, FileNotFoundError):
        st.session_state.pop("run_id", None)
        return
    manifest: dict[str, Any] = run.read_manifest()
    if manifest.get("status") != "succeeded":
        return

    quality = manifest.get("quality", {})
    grade = quality.get("grade", "DISASTER")
    st.markdown(f"### {_GRADE_EMOJI.get(grade, '⚪')} Dataset Quality: **{grade}**")
    st.caption(quality.get("details", ""))
    col1, col2, col3 = st.columns(3)
    col1.metric("Records", quality.get("record_count", 0))
    col2.metric("Avg. Output Length", f"{quality.get('avg_output_length', 0):.0f} chars")
    col3.metric("Uniqueness", f"{quality.get('unique_ratio', 0):.0%}")

    preview = _preview(run)
    if preview:
        with st.expander("👀 Preview first 5 examples", expanded=True):
            for i, rec in enumerate(preview, 1):
                st.markdown(f"**Example {i}**")
                with st.chat_message("user"):
                    st.markdown(rec.prompt)
                with st.chat_message("assistant"):
                    st.markdown(rec.output)
                st.divider()

    dataset = run.root / str(manifest.get("dataset_file", ""))
    if dataset.is_file():
        st.download_button(
            "📥 Download dataset",
            dataset.read_bytes(),
            file_name=f"brainbrew-{run.run_id}-{dataset.name}",
            mime="application/jsonl",
            on_click="ignore",
            help="Download the generated dataset in JSONL format.",
        )
    adapter = run.root / str(manifest.get("adapter_file", ""))
    if manifest.get("adapter_file") and adapter.is_file():
        st.download_button(
            "🎯 Download LoRA adapter",
            adapter.read_bytes(),
            file_name=f"brainbrew-{run.run_id}-adapter.zip",
            mime="application/zip",
            on_click="ignore",
        )
    if repo := manifest.get("published_repo"):
        st.success(f"Published to https://huggingface.co/datasets/{repo}")
    st.caption(f"Run `{run.run_id}` · files saved in `{run.root}`")


# ── Generate button ──────────────────────────────────────────────────────────

_STAGE_LABELS: dict[int, str] = {
    5:   "📄 Reading document…",
    15:  "⚗️  Writing questions, answering, judging… (this is the long part)",
    80:  "🧼 Finishing up…",
    85:  "💾 Exporting dataset…",
    92:  "🎯 Training LoRA adapter…",
    96:  "🚀 Publishing to Hugging Face…",
    100: "✅ Done!",
}

if st.button(
    "🚀 Generate Dataset", type="primary",
    disabled=bool(validation_errors), help=button_help,
) and cfg is not None and uploaded_files:
    for warning in read_errors:
        st.warning(warning)
    run = create_run()
    run.source.write_text(source_text, encoding="utf-8")

    progress_bar = st.progress(0)
    status = st.empty()

    def _on_progress(pct: int) -> None:
        progress_bar.progress(pct)
        label = _STAGE_LABELS.get(pct) or (_STAGE_LABELS[15] if 15 < pct < 80 else None)
        if label:
            status.caption(label)

    try:
        result = run_distillation(cfg, run.source, _on_progress, run=run)
    except Exception as e:
        logger.exception("Generation failed", run_id=run.run_id)
        st.error(f"Generation failed: {e}")
    else:
        st.session_state["run_id"] = result.run.run_id
        status.empty()
        st.success("✅ Dataset generated!")
        st.toast("🎉 Synthetic dataset generated successfully!", icon="🧠")
        if result.published_repo:
            st.balloons()

if "run_id" in st.session_state:
    _render_results(st.session_state["run_id"])
