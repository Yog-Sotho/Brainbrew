"""
Brainbrew — Streamlit UI for synthetic dataset generation.

This is the main entry point for the application. Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
from typing import Any

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
from pipeline.document_loader import read_document
from pipeline.records import Record
from pipeline.runs import RunDir, create_run, open_run

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

st.title("🧠 Brainbrew v1.3.0")
st.caption("Production-grade synthetic dataset generator — GPU edition")

# ── Sidebar: settings ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Advanced Settings")
    use_vllm: bool = st.checkbox("Use vLLM (GPU required)", value=True)

    # Server-side secrets are never used as widget values: Streamlit sends widget
    # state to the browser, so a pre-filled password field discloses the key to
    # every visitor. The env value is applied server-side as a fallback instead.
    openai_env_key = os.getenv("OPENAI_API_KEY", "")
    openai_key: str = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Using server key" if openai_env_key else "sk-...",
        help="Enter your OpenAI API key. Get one at the [OpenAI API Keys page](https://platform.openai.com/api-keys).",
    )
    if openai_env_key:
        st.caption("🔑 *Server API key configured; leave blank to use it*")
    elif not use_vllm:
        st.caption("⚠️ *API Key required to run OpenAI models*")

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
    st.subheader("🧪 Experimental")
    use_semantic_chunking: bool = st.checkbox(
        "Semantic chunking",
        value=False,
        help="Split documents by paragraph + sentence boundaries instead of fixed character windows.",
    )
    enable_dedup: bool = st.checkbox(
        "Deduplicate dataset",
        value=True,
        help="Remove exact and near-duplicate instruction/output pairs.",
    )
    sanitize_dataset: bool = st.checkbox(
        "Clean & sanitize dataset",
        value=False,
        help=(
            "Remove PII (emails, phone numbers, URLs, IPs, card numbers), strip HTML "
            "artifacts, deduplicate, and drop low-quality pairs before export."
        ),
    )

    with st.expander("🔧 Generation settings"):
        temperature: float = st.slider(
            "Temperature", 0.0, 2.0, 0.7, 0.1,
            help="Higher values give more varied answers; lower values more predictable ones.",
        )
        max_new_tokens: int = st.number_input(
            "Max answer length (tokens)", min_value=128, max_value=32768, value=2048, step=128,
        )
        batch_size: int = st.number_input(
            "Batch size", min_value=1, max_value=1024, value=64,
            help="Prompts sent to the model per batch.",
        )

# ── Main panel ───────────────────────────────────────────────────────────────

teacher_model: str = st.text_input(
    "Teacher Model(s)",
    value="gpt-4o" if not use_vllm else "meta-llama/Meta-Llama-3.1-8B-Instruct",
    help="Comma-separated list for multi-model ensemble (e.g. gpt-4o,gpt-4.1).",
)
st.caption(
    "💡 **Popular Presets:** `gpt-4o` (OpenAI default) · `gpt-4o-mini` (Fast & Cheap) · "
    "`meta-llama/Meta-Llama-3.1-8B-Instruct` (vLLM local) · `Qwen/Qwen2.5-72B-Instruct` (vLLM large)"
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

dataset_size: int = st.slider("Target Dataset Size", 500, 20000, 2000)

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


# ── FIX M-06: Cost / time estimator with current pricing ────────────────────

# Pricing as of March 2026 (USD per 1M tokens, blended input+output estimate)
# Source: https://openai.com/api/pricing/
_MODEL_PRICING: dict[str, float] = {
    "gpt-4o":         8.00,    # $2.50 input + $10 output per 1M
    "gpt-4o-mini":    0.50,    # $0.15 input + $0.60 output per 1M
    "gpt-4.1":        6.50,    # $2.00 input + $8.00 output per 1M
    "gpt-4.1-mini":   0.35,    # $0.10 input + $0.40 output per 1M
    "gpt-3.5-turbo":  1.00,    # legacy pricing estimate
}
_DEFAULT_COST_PER_M: float = 8.00  # conservative default for unknown models


def _estimate(
    model: str,
    size: int,
    mode: str,
    vllm: bool,
) -> tuple[str, str]:
    """Return (cost_str, time_str) estimates for the UI info bar."""
    evolutions: int = {"fast": 1, "balanced": 2, "research": 3}.get(mode, 2)

    if vllm:
        minutes = max(1, int(size * evolutions * 0.3 / 60))
        return "Free (local GPU)", f"~{minutes} min"

    # Estimate tokens: ~800 tokens per pair × evolutions
    total_tokens: int = size * 800 * evolutions
    first_model = model.split(",")[0].strip()

    # Look up pricing — try exact match, then partial match
    cost_per_m = _DEFAULT_COST_PER_M
    for key, price in _MODEL_PRICING.items():
        if key in first_model.lower():
            cost_per_m = price
            break

    cost: float = total_tokens * (cost_per_m / 1_000_000)
    minutes = max(1, int(size * evolutions * 0.5 / 60))
    return f"~${cost:.2f}", f"~{minutes} min"


est_cost, est_time = _estimate(teacher_model, dataset_size, quality_mode.value, use_vllm)
st.info(
    f"💰 Estimated cost: **{est_cost}**  ·  ⏱️ Estimated time: **{est_time}**  "
    f"·  📦 Up to **{dataset_size}** pairs  ·  Mode: **{quality_label}**  "
    f"·  Format: **{output_format.value}**"
)

# ── Validation: DistillationConfig is the single source of truth ─────────────

_FIELD_LABELS: dict[str, str] = {
    "teacher_model": "Teacher model",
    "base_model": "Base model",
    "hf_repo": "Hugging Face repo",
    "api_key": "API key",
    "hf_token": "Hugging Face token",
    "dataset_size": "Dataset size",
    "temperature": "Temperature",
    "max_new_tokens": "Max answer length",
    "batch_size": "Batch size",
    "lora_rank": "LoRA rank",
}


def _friendly_errors(exc: ValidationError) -> list[str]:
    """Turn pydantic errors into one readable line each."""
    messages = []
    for err in exc.errors():
        msg = str(err["msg"]).removeprefix("Value error, ")
        field = str(err["loc"][0]) if err["loc"] else ""
        label = _FIELD_LABELS.get(field)
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

cfg: DistillationConfig | None = None
try:
    cfg = DistillationConfig(
        teacher_model=teacher_model,
        quality_mode=quality_mode,
        output_format=output_format,
        dataset_size=dataset_size,
        use_vllm=use_vllm,
        train_model=train_model,
        base_model=base_model,
        lora_rank=lora_rank,
        publish_dataset=publish,
        hf_repo=hf_repo_name if publish else None,
        api_key=openai_key or os.getenv("OPENAI_API_KEY"),
        hf_token=hf_token or os.getenv("HF_TOKEN"),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
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
    button_help = "Click to start the synthetic dataset distillation pipeline."


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
    15:  "✂️  Chunking text…",
    20:  "🤖 Initialising model…",
    70:  "⚗️  Running pipeline… (this is the long part)",
    80:  "🧹 Deduplicating…",
    85:  "🧼 Sanitizing dataset…",
    92:  "🎯 Training LoRA adapter…",
    96:  "🚀 Publishing to Hugging Face…",
    100: "✅ Done!",
}

if st.button(
    "🚀 Generate Dataset", type="primary",
    disabled=bool(validation_errors), help=button_help,
) and cfg is not None and uploaded_files:
    run = create_run()
    with open(run.source, "w", encoding="utf-8") as f:
        for uploaded in uploaded_files:
            try:
                f.write(read_document(uploaded.name, uploaded.getvalue()) + "\n\n")
            except Exception as e:
                st.warning(f"Could not parse '{uploaded.name}': {e} — skipping.")

    if not run.source.read_text(encoding="utf-8").strip():
        st.error("No text could be extracted from the uploaded documents.")
        st.stop()

    progress_bar = st.progress(0)
    status = st.empty()

    def _on_progress(pct: int) -> None:
        progress_bar.progress(pct)
        if label := _STAGE_LABELS.get(pct):
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
