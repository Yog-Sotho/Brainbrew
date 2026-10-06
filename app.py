"""
Brainbrew — Streamlit UI for synthetic dataset generation.

This is the main entry point for the application. Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st
import structlog
from pydantic import ValidationError

from config import (
    DEFAULT_BASE_MODEL,
    OUTPUT_FORMAT_LABELS,
    QUALITY_MODE_LABELS,
    DistillationConfig,
)
from pipeline.document_loader import source_chunks
from pipeline.jobs import get_runner
from pipeline.pricing import is_local
from pipeline.service import new_run, read_documents
from pipeline.synth import PAIRS_PER_CHUNK_ESTIMATE
from pipeline.version import __version__
from publish.dataset_card import LICENSES
from ui.common import (
    current_owner,
    custom_endpoints_allowed,
    login_required,
    server_hf_token_allowed,
    setup_page,
    visible_run,
)
from ui.generate import (
    MAX_WARN_BYTES,
    endpoint_options,
    estimate,
    friendly_errors,
    server_endpoint_label,
    upload_errors,
)
from ui.results import render_run
from ui.sidebar import render_sidebar

setup_page("Generate")
logger = structlog.get_logger(__name__)

st.title(f"🧠 Brainbrew v{__version__}")
st.caption("Grounded synthetic dataset generator for any OpenAI-compatible model")

SERVER_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL", "").strip() or None
DEFAULT_MODEL = os.getenv("BRAINBREW_DEFAULT_MODEL", "").strip() or "gpt-4o-mini"
ENDPOINTS = endpoint_options(SERVER_BASE_URL, custom_endpoints_allowed())
side = render_sidebar(ENDPOINTS, server_endpoint_label(SERVER_BASE_URL))

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
hf_public = False
dataset_license = "other"
publish_adapter = False
if publish:
    default_repo: str = f"{os.getenv('HF_USERNAME', 'yourusername')}/brainbrew-dataset"
    hf_repo_name = st.text_input(
        "Hugging Face Repo",
        value=default_repo,
        help="Format: username/repo-slug. A dataset card describing how the data was made is uploaded too.",
    )
    col_lic, col_pub = st.columns([2, 1])
    dataset_license = col_lic.selectbox(
        "License", options=list(LICENSES),
        help="Shown on the dataset card. Data derived from your documents may be bound by their terms.",
    )
    hf_public = col_pub.checkbox("Make it public", value=False,
                                 help="New repos are private unless you tick this.")
    if train_model:
        publish_adapter = st.checkbox(
            "Also publish the LoRA adapter", value=False,
            help="Uploaded as a model repo named <dataset repo>-lora, with a model card.",
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
    return read_documents(files)


source_text: str = ""
read_errors: list[str] = []
if uploaded_files:
    source_text, read_errors = _read_uploads(tuple((f.name, f.getvalue()) for f in uploaded_files))
chunk_count = len(source_chunks(source_text, side.use_semantic_chunking)) if source_text.strip() else 0


# ── Cost / time / yield estimate ─────────────────────────────────────────────

est_cost, est_time = estimate(teacher_model, dataset_size, quality_mode.value, is_local(side.base_url))
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

validation_errors = upload_errors(uploaded_files, source_text, read_errors)
if side.custom_url_missing:
    validation_errors.append("Enter the endpoint URL.")

# The server's HF token. With login on, several people share the app, so the
# server token may only publish into the operator's namespace (HF_USERNAME);
# anything else needs the user's own token. Otherwise any user could overwrite
# any repo the server token can write to.
server_hf_token: str | None = None
if publish and not side.hf_token and side.hf_env_token:
    namespace = os.getenv("HF_USERNAME", "").strip()
    if server_hf_token_allowed(hf_repo_name, namespace, login_required()):
        server_hf_token = side.hf_env_token
    else:
        validation_errors.append(
            f"The server's Hugging Face token can only publish to {namespace or 'the operator'}'s repos. "
            "Enter your own token to publish elsewhere."
        )

cfg: DistillationConfig | None = None
try:
    cfg = DistillationConfig(
        teacher_model=teacher_model,
        judge_model=side.judge_model or None,
        judge_threshold=side.judge_threshold,
        embedding_model=side.embedding_model or None,
        semantic_dedup_threshold=side.semantic_dedup_threshold,
        base_url=side.base_url,
        quality_mode=quality_mode,
        output_format=output_format,
        dataset_size=dataset_size,
        train_model=train_model,
        base_model=base_model,
        lora_rank=lora_rank,
        publish_dataset=publish,
        hf_repo=hf_repo_name if publish else None,
        hf_private=not hf_public,
        dataset_license=dataset_license,
        publish_adapter=publish_adapter,
        api_key=side.api_key,
        hf_token=side.hf_token or server_hf_token,
        temperature=side.temperature,
        max_new_tokens=side.max_new_tokens,
        concurrency=side.concurrency,
        request_timeout=side.request_timeout,
        use_semantic_chunking=side.use_semantic_chunking,
        enable_dedup=side.enable_dedup,
        sanitize_dataset=side.sanitize_dataset,
        pii_url_policy=side.pii_url_policy,
        pii_presidio=side.pii_presidio,
        decontaminate=side.decontaminate,
    )
except ValidationError as exc:
    validation_errors.extend(friendly_errors(exc))

if validation_errors:
    st.error(
        "⚠️ **Please resolve the following issues to enable dataset generation:**\n\n"
        + "\n".join(f"- {err}" for err in validation_errors)
    )
    button_help = "Solve the validation errors listed above to enable dataset generation."
else:
    button_help = "Click to start generating the dataset."


# ── Generate button ──────────────────────────────────────────────────────────
# The run is handed to the process-wide job runner, so it keeps going when the
# page is closed or reloaded. Its id goes into the URL (?run=...), which is how
# a reload finds it again.

if st.button(
    "🚀 Generate Dataset", type="primary",
    disabled=bool(validation_errors), help=button_help,
) and cfg is not None and uploaded_files:
    for warning in read_errors:
        st.warning(warning)
    owner = current_owner()
    new = new_run(source_text, owner)
    get_runner().submit(cfg, new, owner)
    st.session_state["run_id"] = new.run_id
    st.query_params["run"] = new.run_id
    logger.info("Run submitted", run_id=new.run_id)

run_id = st.session_state.get("run_id") or st.query_params.get("run")
if run_id:
    shown = visible_run(str(run_id))
    if shown is None:
        st.session_state.pop("run_id", None)
        st.query_params.pop("run", None)
    else:
        st.session_state["run_id"] = shown.run_id
        st.divider()
        render_run(shown)
