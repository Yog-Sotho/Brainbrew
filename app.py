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
from pipeline.decontam import EVAL_SETS
from pipeline.document_loader import source_chunks
from pipeline.jobs import get_runner
from pipeline.pii import presidio_available
from pipeline.pricing import estimate_cost, is_local
from pipeline.service import new_run, read_documents
from pipeline.synth import PAIRS_PER_CHUNK_ESTIMATE
from pipeline.version import __version__
from publish.dataset_card import LICENSES
from ui.common import current_owner, setup_page, visible_run
from ui.results import render_run

setup_page("Generate")
logger = structlog.get_logger(__name__)

MAX_WARN_BYTES: int = 10 * 1024 * 1024   # warn at 10 MB
MAX_HARD_BYTES: int = 50 * 1024 * 1024   # hard limit at 50 MB per file

st.title(f"🧠 Brainbrew v{__version__}")
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
_URL_POLICY_LABELS = {
    "domain": "Keep the site, drop the path",
    "redact": "Remove links",
    "keep": "Keep links",
}
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
            "Remove PII (emails, phone numbers, IPs, card and bank numbers, links), strip HTML "
            "artifacts, and drop low-quality pairs before export."
        ),
    )
    pii_url_policy = "domain"
    pii_presidio = False
    if sanitize_dataset:
        pii_url_policy = st.selectbox(
            "Links in the data",
            options=list(_URL_POLICY_LABELS),
            format_func=_URL_POLICY_LABELS.__getitem__,
            help="Login details and secret-looking query values are removed from links in every mode.",
        )
        if presidio_available():
            pii_presidio = st.checkbox(
                "Also detect names (Presidio)",
                value=False,
                help="NER-based detection of person names, passport and licence numbers. Slower.",
            )
    decontaminate: list[str] = st.multiselect(
        "Remove benchmark overlap",
        options=list(EVAL_SETS),
        format_func=lambda key: EVAL_SETS[key].label,
        help=(
            "Drop pairs that share a 13-word passage with these public test sets, so models "
            "trained on the dataset are not evaluated on text they have seen. Downloads the "
            "benchmarks from Hugging Face on first use."
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
        embedding_model: str = st.text_input(
            "Embedding model (semantic dedup)", value="",
            help=(
                "Also drop paraphrased duplicates using embeddings from the same endpoint, e.g. "
                "`text-embedding-3-small` (OpenAI) or an embedding model your server hosts. Blank: off."
            ),
        )
        semantic_dedup_threshold: float = st.slider(
            "Paraphrase similarity cut-off", 0.80, 0.99, 0.92, 0.01,
            help="Pairs at least this similar (cosine) to an accepted pair are dropped.",
            disabled=not embedding_model.strip(),
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
chunk_count = len(source_chunks(source_text, use_semantic_chunking)) if source_text.strip() else 0


# ── Cost / time / yield estimate ─────────────────────────────────────────────

# Tokens per *accepted* pair, including over-generation, question writing,
# answering, and (Balanced/Research) judging and evolving.
_TOKENS_PER_PAIR: dict[str, int] = {"fast": 1500, "balanced": 2600, "research": 3800}


def _estimate(model: str, size: int, mode: str, local: bool) -> tuple[str, str]:
    """Return (cost_str, time_str) estimates for the UI info bar."""
    total_tokens = size * _TOKENS_PER_PAIR.get(mode, 2600)
    minutes = max(1, round(total_tokens / 60_000))  # ~1k tokens/s across parallel requests
    if local:
        return "Free (your server)", f"~{minutes} min"
    cost = estimate_cost(model.split(",")[0].strip(), total_tokens, local) or 0.0
    return f"~${cost:.2f}", f"~{minutes} min"


est_cost, est_time = _estimate(teacher_model, dataset_size, quality_mode.value, is_local(base_url))
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
    "embedding_model": "Embedding model",
    "base_model": "Base model",
    "base_url": "Endpoint URL",
    "hf_repo": "Hugging Face repo",
    "hf_model_repo": "Adapter repo",
    "api_key": "API key",
    "hf_token": "Hugging Face token",
    "dataset_size": "Dataset size",
    "temperature": "Temperature",
    "max_new_tokens": "Max answer length",
    "concurrency": "Parallel requests",
    "request_timeout": "Request timeout",
    "lora_rank": "LoRA rank",
    "decontaminate": "Benchmarks",
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
        embedding_model=embedding_model or None,
        semantic_dedup_threshold=semantic_dedup_threshold,
        base_url=base_url,
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
        api_key=openai_key or (openai_env_key if use_server_key else None),
        hf_token=hf_token or os.getenv("HF_TOKEN"),
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        concurrency=concurrency,
        request_timeout=request_timeout,
        use_semantic_chunking=use_semantic_chunking,
        enable_dedup=enable_dedup,
        sanitize_dataset=sanitize_dataset,
        pii_url_policy=pii_url_policy,
        pii_presidio=pii_presidio,
        decontaminate=decontaminate,
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
