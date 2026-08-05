"""
Brainbrew — Streamlit UI for synthetic dataset generation.

This is the main entry point for the application. Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
import structlog
from dotenv import load_dotenv

from config import (
    OUTPUT_FORMAT_LABELS,
    QUALITY_MODE_LABELS,
    DistillationConfig,
    OutputFormat,
    QualityMode,
)
from orchestrator import run_distillation, score_dataset

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger("INFO"))
logger = structlog.get_logger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Brainbrew", page_icon="🧠", layout="wide")
st.title("🧠 Brainbrew v1.2.0")
st.caption("Production-grade synthetic dataset generator — GPU edition")

# ── Sidebar: advanced settings ───────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Advanced Settings")
    use_vllm: bool = st.checkbox("Use vLLM (GPU required)", value=True)

    openai_env_key = os.getenv("OPENAI_API_KEY", "")
    openai_key: str = st.text_input(
        "OpenAI API Key",
        value=openai_env_key,
        type="password",
        help="Enter your OpenAI API key. Get one at the [OpenAI API Keys page](https://platform.openai.com/api-keys).",
    )
    if openai_env_key:
        st.caption("🔑 *API Key loaded automatically from environment*")
    elif not use_vllm:
        st.caption("⚠️ *API Key required to run OpenAI models*")

    hf_env_token = os.getenv("HF_TOKEN", "")
    hf_token: str = st.text_input(
        "Hugging Face Token",
        value=hf_env_token,
        type="password",
        help="Enter your Hugging Face write token. Create one at the [Hugging Face Settings page](https://huggingface.co/settings/tokens).",
    )
    if hf_env_token:
        st.caption("🔑 *HF Token loaded automatically from environment*")

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
            "Run the Dataset Sanitizer after generation to remove PII, "
            "deduplicate, strip HTML artifacts, and enforce quality gates."
        ),
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

# Quality mode selector
quality_label: str = st.selectbox(
    "Quality Mode",
    options=list(QUALITY_MODE_LABELS.values()),
    index=1,
)
quality_mode: str = next(
    k.value for k, v in QUALITY_MODE_LABELS.items() if v == quality_label
)

# Enhancement 6: output format selector
format_label: str = st.selectbox(
    "Output Format",
    options=list(OUTPUT_FORMAT_LABELS.values()),
    index=0,
    help="Choose the dataset format your training framework expects.",
)
output_format: str = next(
    k.value for k, v in OUTPUT_FORMAT_LABELS.items() if v == format_label
)

dataset_size: int = st.slider("Target Dataset Size", 500, 20000, 2000)
train_model: bool = st.checkbox("Auto-train LoRA adapter", value=False)
publish: bool = st.checkbox("Publish to Hugging Face", value=False)

# Editable HF repo name
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
    help="Upload PDF or TXT documents. These documents will be split, chunked, and processed as source knowledge to generate synthetic dataset pairs.",
)

# ── File safety ──────────────────────────────────────────────────────────────

_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")
MAX_WARN_BYTES: int = 10 * 1024 * 1024   # warn at 10 MB
MAX_HARD_BYTES: int = 50 * 1024 * 1024   # hard limit at 50 MB per file

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


est_cost, est_time = _estimate(teacher_model, dataset_size, quality_mode, use_vllm)
st.info(
    f"💰 Estimated cost: **{est_cost}**  ·  ⏱️ Estimated time: **{est_time}**  "
    f"·  📦 Up to **{dataset_size}** pairs  ·  Mode: **{quality_label}**  "
    f"·  Format: **{output_format}**"
)

# ── Proactive Input Validation ───────────────────────────────────────────────
validation_errors = []

if not uploaded_files:
    validation_errors.append("Upload at least one document (PDF/TXT) to begin.")
else:
    for uploaded in uploaded_files:
        # Reset stream pointer as a best practice
        uploaded.seek(0)

        # Check filename safety
        if not _SAFE_FILENAME_RE.match(uploaded.name):
            validation_errors.append(
                f"File '{uploaded.name}' has an unsafe filename. "
                "Only alphanumeric characters, dashes, underscores, spaces, and periods are allowed."
            )
        # Check file size limit
        if getattr(uploaded, "size", 0) > MAX_HARD_BYTES:
            validation_errors.append(
                f"File '{uploaded.name}' exceeds the 50 MB hard size limit "
                f"({uploaded.size / 1e6:.1f} MB)."
            )

if not use_vllm and not openai_key and not os.getenv("OPENAI_API_KEY"):
    validation_errors.append("OpenAI API Key is required when not using vLLM.")

if publish:
    if not hf_token and not os.getenv("HF_TOKEN"):
        validation_errors.append("Hugging Face Token is required when publishing.")

    if not hf_repo_name or not hf_repo_name.strip():
        validation_errors.append("Hugging Face repository name is required when publishing.")
    else:
        _REPO_NAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
        if not _REPO_NAME_RE.match(hf_repo_name.strip()):
            validation_errors.append("Hugging Face repository format is invalid (must be 'username/repo-slug').")

if validation_errors:
    st.error(
        "⚠️ **Please resolve the following issues to enable dataset generation:**\n\n"
        + "\n".join(f"- {err}" for err in validation_errors)
    )
    button_disabled = True
    button_help = "Solve the validation errors listed above to enable dataset generation."
else:
    button_disabled = False
    button_help = "Click to start the synthetic dataset distillation pipeline."

# ── Generate button ──────────────────────────────────────────────────────────

if st.button("🚀 Generate Dataset", type="primary", disabled=button_disabled, help=button_help):
    if not uploaded_files:
        st.error("Upload at least one document")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source_path: Path = tmp_path / "source.txt"

        with open(source_path, "w", encoding="utf-8") as f:
            for uploaded in uploaded_files:
                # Reset stream pointer
                uploaded.seek(0)
                # Validate filename
                if not _SAFE_FILENAME_RE.match(uploaded.name):
                    st.warning(f"Skipped '{uploaded.name}' — unsafe filename.")
                    continue
                # Hard file-size limit per file
                if getattr(uploaded, "size", 0) > MAX_HARD_BYTES:
                    st.warning(f"Skipped '{uploaded.name}' — exceeds 50 MB limit.")
                    continue
                # Parse with error handling
                try:
                    if uploaded.type == "application/pdf":
                        from pdfminer.high_level import extract_text
                        content: str = extract_text(uploaded)
                    else:
                        content = uploaded.read().decode("utf-8")
                    f.write(content + "\n\n")
                except Exception as e:
                    st.warning(f"Could not parse '{uploaded.name}': {e} — skipping.")
                    continue

        cfg = DistillationConfig(
            teacher_model=teacher_model,
            quality_mode=QualityMode(quality_mode),
            output_format=OutputFormat(output_format),
            dataset_size=dataset_size,
            use_vllm=use_vllm,
            train_model=train_model,
            publish_dataset=publish,
            hf_repo=hf_repo_name if publish else None,
            api_key=openai_key or os.getenv("OPENAI_API_KEY"),
            hf_token=hf_token or os.getenv("HF_TOKEN"),
            use_semantic_chunking=use_semantic_chunking,
            enable_dedup=enable_dedup,
            sanitize_dataset=sanitize_dataset,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        # Honest progress bar with stage labels
        progress_bar = st.progress(0)
        status = st.empty()

        _STAGE_LABELS: dict[int, str] = {
            5:   "📄 Reading document…",
            15:  "✂️  Chunking text…",
            20:  "🤖 Initialising model…",
            70:  "⚗️  Running pipeline… (this is the long part)",
            80:  "💾 Exporting dataset…",
            85:  "🧹 Sanitizing dataset…",
            92:  "🎯 Training LoRA adapter…",
            96:  "🚀 Publishing to Hugging Face…",
            100: "✅ Done!",
        }

        def _on_progress(pct: int) -> None:
            progress_bar.progress(pct)
            label = _STAGE_LABELS.get(pct, "")
            if label:
                status.caption(label)

        try:
            final_path: Path = run_distillation(
                cfg, source_path, _on_progress, output_dir=tmp_path
            )
            st.success("✅ Dataset generated!")
            st.toast("🎉 Synthetic dataset generated successfully!", icon="🧠")
            status.empty()

            # ── Enhancement 10: Quality scoring dashboard ────────────────
            quality_report = score_dataset(final_path)
            grade = quality_report["grade"]

            grade_colors = {
                "SUPER": "🟢", "GOOD": "🔵", "NORMAL": "🟡",
                "BAD": "🟠", "DISASTER": "🔴",
            }
            grade_emoji = grade_colors.get(grade, "⚪")

            st.markdown(f"### {grade_emoji} Dataset Quality: **{grade}**")
            st.caption(quality_report["details"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Records", quality_report["record_count"])
            col2.metric("Avg. Output Length", f"{quality_report['avg_output_length']:.0f} chars")
            col3.metric("Uniqueness", f"{quality_report['unique_ratio']:.0%}")

            # ── Preview first 5 examples ─────────────────────────────────
            try:
                with open(final_path, encoding="utf-8") as fh:
                    preview_rows = [json.loads(line) for line in fh if line.strip()][:5]
                if preview_rows:
                    with st.expander("👀 Preview first 5 examples", expanded=True):
                        for i, row in enumerate(preview_rows, 1):
                            st.markdown(f"**Example {i}**")
                            # Handle different output formats
                            if "instruction" in row:
                                inst_text = row.get("instruction", "")
                                with st.chat_message("user"):
                                    st.markdown(inst_text[:1000] + ("…" if len(inst_text) > 1000 else ""))
                                output_text = row.get("output", "")
                                with st.chat_message("assistant"):
                                    st.markdown(output_text[:1000] + ("…" if len(output_text) > 1000 else ""))
                            elif "messages" in row:
                                for msg in row["messages"]:
                                    role = msg.get("role", "unknown")
                                    content = msg.get("content", "")
                                    mapped_role = "user" if role.lower() == "user" else ("assistant" if role.lower() == "assistant" else "user")
                                    with st.chat_message(mapped_role):
                                        st.markdown(f"**{role.capitalize()}**")
                                        st.markdown(content[:1000] + ("…" if len(content) > 1000 else ""))
                            elif "conversations" in row:
                                for turn in row["conversations"]:
                                    who = turn.get("from", "unknown")
                                    val = turn.get("value", "")
                                    mapped_role = "user" if who.lower() in ("human", "user") else ("assistant" if who.lower() in ("gpt", "assistant") else "user")
                                    with st.chat_message(mapped_role):
                                        st.markdown(f"**{who.capitalize()}**")
                                        st.markdown(val[:1000] + ("…" if len(val) > 1000 else ""))
                            st.divider()
            except Exception:
                logger.debug("Preview rendering failed", exc_info=True)

            # FIX H-03: use context manager for file handle
            with open(final_path, encoding="utf-8") as f:
                st.download_button(
                    "📥 Download dataset",
                    f.read(),
                    file_name=final_path.name,
                    help="Click to save the generated synthetic dataset to your local machine.",
                    use_container_width=True,
                )

            if cfg.publish_dataset:
                st.balloons()

        except Exception as e:
            logger.exception("Failed")
            st.error(f"Generation failed: {e}")
