import streamlit as st
import tempfile
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import structlog

from config import DistillationConfig, QualityMode, QUALITY_MODE_LABELS
from orchestrator import run_distillation

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger("INFO"))
logger = structlog.get_logger(__name__)

st.set_page_config(page_title="Brainbrew", page_icon="🧠", layout="wide")
st.title("🧠 Brainbrew v1.1.0")
st.caption("Production-grade synthetic dataset generator")

with st.sidebar:
    st.header("⚙️ Advanced Settings")
    use_vllm = st.checkbox("Use vLLM (GPU required)", value=True)
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    hf_token = st.text_input("Hugging Face Token", value=os.getenv("HF_TOKEN", ""), type="password")

teacher_model = st.text_input("Teacher Model(s)", value="gpt-4o" if not use_vllm else "meta-llama/Meta-Llama-3.1-8B-Instruct")

# FIX: friendly quality mode display names instead of raw enum values
quality_label = st.selectbox(
    "Quality Mode",
    options=list(QUALITY_MODE_LABELS.values()),
    index=1,
)
# Map friendly label back to internal QualityMode value
quality_mode = next(k.value for k, v in QUALITY_MODE_LABELS.items() if v == quality_label)

dataset_size = st.slider("Target Dataset Size", 500, 20000, 2000)
train_model = st.checkbox("Auto-train LoRA adapter", value=False)
publish = st.checkbox("Publish to Hugging Face", value=False)

# FIX: editable HF repo name — was hardcoded, user had to edit the source code to change it
hf_repo_name = None
if publish:
    default_repo = f"{os.getenv('HF_USERNAME', 'yourusername')}/brainbrew-dataset"
    hf_repo_name = st.text_input("Hugging Face Repo", value=default_repo,
                                  help="Format: username/repo-slug. Created as private if it does not exist.")

uploaded_files = st.file_uploader("Upload documents (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

# FIX S-02: safe filename pattern to prevent path traversal
_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ]+$")

# FIX: file-size warning — surface large uploads before the user waits 20 minutes
MAX_WARN_BYTES = 10 * 1024 * 1024   # warn at 10 MB
MAX_HARD_BYTES = 50 * 1024 * 1024   # hard limit at 50 MB per file
if uploaded_files:
    total_bytes = sum(getattr(f, "size", 0) or 0 for f in uploaded_files)
    if total_bytes > MAX_WARN_BYTES:
        st.warning(
            f"⚠️ Total upload is **{total_bytes / 1e6:.1f} MB**. "
            "Large documents produce more chunks and will take longer to process. "
            "Consider splitting into smaller files for faster iteration."
        )


# ── Cost / time estimator ────────────────────────────────────────────────────

def _estimate(model: str, size: int, mode: str, vllm: bool) -> tuple[str, str]:
    """Return (cost_str, time_str) estimates shown before the user clicks Go."""
    if vllm:
        minutes = max(1, int(size * {"fast": 1, "balanced": 2, "research": 3}.get(mode, 2) * 0.3 / 60))
        return "Free (local GPU)", f"~{minutes} min"
    evolutions = {"fast": 1, "balanced": 2, "research": 3}.get(mode, 2)
    # rough: ~800 tokens per pair × evolutions for generation + evolution passes
    total_tokens = size * 800 * evolutions
    if "mini" in model or "3.5" in model:
        cost = total_tokens * 0.0000004   # ~$0.40/1M
    else:
        cost = total_tokens * 0.000010    # ~$10/1M (gpt-4o)
    minutes = max(1, int(size * evolutions * 0.5 / 60))
    return f"~${cost:.2f}", f"~{minutes} min"


est_cost, est_time = _estimate(teacher_model, dataset_size, quality_mode, use_vllm)
st.info(
    f"💰 Estimated cost: **{est_cost}**  ·  ⏱️ Estimated time: **{est_time}**  "
    f"·  📦 Up to **{dataset_size}** pairs  ·  Mode: **{quality_label}**"
)

if st.button("🚀 Generate Dataset", type="primary"):
    if not uploaded_files:
        st.error("Upload at least one document")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / "source.txt"
        with open(source_path, "w", encoding="utf-8") as f:
            for uploaded in uploaded_files:
                # FIX S-02: validate filename before processing
                if not _SAFE_FILENAME_RE.match(uploaded.name):
                    st.warning(f"Skipped '{uploaded.name}' — unsafe filename.")
                    continue
                # FIX: hard file-size limit per file
                if getattr(uploaded, "size", 0) > MAX_HARD_BYTES:
                    st.warning(f"Skipped '{uploaded.name}' — exceeds 50 MB limit.")
                    continue
                # FIX: wrap PDF extraction in try/except so one bad file doesn't crash all
                try:
                    if uploaded.type == "application/pdf":
                        from pdfminer.high_level import extract_text
                        content = extract_text(uploaded)
                    else:
                        content = uploaded.read().decode("utf-8")
                    f.write(content + "\n\n")
                except Exception as e:
                    st.warning(f"Could not parse '{uploaded.name}': {e} — skipping.")
                    continue

        cfg = DistillationConfig(
            teacher_model=teacher_model,
            quality_mode=QualityMode(quality_mode),
            dataset_size=dataset_size,
            use_vllm=use_vllm,
            train_model=train_model,
            publish_dataset=publish,
            hf_repo=hf_repo_name if publish else None,
            api_key=openai_key or os.getenv("OPENAI_API_KEY"),
        )

        # FIX: progress bar now reflects honest pipeline stages, not just 0→100 at the end
        progress_bar = st.progress(0)
        status = st.empty()

        _STAGE_LABELS = {
            5:  "📄 Reading document…",
            15: "✂️  Chunking text…",
            20: "🤖 Initialising model…",
            70: "⚗️  Running pipeline… (this is the long part)",
            80: "💾 Exporting dataset…",
            92: "🎯 Training LoRA adapter…",
            96: "🚀 Publishing to Hugging Face…",
            100: "✅ Done!",
        }

        def _on_progress(pct: int) -> None:
            progress_bar.progress(pct)
            label = _STAGE_LABELS.get(pct, "")
            if label:
                status.caption(label)

        try:
            final_path = run_distillation(cfg, source_path, _on_progress)
            st.success("✅ Dataset generated!")
            status.empty()

            # ── FIX: Preview 5 examples before downloading ───────────────────
            try:
                with open(final_path, encoding="utf-8") as fh:
                    preview_rows = [json.loads(line) for line in fh if line.strip()][:5]
                if preview_rows:
                    with st.expander("👀 Preview first 5 examples", expanded=True):
                        for i, row in enumerate(preview_rows, 1):
                            st.markdown(f"**Example {i}**")
                            st.markdown(f"*Instruction:* {row['instruction']}")
                            st.markdown(f"*Output:* {row['output'][:300]}{'…' if len(row['output']) > 300 else ''}")
                            st.divider()
            except Exception:
                pass  # preview is best-effort — never block the download

            st.download_button("📥 Download dataset.jsonl", open(final_path).read(), "dataset.jsonl")
            if cfg.publish_dataset:
                st.balloons()
        except Exception as e:
            logger.exception("Failed")
            st.error(f"Generation failed: {e}")
# FIX: removed the entire broken block that followed the except clause in the original:
#   stray `publish_dataset=publish` token, unmatched `)`, and second run_distillation(cfg)
#   call with wrong signature — all deleted.
