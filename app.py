import streamlit as st
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
import structlog

from config import DistillationConfig, QualityMode
from orchestrator import run_distillation

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger("INFO"))
logger = structlog.get_logger(__name__)

st.set_page_config(page_title="Brainbrew", page_icon="🧠", layout="wide")
st.title("🧠 Brainbrew v1.0")
st.caption("Production-grade synthetic dataset generator")

with st.sidebar:
    st.header("⚙️ Advanced Settings")
    use_vllm = st.checkbox("Use vLLM (GPU required)", value=True)
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    hf_token = st.text_input("Hugging Face Token", value=os.getenv("HF_TOKEN", ""), type="password")

teacher_model = st.text_input("Teacher Model(s)", value="gpt-4o" if not use_vllm else "meta-llama/Meta-Llama-3.1-8B-Instruct")
quality_mode = st.selectbox("Quality Mode", [q.value for q in QualityMode], index=1)
dataset_size = st.slider("Target Dataset Size", 500, 20000, 2000)
train_model = st.checkbox("Auto-train LoRA adapter", value=False)
publish = st.checkbox("Publish to Hugging Face", value=False)

uploaded_files = st.file_uploader("Upload documents (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if st.button("🚀 Generate Dataset", type="primary"):
    if not uploaded_files:
        st.error("Upload at least one document")
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / "source.txt"
        with open(source_path, "w", encoding="utf-8") as f:
            for uploaded in uploaded_files:
                if uploaded.type == "application/pdf":
                    from pdfminer.high_level import extract_text
                    content = extract_text(uploaded)
                else:
                    content = uploaded.read().decode("utf-8")
                f.write(content + "\n\n")

        cfg = DistillationConfig(
            teacher_model=teacher_model,
            quality_mode=QualityMode(quality_mode),
            dataset_size=dataset_size,
            use_vllm=use_vllm,
            train_model=train_model,
            publish_dataset=publish,
            hf_repo=f"{os.getenv('HF_USERNAME', 'yourusername')}/brainbrew-dataset" if publish else None,
            api_key=openai_key or os.getenv("OPENAI_API_KEY"),
        )

        progress_bar = st.progress(0)
        try:
            final_path = run_distillation(cfg, source_path, lambda p: progress_bar.progress(p))
            st.success("✅ Dataset generated!")
            st.download_button("📥 Download dataset.jsonl", open(final_path).read(), "dataset.jsonl")
            if cfg.publish_dataset:
                st.balloons()
        except Exception as e:
            logger.exception("Failed")
            st.error(f"Generation failed: {e}")        publish_dataset=publish

    )

    dataset = run_distillation(cfg)

    st.success("Dataset generated")

    with open(dataset,"rb") as f:

        st.download_button(
            "Download dataset",
            f,
            file_name="dataset.jsonl"
        )
