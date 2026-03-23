"""
Brainbrew - Production-grade synthetic dataset generator
Main Streamlit application
"""
import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import structlog

from config import DistillationConfig, QualityMode
from orchestrator import run_distillation

# Configure logging
load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger("INFO"))
logger = structlog.get_logger(__name__)

# Constants for validation
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FILES = 10

# Page configuration
st.set_page_config(page_title="Brainbrew", page_icon="🧠", layout="wide")
st.title("🧠 Brainbrew v1.0")
st.caption("Production-grade synthetic dataset generator")


def validate_file_content(uploaded_file) -> Optional[str]:
    """
    Validate file content based on MIME type and actual content.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Extracted text content or None if validation fails
    """
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File {uploaded_file.name} exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)}MB")
        return None

    # For PDF files, validate magic bytes
    if uploaded_file.type == "application/pdf":
        uploaded_file.seek(0)
        header = uploaded_file.read(4)
        if header != b'%PDF':
            st.error(f"Invalid file format for {uploaded_file.name}: Not a valid PDF file")
            return None
        uploaded_file.seek(0)

        try:
            from pdfminer.high_level import extract_text
            content = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Failed to extract text from {uploaded_file.name}: {e}")
            return None

        if not content or not content.strip():
            st.error(f"File {uploaded_file.name} appears to be empty or contains no extractable text")
            return None

        return content

    # For text files, validate encoding
    elif uploaded_file.type == "text/plain":
        try:
            content = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error(f"Invalid encoding for {uploaded_file.name}. Please upload UTF-8 encoded files.")
            return None
        except Exception as e:
            st.error(f"Failed to read {uploaded_file.name}: {e}")
            return None

        if not content or not content.strip():
            st.error(f"File {uploaded_file.name} is empty")
            return None

        return content

    # Unknown file type
    else:
        st.error(f"Unsupported file type: {uploaded_file.type} for file {uploaded_file.name}")
        return None


def get_api_key(use_vllm: bool) -> Optional[str]:
    """
    Get and validate API key based on mode.

    Args:
        use_vllm: Whether using vLLM (GPU) mode

    Returns:
        API key string or None
    """
    # Try to get from session state first, then environment
    if "openai_key" in st.session_state and st.session_state.openai_key:
        return st.session_state.openai_key

    # Fall back to environment variable
    return os.getenv("OPENAI_API_KEY") if not use_vllm else None


def calculate_estimated_cost(dataset_size: int, quality_mode: str, use_vllm: bool) -> tuple[float, str]:
    """
    Estimate cost and time for dataset generation.

    Args:
        dataset_size: Number of samples to generate
        quality_mode: Quality mode (fast, balanced, research)
        use_vllm: Whether using vLLM

    Returns:
        Tuple of (estimated_cost_usd, estimated_time)
    """
    # Base cost estimates (rough approximations)
    if use_vllm:
        # vLLM is free but requires GPU
        cost = 0.0
        time_per_sample = 2  # seconds
    else:
        # OpenAI API costs
        cost_per_1k = 0.03  # GPT-4o mini pricing
        cost = (dataset_size * cost_per_1k) / 1000
        time_per_sample = 1  # seconds

    # Quality mode affects time
    if quality_mode == "research":
        time_multiplier = 3
    elif quality_mode == "balanced":
        time_multiplier = 2
    else:
        time_multiplier = 1

    total_seconds = dataset_size * time_per_sample * time_multiplier

    if total_seconds < 60:
        time_str = f"{total_seconds} seconds"
    elif total_seconds < 3600:
        time_str = f"{total_seconds // 60} minutes"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        time_str = f"{hours}h {minutes}m"

    return cost, time_str


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Advanced Settings")

    # Model selection
    use_vllm = st.checkbox("Use vLLM (GPU required)", value=True)

    # API key inputs
    st.subheader("API Keys")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Required only if not using vLLM"
    )
    hf_token = st.text_input(
        "Hugging Face Token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Required for publishing to Hugging Face"
    )

    # Store in session state for persistence
    st.session_state.openai_key = openai_key
    st.session_state.hf_token = hf_token

    # Model configuration
    st.subheader("Model Configuration")
    teacher_model = st.text_input(
        "Teacher Model(s)",
        value="gpt-4o" if not use_vllm else "meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model to use for dataset generation"
    )

    # Quality settings
    quality_mode = st.selectbox(
        "Quality Mode",
        [q.value for q in QualityMode],
        index=1,
        help="Higher quality = more processing time"
    )

    # Dataset size with cost estimation
    dataset_size = st.slider(
        "Target Dataset Size",
        min_value=500,
        max_value=20000,
        value=2000,
        step=500
    )

    # Show cost/time estimation
    estimated_cost, estimated_time = calculate_estimated_cost(dataset_size, quality_mode, use_vllm)
    if use_vllm:
        st.info(f"⏱️ Estimated time: {estimated_time}")
    else:
        st.info(f"💰 Estimated cost: ${estimated_cost:.2f} | ⏱️ {estimated_time}")

    # Training options
    st.subheader("Training & Publishing")
    train_model = st.checkbox(
        "Auto-train LoRA adapter",
        value=False,
        help="Train a LoRA adapter after dataset generation"
    )
    publish = st.checkbox(
        "Publish to Hugging Face",
        value=False,
        help="Automatically publish dataset to Hugging Face Hub"
    )


# Main area
st.header("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload documents (PDF/TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help=f"Maximum {MAX_FILES} files, {MAX_FILE_SIZE // (1024*1024)}MB per file"
)

# File validation display
if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.error(f"Maximum {MAX_FILES} files allowed. Please remove {len(uploaded_files) - MAX_FILES} files.")
        st.stop()

    st.success(f"📎 {len(uploaded_files)} file(s) ready for processing")

    # Show file details
    with st.expander("View uploaded files"):
        for uploaded in uploaded_files:
            size_mb = uploaded.size / (1024 * 1024)
            st.write(f"- {uploaded.name} ({size_mb:.2f} MB)")

# Generate button
if st.button("🚀 Generate Dataset", type="primary", disabled=not uploaded_files):
    if not uploaded_files:
        st.error("Please upload at least one document")
        st.stop()

    # Validate files before processing
    valid_contents = []
    for uploaded in uploaded_files:
        content = validate_file_content(uploaded)
        if content is None:
            st.stop()
        valid_contents.append(content)

    # Combine all content
    combined_content = "\n\n".join(valid_contents)

    if not combined_content.strip():
        st.error("No valid content extracted from uploaded files")
        st.stop()

    # Show processing status
    with st.spinner("Processing documents..."):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.txt"
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(combined_content)

            # Build configuration
            hf_username = os.getenv("HF_USERNAME", "yourusername")
            hf_repo = f"{hf_username}/brainbrew-dataset" if publish else None

            cfg = DistillationConfig(
                teacher_model=teacher_model,
                quality_mode=QualityMode(quality_mode),
                dataset_size=dataset_size,
                use_vllm=use_vllm,
                train_model=train_model,
                publish_dataset=publish,
                hf_repo=hf_repo,
                api_key=openai_key if not use_vllm else None,
            )

            # Run distillation with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Initializing pipeline...")
                progress_bar.progress(10)

                final_path = run_distillation(
                    cfg,
                    source_path,
                    lambda p: progress_bar.progress(min(p, 100))
                )

                progress_bar.progress(100)
                status_text.text("Complete!")

                st.success("✅ Dataset generated successfully!")

                # Display download button
                with open(final_path, "r", encoding="utf-8") as f:
                    file_content = f.read()

                st.download_button(
                    "📥 Download dataset.jsonl",
                    file_content,
                    "dataset.jsonl",
                    mime="application/json"
                )

                # Show balloons for successful publish
                if cfg.publish_dataset:
                    st.balloons()
                    st.success(f"🚀 Dataset published to https://huggingface.co/{cfg.hf_repo}")

            except ValueError as e:
                logger.warning(f"Validation error: {e}")
                st.error(f"Configuration error: {e}")
            except RuntimeError as e:
                logger.error(f"Runtime error: {e}")
                st.error(f"Generation failed: {e}")
            except Exception as e:
                logger.exception("Unexpected error")
                st.error(f"An unexpected error occurred: {e}")

# Footer
st.markdown("---")
st.caption("🧠 Brainbrew v1.0 | Powered by distilabel, vLLM, and Unsloth")
