"""
The Generate page's sidebar: model endpoint and keys, data cleaning and
generation settings. `render_sidebar` draws it and returns what was chosen.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict

import streamlit as st

from pipeline.decontam import EVAL_SETS
from pipeline.pii import presidio_available
from ui.generate import CUSTOM, URL_POLICY_LABELS


class _Cleaning(TypedDict):
    use_semantic_chunking: bool
    enable_dedup: bool
    sanitize_dataset: bool
    pii_url_policy: str
    pii_presidio: bool
    decontaminate: list[str]


class _Generation(TypedDict):
    temperature: float
    max_new_tokens: int
    concurrency: int
    request_timeout: int
    judge_model: str
    judge_threshold: int
    embedding_model: str
    semantic_dedup_threshold: float


@dataclass(frozen=True)
class SidebarSettings:
    endpoint_label: str
    base_url: str | None
    custom_url_missing: bool
    api_key: str | None          # typed by the visitor, or the server key on the server endpoint
    hf_token: str                # typed by the visitor ("" when blank)
    hf_env_token: str            # the server's token; scoped in app.py before use
    use_semantic_chunking: bool
    enable_dedup: bool
    sanitize_dataset: bool
    pii_url_policy: str
    pii_presidio: bool
    decontaminate: list[str]
    temperature: float
    max_new_tokens: int
    concurrency: int
    request_timeout: int
    judge_model: str
    judge_threshold: int
    embedding_model: str
    semantic_dedup_threshold: float


def render_sidebar(endpoints: dict[str, str | None], server_label: str) -> SidebarSettings:
    with st.sidebar:
        st.header("⚙️ Settings")
        endpoint_label, base_url, api_key, hf_token, hf_env_token = _endpoint_and_keys(endpoints, server_label)
        st.divider()
        st.subheader("🧪 Data cleaning")
        cleaning = _cleaning()
        with st.expander("🔧 Generation settings"):
            generation = _generation()
    return SidebarSettings(
        endpoint_label=endpoint_label,
        base_url=base_url,
        custom_url_missing=endpoints[endpoint_label] == CUSTOM and not base_url,
        api_key=api_key,
        hf_token=hf_token,
        hf_env_token=hf_env_token,
        **cleaning,
        **generation,
    )


def _endpoint_and_keys(
    endpoints: dict[str, str | None], server_label: str
) -> tuple[str, str | None, str | None, str, str]:
    endpoint_label: str = st.selectbox(
        "Model endpoint",
        options=list(endpoints),
        help="Any OpenAI-compatible API: OpenAI, `vllm serve`, Ollama, llama.cpp, or a hosted provider.",
    )
    base_url = endpoints[endpoint_label]
    if base_url == CUSTOM:
        base_url = st.text_input("Endpoint URL", placeholder="https://my-server.example.com/v1") or None

    # Server-side secrets are never used as widget values: Streamlit sends widget
    # state to the browser, so a pre-filled password field discloses the key to
    # every visitor. The env value is applied server-side as a fallback instead,
    # and only for the server's own endpoint, so it cannot be redirected to a URL
    # the visitor controls.
    openai_env_key = os.getenv("OPENAI_API_KEY", "")
    use_server_key = bool(openai_env_key) and endpoint_label == server_label
    typed_key: str = st.text_input(
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
    api_key = typed_key or (openai_env_key if use_server_key else None)
    return endpoint_label, base_url, api_key, hf_token, hf_env_token


def _cleaning() -> _Cleaning:
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
            options=list(URL_POLICY_LABELS),
            format_func=URL_POLICY_LABELS.__getitem__,
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
    return {
        "use_semantic_chunking": use_semantic_chunking,
        "enable_dedup": enable_dedup,
        "sanitize_dataset": sanitize_dataset,
        "pii_url_policy": pii_url_policy,
        "pii_presidio": pii_presidio,
        "decontaminate": decontaminate,
    }


def _generation() -> _Generation:
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
    return {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "judge_model": judge_model,
        "judge_threshold": judge_threshold,
        "embedding_model": embedding_model,
        "semantic_dedup_threshold": semantic_dedup_threshold,
    }
