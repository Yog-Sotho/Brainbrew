"""
The run panel: live progress with a cancel button while a run is active, and
the results (quality, preview, downloads, details) once it has finished.
Used by the generate page and the run-history page.
"""
from __future__ import annotations

import json
from typing import Any

import streamlit as st
import structlog

from pipeline.jobs import get_runner, run_state
from pipeline.records import Record
from pipeline.runs import RunDir

logger = structlog.get_logger(__name__)

POLL_SECONDS = 2
_GRADE_EMOJI = {"SUPER": "🟢", "GOOD": "🔵", "NORMAL": "🟡", "BAD": "🟠", "DISASTER": "🔴"}
LIVE_STATES = ("queued", "running", "running elsewhere")


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


def render_run(run: RunDir, details: bool = False) -> None:
    """Everything about one run, for whatever state it is in."""
    manifest = run.read_manifest()
    state = run_state(manifest, get_runner())
    if state in LIVE_STATES:
        _live_panel(run)
    elif state == "succeeded":
        _results(run, manifest)
    elif state == "failed":
        st.error(f"Generation failed: {manifest.get('error') or 'unknown error'}")
    elif state == "cancelled":
        st.warning("This run was cancelled.")
    elif state == "interrupted":
        st.warning("This run was interrupted (the process running it stopped). Start a new run.")
    if details:
        _details(run, manifest)


@st.fragment(run_every=POLL_SECONDS)
def _live_panel(run: RunDir) -> None:
    runner = get_runner()
    job = runner.get(run.run_id)
    manifest = run.read_manifest()
    state = run_state(manifest, runner)
    if state not in LIVE_STATES:
        st.rerun()  # finished: re-render the whole page with the results
    pct = job.progress if job else int(manifest.get("progress") or 0)
    stage = job.stage if job else str(manifest.get("stage") or state.capitalize())
    st.progress(pct, text=f"{stage} · {pct}%")
    if state == "running elsewhere":
        st.caption("This run belongs to another Brainbrew process (for example the CLI).")
    elif st.button("⏹ Cancel run", key=f"cancel-{run.run_id}"):
        runner.cancel(run.run_id)
        st.toast("Cancelling… in-flight requests are being stopped.")
    st.caption(f"Run `{run.run_id}` · it keeps running if you close or reload this page.")


def _results(run: RunDir, manifest: dict[str, Any]) -> None:
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
    if manifest.get("dataset_file") and dataset.is_file():
        st.download_button(
            "📥 Download dataset",
            dataset.read_bytes(),
            file_name=f"brainbrew-{run.run_id}-{dataset.name}",
            mime="application/jsonl",
            on_click="ignore",
            key=f"dl-dataset-{run.run_id}",
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
            key=f"dl-adapter-{run.run_id}",
        )
    if repo := manifest.get("published_repo"):
        st.success(f"Published to https://huggingface.co/datasets/{repo}")
    if model_repo := manifest.get("published_model_repo"):
        st.success(f"Adapter published to https://huggingface.co/{model_repo}")
    st.caption(f"Run `{run.run_id}` · files saved in `{run.root}`")


def _details(run: RunDir, manifest: dict[str, Any]) -> None:
    with st.expander("🔎 Run details"):
        cost = (manifest.get("cost_usd") or {}).get("total")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cost", "unknown" if cost is None and manifest.get("usage") else f"${cost or 0:.4f}")
        col2.metric("Seed", manifest.get("seed", "–"))
        col3.metric("Time", f"{sum((manifest.get('timings') or {}).values()):.0f} s")
        st.json({k: manifest.get(k) for k in
                 ("status", "created_at", "started_at", "finished_at", "brainbrew_version", "models",
                  "counts", "generation", "usage", "cost_usd", "timings", "decontamination", "sanitizer")
                 if manifest.get(k) is not None}, expanded=False)
        for path, label, mime in ((run.rejected, "Rejected pairs", "application/jsonl"),
                                  (run.log, "Run log", "application/jsonl")):
            if path.is_file() and path.stat().st_size:
                st.download_button(f"⬇️ {label}", path.read_bytes(), file_name=f"{run.run_id}-{path.name}",
                                   mime=mime, on_click="ignore", key=f"dl-{path.name}-{run.run_id}")
