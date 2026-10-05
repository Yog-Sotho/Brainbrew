"""
Run history: every run in the runs folder (only your own when login is on),
with its status, live progress and cancel for active runs, results and
downloads for finished ones, and the full manifest details.
"""
from __future__ import annotations

import streamlit as st

from pipeline.jobs import get_runner, list_runs, run_state
from pipeline.runs import runs_base
from ui.common import current_owner, login_required, setup_page, visible_run
from ui.results import render_run

setup_page("Run history")
st.title("🗂️ Run history")

runs = list_runs(runs_base(), current_owner(), only_owner=login_required())
if not runs:
    st.info("No runs yet. Start one on the Generate page.")
    st.stop()

runner = get_runner()
rows = []
for run in runs:
    m = run.read_manifest()
    cost = (m.get("cost_usd") or {}).get("total")
    rows.append({
        "Run": run.run_id,
        "Status": run_state(m, runner),
        "Created": m.get("created_at", ""),
        "Pairs": (m.get("counts") or {}).get("exported"),
        "Grade": (m.get("quality") or {}).get("grade"),
        "Mode": (m.get("config") or {}).get("quality_mode"),
        "Teacher": ", ".join((m.get("models") or {}).get("teacher") or []),
        "Cost (USD)": None if cost is None else round(cost, 4),
    })
st.dataframe(rows, hide_index=True, use_container_width=True)

selected = st.selectbox("Open a run", [r["Run"] for r in rows],
                        format_func=lambda rid: f"{rid} · {next(r['Status'] for r in rows if r['Run'] == rid)}")
opened = visible_run(selected) if selected else None
if opened is not None:
    render_run(opened, details=True)
