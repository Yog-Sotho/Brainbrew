"""
Page setup shared by every page: page config, the optional login gate, and
who the current user is.

Every page must call `setup_page` first. Streamlit serves each page under its
own URL, so a gate that only ran in app.py could be bypassed by opening
another page directly.
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from pipeline.logs import configure_logging
from pipeline.runs import RunDir, open_run

LOGIN_ENV = "BRAINBREW_REQUIRE_LOGIN"


def login_required() -> bool:
    return os.getenv(LOGIN_ENV, "").strip().lower() in {"1", "true", "yes"}


def _auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except FileNotFoundError:  # no secrets.toml at all
        return False


def setup_page(title: str) -> None:
    """Page config, logging and (when BRAINBREW_REQUIRE_LOGIN is set) the login gate."""
    load_dotenv()
    configure_logging()
    st.set_page_config(page_title=f"Brainbrew · {title}", page_icon="🧠", layout="wide")
    if not login_required():
        return
    # Set BRAINBREW_REQUIRE_LOGIN=1 and an [auth] block in .streamlit/secrets.toml
    # (OIDC provider) before exposing the app beyond localhost.
    if not _auth_configured():
        # Fail closed: never fall through to an unauthenticated app.
        st.error(
            f"{LOGIN_ENV} is set but no [auth] section was found in "
            ".streamlit/secrets.toml. Configure an OIDC provider to continue."
        )
        st.stop()
    if not st.user.is_logged_in:
        st.title("🧠 Brainbrew")
        st.info("This Brainbrew instance is private. Please log in.")
        st.button("Log in", on_click=st.login, type="primary")
        st.stop()
    st.sidebar.button("Log out", on_click=st.logout)


def current_owner() -> str | None:
    """The logged-in user's identity, or None when login is off (single-user mode)."""
    if not login_required():
        return None
    email = st.user.get("email")
    sub = st.user.get("sub")
    return str(email or sub) if (email or sub) else None


def visible_run(run_id: str) -> RunDir | None:
    """Open *run_id* if it exists and the current user may see it.

    With login on, users only see their own runs, so a run id in a URL is not
    enough to read someone else's documents or dataset.
    """
    try:
        run = open_run(run_id)
    except (ValueError, FileNotFoundError):
        return None
    if login_required() and run.read_manifest().get("owner") != current_owner():
        return None
    return run


def server_hf_token_allowed(repo: str | None, namespace: str, login_on: bool) -> bool:
    """May the server's HF token publish to *repo*?

    In single-user mode (no login) the visitor is the operator. With login on,
    only repos in the operator's namespace (HF_USERNAME) qualify, so one user
    cannot use the server token to overwrite repos it can write to elsewhere.
    """
    if not login_on:
        return True
    return bool(namespace) and bool(repo) and str(repo).startswith(f"{namespace}/")


def custom_endpoints_allowed() -> bool:
    """May visitors pick their own model endpoint?

    BRAINBREW_ALLOW_CUSTOM_ENDPOINTS decides when it is set. When it is not set,
    custom endpoints are on in single-user mode and off with login on: on a
    shared server, a user-chosen URL makes the server send requests into its own
    network (SSRF), so the operator has to opt in.
    """
    raw = os.getenv("BRAINBREW_ALLOW_CUSTOM_ENDPOINTS", "").strip().lower()
    if raw:
        return raw not in {"0", "false", "no"}
    return not login_required()
