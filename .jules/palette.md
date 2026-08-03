# Palette's Journal - Critical UX/Accessibility Learnings

This journal is a collection of critical UX and accessibility learnings from Palette's work on the codebase.

## 2025-01-20 - Proactive Form Validation in Streamlit
**Learning:** In Streamlit applications, deferring validation of mandatory parameters (such as file uploads or API keys) until *after* the user triggers a submit button leads to a frustrating user experience, often exposing raw tracebacks or generic error messages. Proactively displaying localized validation warnings (`st.warning` or `st.error`) and dynamically disabling the submission button (`disabled=True`) with a helpful descriptive tooltip (`help="..."`) dramatically enhances accessibility and visual clarity, making the interface self-correcting and highly intuitive.
**Action:** Always implement inline reactive validation error lists and dynamically disable primary action buttons when mandatory configuration parameters are missing or invalid in any Streamlit dashboard.

## 2026-07-30 - Proactive Uploaded File Verification & Stream Resets in Streamlit
**Learning:** When using Streamlit's `st.file_uploader`, checking file metadata (such as filename safety and file size) only after submission can cause silent processing skips or empty-file runtime crashes. Proactively validating uploaded files in the main validation flow, along with immediately resetting their stream pointers via `uploaded.seek(0)`, prevents these downstream failures and guarantees that files are fully readable for downstream engines (like PDF miners or chunkers).
**Action:** Always proactively validate uploaded files' properties (e.g., names, sizes) alongside other input parameters, and ensure that stream pointers are explicitly reset before any read operations to maintain a flawless file-processing pipeline.

## 2026-08-03 - Native Descriptive Tooltips for Streamlit Form Controls
**Learning:** Complex configuration panels can be visually overwhelming and inaccessible to screen readers without descriptive tooltips. Utilizing Streamlit's native `help="..."` parameter for all interactive input widgets is a robust, lightweight approach that automatically adds clean visual helper icons (`?`) and registers accessible descriptions to help screen readers announce the input's purpose and requirements without introducing any custom CSS or HTML overhead.
**Action:** Always provide rich native tooltips for every key interactive widget (checkboxes, sliders, selectboxes, buttons, file uploaders) to establish a consistent visual and accessible UX vocabulary.
