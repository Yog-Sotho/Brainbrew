# Palette's Journal - Critical UX/Accessibility Learnings

This journal is a collection of critical UX and accessibility learnings from Palette's work on the codebase.

## 2025-01-20 - Proactive Form Validation in Streamlit
**Learning:** In Streamlit applications, deferring validation of mandatory parameters (such as file uploads or API keys) until *after* the user triggers a submit button leads to a frustrating user experience, often exposing raw tracebacks or generic error messages. Proactively displaying localized validation warnings (`st.warning` or `st.error`) and dynamically disabling the submission button (`disabled=True`) with a helpful descriptive tooltip (`help="..."`) dramatically enhances accessibility and visual clarity, making the interface self-correcting and highly intuitive.
**Action:** Always implement inline reactive validation error lists and dynamically disable primary action buttons when mandatory configuration parameters are missing or invalid in any Streamlit dashboard.
