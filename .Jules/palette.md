# Palette's Journal — Critical UX/Accessibility Learnings

## 2026-03-03 - Scanned Document and Empty Parse Prevention
**Learning:** Document-to-dataset pipelines often ingest corrupt or scanned (image-only) PDFs which return empty text. Waiting until the expensive and slow distillation step to fail on empty/whitespace text degrades user experience and trust.
**Action:** Always provide an instant, interactive parsed text preview and length/word check right after file upload, so users can catch unreadable documents before triggering heavy LLM pipelines.
