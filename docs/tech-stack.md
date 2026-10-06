# Tech stack

| Area | Library |
|---|---|
| Web UI | Streamlit (fragments, multipage, OIDC login) |
| Command line | Typer, Rich |
| Model calls | openai SDK (async), tenacity |
| Validation and records | Pydantic |
| Documents | pdfminer.six, LangChain text splitters |
| Dedup | numpy, xxhash (MinHash-LSH) |
| Benchmarks and publishing | Hugging Face `datasets`, `huggingface_hub` |
| PII (optional) | Presidio, spaCy |
| Training (optional) | PyTorch, Transformers, TRL, PEFT, bitsandbytes |
| Local serving (optional) | vLLM |
| Logging | structlog on the standard library |
| Tooling | uv, ruff, mypy, pytest, Hypothesis, mutmut, pre-commit, MkDocs |
