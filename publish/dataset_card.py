"""
Dataset and model cards for Hugging Face uploads, built from a run's manifest.

The dataset card records provenance (Brainbrew version, run, date, models,
seed, settings), every filter count, the quality report and a sample record,
so anyone who finds the dataset on the Hub can see how it was made. Source
document names are not published (they may be private); a SHA-256 of the
source text identifies it instead.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from pipeline.version import __version__

BRAINBREW_URL = "https://github.com/Yog-Sotho/Brainbrew"
LICENSES = ("other", "unknown", "cc-by-4.0", "cc-by-sa-4.0", "cc-by-nc-4.0", "cc0-1.0", "odc-by",
            "mit", "apache-2.0")
_TASKS = {"alpaca": "text-generation", "sharegpt": "text-generation",
          "chatml": "text-generation", "openai": "text-generation"}


def size_category(n: int) -> str:
    for bound, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < bound:
            return label
    return "100K<n<1M"


def _front_matter(data: dict[str, Any]) -> str:
    body: str = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return "---\n" + body + "---\n"


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _first_record(dataset_path: Path) -> str | None:
    try:
        with open(dataset_path, encoding="utf-8") as fh:
            line = fh.readline().strip()
        return json.dumps(json.loads(line), indent=2, ensure_ascii=False) if line else None
    except (OSError, ValueError):
        return None


def _row(label: str, value: Any) -> str:
    return f"| {label} | {value} |"


def dataset_card(
    repo: str,
    manifest: dict[str, Any],
    dataset_path: Path,
    source_path: Path | None = None,
    license: str = "other",
) -> str:
    """README.md for the dataset repo."""
    cfg = manifest.get("config") or {}
    counts = manifest.get("counts") or {}
    quality = manifest.get("quality") or {}
    models = manifest.get("models") or {}
    fmt = str(cfg.get("output_format", "alpaca"))
    records = int(counts.get("exported") or quality.get("record_count") or 0)
    judge = models.get("judge")

    meta = {
        "license": license,
        "pretty_name": repo.split("/", 1)[-1],
        "size_categories": [size_category(records)],
        "task_categories": ["question-answering", _TASKS.get(fmt, "text-generation")],
        "tags": ["synthetic", "brainbrew", "instruction-tuning", fmt],
    }
    lines = [
        _front_matter(meta),
        f"# {repo}",
        "",
        f"Synthetic question/answer pairs generated with [Brainbrew]({BRAINBREW_URL}) v"
        f"{manifest.get('brainbrew_version') or __version__}. Each question was written from a passage of "
        "the source documents and answered from that passage only"
        + (", then scored by an LLM judge." if judge else "."),
        "",
        *_summary_section(records, fmt, quality),
        "",
        *_method_section(manifest, cfg, models),
        "",
        *_filtering_section(manifest, counts, records, judge),
        "",
        *_source_section(dataset_path, source_path),
        "",
        *_license_section(license),
        "",
    ]
    return "\n".join(lines)


def _summary_section(records: int, fmt: str, quality: dict[str, Any]) -> list[str]:
    return [
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        _row("Records", records),
        _row("Format", fmt),
        _row("Quality grade", quality.get("grade", "–")),
        _row("Average answer length", f"{quality.get('avg_output_length', 0):.0f} characters"),
        _row("Unique answers", f"{quality.get('unique_ratio', 0):.0%}"),
    ]


def _method_section(manifest: dict[str, Any], cfg: dict[str, Any], models: dict[str, Any]) -> list[str]:
    judge = models.get("judge")
    return [
        "## How it was made",
        "",
        "| | |",
        "|---|---|",
        _row("Teacher model(s)", ", ".join(models.get("teacher") or []) or "–"),
        _row("Judge", f"{judge} (kept pairs scoring at least {cfg.get('judge_threshold')} / 5 on faithfulness, "
                      "helpfulness and correctness)" if judge else "none (Fast mode: pattern filters only)"),
        _row("Paraphrase dedup embeddings", models.get("embedding") or "–"),
        _row("Quality mode", cfg.get("quality_mode", "–")),
        _row("Temperature", cfg.get("temperature", "–")),
        _row("Seed", manifest.get("seed", "–")),
        _row("Run", f"`{manifest.get('run_id', '–')}` ({manifest.get('finished_at') or manifest.get('started_at') or '–'})"),
    ]


def _filtering_section(manifest: dict[str, Any], counts: dict[str, Any], records: int, judge: Any) -> list[str]:
    """Every step between the source chunks and the published records, with its count."""
    gen = manifest.get("generation") or {}
    lines = [
        "## Filtering",
        "",
        "| Step | Count |",
        "|---|---|",
        _row("Source chunks", counts.get("chunks", "–")),
        _row("Questions generated", gen.get("questions_generated", "–")),
        _row("Questions dropped (malformed or not self-contained)", gen.get("questions_dropped", "–")),
        _row("Answers written", gen.get("answers", "–")),
    ]
    for reason, n in sorted((gen.get("answers_filtered") or {}).items()):
        lines.append(_row(f"Answers removed: {reason}", n))
    if judge:
        lines.append(_row("Rejected by the judge", gen.get("judge_rejected", 0)))
    lines.append(_row("Near-duplicates removed", gen.get("duplicates", 0)))
    if gen.get("semantic_duplicates"):
        lines.append(_row("Paraphrases removed", gen["semantic_duplicates"]))
    for bench, n in (manifest.get("decontamination") or {}).items():
        lines.append(_row(f"Overlap with {bench} removed", n))
    if sanitizer := manifest.get("sanitizer"):
        lines.append(_row("Removed by sanitizing", sanitizer.get("total", 0) - sanitizer.get("kept", 0)))
        lines.append(_row("Records with PII redacted", sanitizer.get("pii_redacted", 0)))
    lines.append(_row("Published", records))
    return lines


def _source_section(dataset_path: Path, source_path: Path | None) -> list[str]:
    """Source documents are identified by a hash only; their names are never published."""
    digest = _sha256(source_path) if source_path else None
    lines = [
        "## Source documents",
        "",
        "Document names are not published. "
        + (f"The concatenated source text has SHA-256 `{digest}`." if digest else ""),
    ]
    if example := _first_record(dataset_path):
        lines += ["", "## Example record", "", "```json", example, "```"]
    return lines


def _license_section(license: str) -> list[str]:
    return [
        "## License and limitations",
        "",
        f"The `{license}` license was chosen by the publisher. Data derived from source documents may be "
        "subject to the terms of those documents; check them before redistributing.",
        "",
        "All questions and answers are model-generated. The filters and the judge reduce, but do not "
        "eliminate, errors and unsupported claims. Review the data before training on it for "
        "high-stakes use.",
    ]


def model_card(repo: str, manifest: dict[str, Any], dataset_repo: str | None, license: str = "other") -> str:
    """README.md for the LoRA adapter repo."""
    cfg = manifest.get("config") or {}
    base = str(cfg.get("base_model", ""))
    meta: dict[str, Any] = {
        "base_model": base,
        "library_name": "peft",
        "license": license,
        "tags": ["lora", "peft", "trl", "sft", "brainbrew"],
    }
    if dataset_repo:
        meta["datasets"] = [dataset_repo]
    records = (manifest.get("counts") or {}).get("exported", "–")
    data_ref = f"[{dataset_repo}](https://huggingface.co/datasets/{dataset_repo})" if dataset_repo else "a Brainbrew dataset"
    return "\n".join([
        _front_matter(meta),
        f"# {repo}",
        "",
        f"A LoRA adapter (rank {cfg.get('lora_rank', '–')}) for `{base}`, trained with TRL + PEFT by "
        f"[Brainbrew]({BRAINBREW_URL}) on {records} synthetic question/answer pairs from {data_ref}. "
        "The loss is computed on the answers only.",
        "",
        "## Usage",
        "",
        "```python",
        "from peft import AutoPeftModelForCausalLM",
        "from transformers import AutoTokenizer",
        "",
        f'model = AutoPeftModelForCausalLM.from_pretrained("{repo}")',
        f'tokenizer = AutoTokenizer.from_pretrained("{base}")',
        "```",
        "",
        "## Limitations",
        "",
        "Trained on model-generated data for one run; evaluate it on your own task before relying on it.",
        "",
    ])
