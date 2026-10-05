"""
Brainbrew command line: the same pipeline as the web app, without a browser.

    brainbrew run docs/*.pdf --config cfg.yaml
    brainbrew run notes.txt --model gpt-4o-mini --size 200 --format sharegpt -o dataset.jsonl
    brainbrew runs list
    brainbrew runs show <run-id>

Settings come from an optional YAML/JSON config file (any DistillationConfig
field), overridden by command-line options. Secrets are read only from the
environment (OPENAI_API_KEY, HF_TOKEN), never from files or arguments, so they
do not end up in shell history or committed configs.

Exit status: 0 success, 1 the run failed, 2 bad input, 130 cancelled (Ctrl-C).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from config import DistillationConfig, OutputFormat, QualityMode
from pipeline.jobs import list_runs, run_state
from pipeline.logs import configure_logging
from pipeline.runs import RunCancelled, open_run, runs_base
from pipeline.service import new_run, read_documents
from pipeline.version import __version__

app = typer.Typer(
    help="Brainbrew: grounded synthetic training datasets from your documents.",
    no_args_is_help=True,
    add_completion=False,
)
runs_app = typer.Typer(help="Inspect past runs.", no_args_is_help=True)
app.add_typer(runs_app, name="runs")

err = Console(stderr=True)
out = Console()
SECRET_FIELDS = ("api_key", "hf_token")
EXIT_FAILED, EXIT_USAGE, EXIT_CANCELLED = 1, 2, 130


def _fail(message: str, code: int = EXIT_USAGE) -> typer.Exit:
    err.print(f"[red]error:[/red] {message}")
    return typer.Exit(code)


def load_config_file(path: Path) -> dict[str, Any]:
    """Settings from a YAML or JSON file. Secrets are refused: they belong in the environment."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise _fail(f"cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise _fail(f"config {path} must be a mapping of settings")
    if found := [k for k in SECRET_FIELDS if k in data]:
        raise _fail(
            f"config {path} contains {', '.join(found)}. Put secrets in the environment "
            "(OPENAI_API_KEY, HF_TOKEN), not in config files."
        )
    return data


def build_config(file_settings: dict[str, Any], overrides: dict[str, Any]) -> DistillationConfig:
    settings = {**file_settings, **{k: v for k, v in overrides.items() if v is not None}}
    # Same default as the web app.
    settings.setdefault("teacher_model", os.getenv("BRAINBREW_DEFAULT_MODEL", "").strip() or "gpt-4o-mini")
    settings["api_key"] = os.getenv("OPENAI_API_KEY") or None
    settings["hf_token"] = os.getenv("HF_TOKEN") or None
    try:
        return DistillationConfig(**settings)
    except ValidationError as exc:
        lines = []
        for e in exc.errors():
            field = ".".join(str(p) for p in e["loc"])
            for msg in str(e["msg"]).removeprefix("Value error, ").splitlines():
                lines.append(f"  {field}: {msg}" if field else f"  {msg}")
        raise _fail("invalid settings:\n" + "\n".join(lines)) from exc


@app.command()
def run(
    documents: Annotated[list[Path], typer.Argument(help="PDF or text files.", exists=True, dir_okay=False)],
    config: Annotated[Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False,
                                                help="YAML/JSON file with settings.")] = None,
    model: Annotated[str | None, typer.Option(
        "--model", "-m", help="Teacher model(s), comma-separated (default: BRAINBREW_DEFAULT_MODEL or gpt-4o-mini).",
    )] = None,
    base_url: Annotated[str | None, typer.Option(help="OpenAI-compatible endpoint (default: OPENAI_BASE_URL).",
                                                 envvar="OPENAI_BASE_URL")] = None,
    size: Annotated[int | None, typer.Option("--size", "-n", help="Target number of pairs.")] = None,
    mode: Annotated[QualityMode | None, typer.Option(help="fast, balanced or research.")] = None,
    output_format: Annotated[OutputFormat | None, typer.Option("--format", help="Export format.")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Also copy the dataset here.")] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No progress bar.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show info logs.")] = False,
) -> None:
    """Generate a dataset from DOCUMENTS."""
    configure_logging(level="INFO" if verbose else "WARNING")
    cfg = build_config(load_config_file(config) if config else {}, {
        "teacher_model": model, "base_url": base_url, "dataset_size": size,
        "quality_mode": mode, "output_format": output_format,
    })
    text, problems = read_documents((p.name, p.read_bytes()) for p in documents)
    for problem in problems:
        err.print(f"[yellow]warning:[/yellow] {problem}")
    try:
        target = new_run(text)
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    from orchestrator import run_distillation

    cancel = threading.Event()
    outcome: dict[str, Any] = {}
    state: dict[str, Any] = {"pct": 0, "stage": "Starting"}

    def work() -> None:
        try:
            outcome["result"] = run_distillation(
                cfg, target.source, lambda pct, stage: state.update(pct=pct, stage=stage),
                run=target, cancel=cancel,
            )
        except BaseException as exc:  # reported by the main thread
            outcome["error"] = exc

    err.print(f"Run [bold]{target.run_id}[/bold] · {target.root}")
    worker = threading.Thread(target=work, name="brainbrew-cli-run")
    worker.start()
    try:
        if quiet:
            while worker.is_alive():
                worker.join(0.2)
        else:
            with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
                          TimeElapsedColumn(), console=err, transient=True) as bar:
                task = bar.add_task("Starting", total=100)
                while worker.is_alive():
                    bar.update(task, completed=float(state["pct"]), description=str(state["stage"]))
                    worker.join(0.2)
    except KeyboardInterrupt:
        err.print("Cancelling… (in-flight requests are being stopped)")
        cancel.set()
        worker.join()

    error = outcome.get("error")
    if isinstance(error, RunCancelled):
        err.print(f"[yellow]Cancelled.[/yellow] Run {target.run_id} is marked cancelled.")
        raise typer.Exit(EXIT_CANCELLED)
    if error is not None:
        raise _fail(f"run {target.run_id} failed: {error}", EXIT_FAILED)

    result = outcome["result"]
    manifest = result.run.read_manifest()
    cost = (manifest.get("cost_usd") or {}).get("total")
    out.print(f"[green]✓[/green] {result.record_count} pairs · grade {result.quality['grade']} · "
              f"cost {'unknown' if cost is None else f'${cost:.4f}'}")
    out.print(f"  dataset: {result.dataset_path}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result.dataset_path, output)
        out.print(f"  copied to: {output}")


@runs_app.command("list")
def runs_list(
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many runs to show.")] = 20,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Recent runs, newest first."""
    rows = []
    for r in list_runs(runs_base())[:limit]:
        m = r.read_manifest()
        rows.append({
            "run_id": r.run_id,
            "status": run_state(m),
            "created_at": m.get("created_at"),
            "pairs": (m.get("counts") or {}).get("exported"),
            "grade": (m.get("quality") or {}).get("grade"),
            "cost_usd": (m.get("cost_usd") or {}).get("total"),
        })
    if as_json:
        out.print_json(json.dumps(rows))
        return
    table = Table()
    table.add_column("Run", no_wrap=True, min_width=22)  # ids are copied into `runs show`
    for name in ("Status", "Created", "Pairs", "Grade", "Cost"):
        table.add_column(name, overflow="fold")
    for row in rows:
        cost = row["cost_usd"]
        table.add_row(row["run_id"], row["status"], str(row["created_at"] or ""), str(row["pairs"] or ""),
                      str(row["grade"] or ""), "" if cost is None else f"${cost:.4f}")
    out.print(table)


@runs_app.command("show")
def runs_show(run_id: Annotated[str, typer.Argument(help="A run id from `brainbrew runs list`.")]) -> None:
    """The full manifest of one run."""
    try:
        r = open_run(run_id)
    except (ValueError, FileNotFoundError) as exc:
        raise _fail(str(exc)) from exc
    manifest = r.read_manifest()
    manifest["status"] = run_state(manifest)
    out.print_json(json.dumps(manifest))


@app.command()
def version() -> None:
    """Print the Brainbrew version."""
    out.print(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
