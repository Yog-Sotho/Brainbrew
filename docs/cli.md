# Command line

The `brainbrew` command runs the same pipeline as the web app, without a
browser. It shares the run folders with the app.

```bash
brainbrew run notes.pdf report.txt --model gpt-4o-mini --size 200 --format sharegpt -o dataset.jsonl
brainbrew run docs/*.pdf --config cfg.yaml
brainbrew runs list            # add --json for machine-readable output
brainbrew runs show <run-id>   # the full manifest
brainbrew version
```

## Settings

`--config` takes a YAML or JSON file with any [configuration](configuration.md)
field. Command-line options override it.

```yaml
teacher_model: gpt-4o-mini
quality_mode: balanced
dataset_size: 300
output_format: chatml
sanitize_dataset: true
decontaminate: [gsm8k, mmlu]
```

Secrets are read **only** from the environment (`OPENAI_API_KEY`, `HF_TOKEN`).
A config file that contains `api_key` or `hf_token` is refused, so keys do not
end up in shell history or committed files. The endpoint comes from
`--base-url` or `OPENAI_BASE_URL`; the default model from `--model` or
`BRAINBREW_DEFAULT_MODEL` (else `gpt-4o-mini`).

## Behaviour

- A progress bar shows the current stage (`--quiet` hides it; `--verbose` shows
  info logs). The run's full log is always in `runs/<run-id>/run.log`.
- **Ctrl-C** cancels the run: in-flight requests are stopped and the run is
  recorded as cancelled.
- Exit codes: `0` success, `1` the run failed, `2` bad input or settings,
  `130` cancelled.

In the Docker image the CLI is `python cli.py`:

```bash
docker run --rm -v "$PWD:/data" -v brainbrew-runs:/app/runs -e OPENAI_API_KEY brainbrew-api \
    python cli.py run /data/notes.pdf -o /data/dataset.jsonl
```
