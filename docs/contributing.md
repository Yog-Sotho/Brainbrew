# Contributing

1. Fork and branch.
2. `uv sync`, then `uvx pre-commit install` so ruff and mypy run before each commit.
3. Make the change with tests. Run `uv run pytest`.
4. If you changed dependencies, run `uv lock` and regenerate the exports
   (`uv export --frozen --no-dev --no-emit-project [--extra ...] -o requirements*.txt`);
   CI checks that they match.
5. Add a line to `CHANGELOG.md` under *Unreleased*. Change notes belong there,
   not in code comments.
6. Open a pull request. CI runs lint, types (strict on the core), tests with a
   coverage gate, LoRA training on CPU, the lockfile check, pip-audit, gitleaks,
   CodeQL and a Docker build.

Docs live in `docs/` as Markdown and build with `uv run --group docs mkdocs serve`.
