# Contributing to GenRec

Thanks for your interest! This repository is primarily a research-code release,
so we keep the process lightweight.

## Issues

- **Bugs**: please include the exact command, the full traceback, your GPU /
  CUDA / PyTorch versions, and (if data-related) which dataset and how it was
  preprocessed.
- **Questions** about the paper or method are welcome as issues too.

## Pull requests

1. Fork, branch from `main`, and keep changes focused.
2. Install the dev tools and run the style hooks before committing:

   ```bash
   pip install -e ".[dev]"
   pre-commit install       # runs black / isort / flake8 on commit
   ```

3. Note that `depth_anything_3/` is vendored third-party code — please do not
   reformat or refactor it (see `depth_anything_3/VENDORED.md`).

## Code style

`black` + `isort` + `flake8`, line length 100 (configured in
`.pre-commit-config.yaml` / `pyproject.toml`).
