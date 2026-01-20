# Repository Guidelines

## Project Structure & Module Organization
Core workflows live under `embedding_pipeline/` in task-focused modules: `preprocess/` normalizes tar shards and assembles splits; `generate/` runs DINOv2/DINOv3 inference; `difficulty/` computes visual-neighbour difficulty bands; `train/` runs the triplet-loss trainer; `publish/` exports the trained model to Hugging Face format. Shared utilities sit in `download_raw_data/`, and SLURM wrappers live under each stage’s `slurm/` directory (plus shared helpers in `slurm/common.sh`).

## Build, Test, and Development Commands
Install or refresh the environment with `uv sync` (Python 3.11). All workflows use a unified `config.toml` at the repo root with type-based sections (e.g. `[global]`, `[paths]`, `[embedding_model]`, `[training_model]`, `[training]`, `[data]`, `[infrastructure]`, `[logging]`). Prefer running pipeline stages as modules:\n+\n+- Preprocess: `uv run python -m embedding_pipeline.preprocess.process_tar --config config.toml` then `uv run python -m embedding_pipeline.preprocess.merge_and_split --config config.toml`\n+- Generate embeddings: `uv run python -m embedding_pipeline.generate.generate_embeddings --config config.toml`\n+- Train: `uv run python -m embedding_pipeline.train.train --config config.toml`\n+\n+SLURM submit scripts live under `embedding_pipeline/*/slurm/`.

## Coding Style & Naming Conventions
All new Python should be type-annotated, 4-space indented, and formatted with Black (`uv run black .`). Keep modules narrowly scoped and prefer `snake_case` for files, variables, and config keys; environment variables inherit the existing prefixes (e.g., `TRIPLET_TRAINING_`, `GENERATE_EMBEDDINGS_`). Reuse Rich logging helpers and fail fast when external resources (GPU, Arrow module, tar files) are missing.

## Testing Guidelines
There is no central pytest suite yet, so add lightweight validators near the code you touch (schema checks, metric asserts) and gate them behind `if __name__ == "__main__":`. Before submitting, run the exact command(s) your change affects against a small sample parquet/tar pair in `data/intermediates/` and confirm derived artifacts land in the expected `data/*` directory. For numerical work, log summary metrics (count, mean, std) so reviewers can spot regressions without full reruns.

## Commit & Pull Request Guidelines
Commits follow short, imperative summaries (`Add CI guard`, `Simplify wandb config`). Group related changes and avoid drive-by refactors. Every PR should describe the scenario, list reproducible commands (inputs plus config paths), mention any non-default resources (GPU type, Arrow module), and link to issues or benchmarks. Include screenshots or log excerpts when altering monitoring output, and ensure SLURM scripts remain executable (`chmod +x`) before review.

## Security & Configuration Tips
Never hard-code Hugging Face or Weights & Biases tokens; rely on environment variables loaded by the SLURM wrappers. On clusters, PyArrow typically comes from the site Arrow module; prefer keeping it optional for local development so it doesn’t conflict with module-provided Arrow. When handling tarballs, assume they may be corrupted: validate paths with `_normalize_relative_path` and keep writes scoped to `data/` to avoid leaks outside the workspace.
