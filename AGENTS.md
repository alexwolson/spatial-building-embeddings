# Repository Guidelines

## Project Structure & Module Organization
Core workflows live in task-focused packages: `preprocess_raw_data/` normalizes tar shards, merges parquet intermediates, and writes `data/merged/`; `generate_embeddings/` houses the DINOv2 inference driver; `difficulty_metadata/` computes BallTree difficulty bands; `train_specialized_embeddings/` runs the triplet-loss trainer. Shared assets sit in `download_raw_data/` (ingest helpers), `data/raw|intermediates|merged|difficulty/` (artifacts), and mirrored `slurm/` folders for Alliance batch jobs.

## Build, Test, and Development Commands
Install or refresh the environment with `uv sync` (Python 3.11). Run preprocessing with `uv run python preprocess_raw_data/process_tar.py --config preprocess_raw_data/config.example.toml` followed by `merge_and_split.py` to materialize splits. Generate embeddings for a shard via `uv run python generate_embeddings/generate_embeddings.py --config <config.toml>` on a GPU-enabled host. Train the projection head with `uv run python train_specialized_embeddings/train.py --config train_specialized_embeddings/config.toml`. Cluster wrappers simply pass these configs into `slurm/submit_*.sh`.

## Coding Style & Naming Conventions
All new Python should be type-annotated, 4-space indented, and formatted with Black (`uv run black .`). Keep modules narrowly scoped and prefer `snake_case` for files, variables, and config keys; environment variables inherit the existing prefixes (e.g., `TRIPLET_TRAINING_`, `GENERATE_EMBEDDINGS_`). Reuse Rich logging helpers and fail fast when external resources (GPU, Arrow module, tar files) are missing.

## Testing Guidelines
There is no central pytest suite yet, so add lightweight validators near the code you touch (schema checks, metric asserts) and gate them behind `if __name__ == "__main__":`. Before submitting, run the exact command(s) your change affects against a small sample parquet/tar pair in `data/intermediates/` and confirm derived artifacts land in the expected `data/*` directory. For numerical work, log summary metrics (count, mean, std) so reviewers can spot regressions without full reruns.

## Commit & Pull Request Guidelines
Commits follow short, imperative summaries (`Add CI guard`, `Simplify wandb config`). Group related changes and avoid drive-by refactors. Every PR should describe the scenario, list reproducible commands (inputs plus config paths), mention any non-default resources (GPU type, Arrow module), and link to issues or benchmarks. Include screenshots or log excerpts when altering monitoring output, and ensure SLURM scripts remain executable (`chmod +x`) before review.

## Security & Configuration Tips
Never hard-code Hugging Face or Weights & Biases tokens; rely on environment variables loaded by the SLURM wrappers. PyArrow comes from the cluster’s Arrow module, so keep it out of `pyproject.toml`. When handling tarballs, assume they may be corrupted: validate paths with `_normalize_relative_path` and keep writes scoped to `data/` to avoid leaks outside the workspace.
