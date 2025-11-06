# Repository Guidelines

## Project Structure & Module Organization
The repository centres on a two-stage data pipeline. `download_raw_data/` holds the aria2-based fetch script and usage notes; `preprocess_raw_data/` contains per-tar processors, merge tools, and SLURM launchers. Keep bulky artifacts in `data/raw/` (source tars) and `data/intermediates/` (per-tar parquet outputs); these paths are `.gitignore`d on purpose. Root-level files (`pyproject.toml`, `uv.lock`) fix dependency versions for Python 3.11.

## Build, Test, and Development Commands
- `uv sync` – create/refresh the managed virtual environment and install declared dependencies.
- `source .venv/bin/activate` – activate the environment created by `uv`.
- `python preprocess_raw_data/process_tar.py --help` – review flags for extracting, validating, and logging a single tar run.
- `python preprocess_raw_data/merge_and_split.py --help` – inspect options for combining intermediates and writing deterministic splits.
- `./download_raw_data/download.sh <output_dir>` – download upstream archives (requires `aria2c` in PATH).

## Coding Style & Naming Conventions
Stick to PEP 8 with 4-space indentation and descriptive `snake_case`. Maintain module docstrings, type hints, and `NamedTuple`/dataclass wrappers to document schema expectations. Prefer `pathlib.Path`, `argparse`, and the existing `setup_logging` helpers instead of ad-hoc prints. Script filenames follow a `verb_subject.py` pattern—match it for new CLIs.

## Testing Guidelines
No automated suite exists yet; adopt `pytest` for any new tests. Place them under `tests/`, mirroring the source structure (e.g., `tests/test_process_tar.py`), and name functions `test_<condition>`. Use lightweight fixtures stored in `tests/fixtures/` to simulate tar metadata or image files, keeping large binaries out of Git. Target edge cases the pipeline already guards against: corrupt images, missing `d` lines, duplicate target IDs. Run `pytest` locally and note manual script runs in the PR body when integration validation is needed.

## Commit & Pull Request Guidelines
History uses short, imperative titles (“Update python requirements”), so follow that convention and scope commits narrowly. PRs should summarise intent, call out testing evidence (`pytest`, sample CLI invocations), and link issues or job IDs when applicable. Attach screenshots or log snippets for changes that affect monitoring dashboards, and mention any data reprocessing steps reviewers must repeat.

## Data & Job Orchestration Tips
Keep raw archives and generated parquet files out of Git; stage them in the `data/` tree or external storage. For cluster workloads, extend the SLURM scripts in `preprocess_raw_data/slurm/` and document new parameters alongside the script. Use `tempfile.TemporaryDirectory()` or scratch paths for transient extraction to avoid filling shared disks. Manage secrets through environment variables or cluster secret stores—never commit credentials.
