"""Summarize unique target and patch identifiers in intermediate parquet files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count unique targetIDs and patchIDs from parquet intermediates."
    )
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        default=Path("data/intermediates"),
        help="Directory containing parquet intermediates (default: data/intermediates).",
    )
    parser.add_argument(
        "--glob",
        default="*.parquet",
        help="Glob pattern for parquet files to include (default: *.parquet).",
    )
    return parser.parse_args()


def collect_parquet_files(intermediates_dir: Path, glob: str) -> Iterable[Path]:
    if not intermediates_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {intermediates_dir}")
    return sorted(intermediates_dir.glob(glob))


DATASET_COLUMN_CANDIDATES = ("dataset_id",)
TARGET_COLUMN_CANDIDATES = ("targetID", "target_id")
PATCH_COLUMN_CANDIDATES = ("patchID", "patch_id")


def _resolve_column(schema_names: Iterable[str], candidates: Tuple[str, ...], parquet_path: Path) -> str:
    for candidate in candidates:
        if candidate in schema_names:
            return candidate
    raise ValueError(
        f"None of the expected columns {candidates} found in {parquet_path}. "
        f"Available columns: {', '.join(schema_names)}"
    )


def count_unique_ids(files: Iterable[Path]) -> Dict[str, int]:
    target_ids = set()
    patch_ids = set()
    dataset_ids = set()
    dataset_target_ids = set()
    dataset_patch_ids = set()
    dataset_target_patch_ids = set()

    for parquet_path in files:
        logger.debug("Processing %s", parquet_path)
        schema = pq.read_schema(parquet_path)
        schema_names = schema.names

        dataset_col = next((col for col in DATASET_COLUMN_CANDIDATES if col in schema_names), None)
        target_col = _resolve_column(schema_names, TARGET_COLUMN_CANDIDATES, parquet_path)
        patch_col = _resolve_column(schema_names, PATCH_COLUMN_CANDIDATES, parquet_path)

        required_columns = {target_col, patch_col}
        if dataset_col:
            required_columns.add(dataset_col)
        optional_columns = [
            col for col in ("dataset_target_id", "dataset_patch_id", "dataset_target_patch_id") if col in schema_names
        ]
        df = pd.read_parquet(parquet_path, columns=list(required_columns | set(optional_columns)))

        if dataset_col:
            df["_dataset_id"] = pd.to_numeric(df[dataset_col], errors="coerce").astype("Int64")
        else:
            try:
                derived_dataset_id = int(parquet_path.stem)
            except ValueError:
                derived_dataset_id = None
            df["_dataset_id"] = (
                pd.Series(derived_dataset_id, index=df.index, dtype="Int64")
                if derived_dataset_id is not None
                else pd.Series(pd.NA, index=df.index, dtype="Int64")
            )
        df["_target_id"] = pd.to_numeric(df[target_col], errors="coerce").astype("Int64")
        df["_patch_id"] = pd.to_numeric(df[patch_col], errors="coerce").astype("Int64")

        dataset_ids.update(df["_dataset_id"].dropna().astype(int).unique())
        target_ids.update(df["_target_id"].dropna().astype(int).unique())
        patch_ids.update(df["_patch_id"].dropna().astype(int).unique())

        valid_dt = df.dropna(subset=["_dataset_id", "_target_id"])[["_dataset_id", "_target_id"]].astype(int)
        if not valid_dt.empty:
            dataset_target_ids.update(
                f"{dataset:04d}_{target}" for dataset, target in valid_dt.to_numpy()
            )

        valid_dp = df.dropna(subset=["_dataset_id", "_patch_id"])[["_dataset_id", "_patch_id"]].astype(int)
        if not valid_dp.empty:
            dataset_patch_ids.update(
                f"{dataset:04d}_{patch}" for dataset, patch in valid_dp.to_numpy()
            )

        valid_dtp = df.dropna(subset=["_dataset_id", "_target_id", "_patch_id"])[
            ["_dataset_id", "_target_id", "_patch_id"]
        ].astype(int)
        if not valid_dtp.empty:
            dataset_target_patch_ids.update(
                f"{dataset:04d}_{target}_{patch}" for dataset, target, patch in valid_dtp.to_numpy()
            )

        if "dataset_target_id" in df.columns:
            dataset_target_ids.update(df["dataset_target_id"].dropna().astype(str).unique())
        if "dataset_patch_id" in df.columns:
            dataset_patch_ids.update(df["dataset_patch_id"].dropna().astype(str).unique())
        if "dataset_target_patch_id" in df.columns:
            dataset_target_patch_ids.update(df["dataset_target_patch_id"].dropna().astype(str).unique())

    return {
        "dataset_id": len(dataset_ids),
        "target_id": len(target_ids),
        "patch_id": len(patch_ids),
        "dataset_target_id": len(dataset_target_ids),
        "dataset_patch_id": len(dataset_patch_ids),
        "dataset_target_patch_id": len(dataset_target_patch_ids),
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    files = list(collect_parquet_files(args.intermediates_dir, args.glob))

    if not files:
        logger.warning("No parquet files found under %s matching %s", args.intermediates_dir, args.glob)
        return

    counts = count_unique_ids(files)

    print(f"Unique datasetIDs: {counts['dataset_id']}")
    print(f"Unique targetIDs (global): {counts['target_id']}")
    print(f"Unique patchIDs (global): {counts['patch_id']}")
    print(f"Unique dataset-target IDs: {counts['dataset_target_id']}")
    print(f"Unique dataset-patch IDs: {counts['dataset_patch_id']}")
    print(f"Unique dataset-target-patch IDs: {counts['dataset_target_patch_id']}")


if __name__ == "__main__":
    main()

