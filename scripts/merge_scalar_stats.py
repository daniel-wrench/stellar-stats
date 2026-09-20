"""Merge scalar-statistics CSV files produced by the compute pipeline."""

import argparse
import importlib
from pathlib import Path

import pandas as pd


def merge_scalar_stats(input_dir, output_path):
    """Merge all scalar-statistics CSVs below *input_dir* into one CSV."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    csv_paths = sorted(
        path
        for path in input_dir.rglob("*_scalar_stats.csv")
        if path.resolve() != output_path.resolve()
    )

    if not csv_paths:
        raise FileNotFoundError(
            f"No *_scalar_stats.csv files found under {input_dir}"
        )

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame.insert(0, "source_file", csv_path.relative_to(input_dir).as_posix())
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged, csv_paths


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge pipeline scalar-statistics CSV files."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs.config_wind_mag",
        help="Config module used to locate data/processed (default: %(default)s)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory to search instead of the config-derived processed directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: <input-dir>/merged_scalar_stats.csv)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = importlib.import_module(args.config).CONFIG
    input_dir = args.input_dir or Path(
        "data", "processed", config["spacecraft"], config["instrument"]
    )
    output_path = args.output or input_dir / "merged_scalar_stats.csv"

    merged, csv_paths = merge_scalar_stats(input_dir, output_path)
    print(f"Merged {len(csv_paths)} files ({len(merged)} rows) into {output_path}")


if __name__ == "__main__":
    main()
