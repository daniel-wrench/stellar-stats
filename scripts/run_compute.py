import sys
import glob
import pickle
from pathlib import Path
import importlib

from pipeline.compute import run_pipeline

if __name__ == "__main__":

    # --- config selection ---
    config_name = sys.argv[1] if len(sys.argv) > 1 else "configs.config_wind_mag"
    config = importlib.import_module(config_name).CONFIG

    # --- optional file index ---
    file_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    spacecraft = config["spacecraft"]
    instrument = config["instrument"]

    raw_file_list = sorted(
        glob.iglob(f"data/raw/{spacecraft}/{instrument}/**/*.cdf", recursive=True)
    )

    if not raw_file_list:
        raise FileNotFoundError(f"No CDF files found for {spacecraft}/{instrument}")

    file_path = raw_file_list[file_index]

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    print(f"\n FILE {file_index}: {file_path}")
    full_results, scalar_df = run_pipeline(file_path, config)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------

    out_stem = Path(file_path.replace("raw", "processed").replace(".cdf", ""))
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    gap_tag = f"_{config['gap_mode']}" if config["gap_mode"] != "none" else ""

    scalar_path = out_stem.with_name(out_stem.name + f"{gap_tag}_scalar_stats.csv")
    scalar_df.to_csv(scalar_path, index=False)
    print(f"Scalar stats → {scalar_path}")

    full_path = out_stem.with_name(out_stem.name + f"{gap_tag}_full_stats.pkl")
    with open(full_path, "wb") as f:
        pickle.dump(full_results, f)
    print(f"Full stats   → {full_path}")

    print("\nPipeline complete.")
    print("----" * 16)
