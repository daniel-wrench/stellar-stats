import sys
import importlib
import glob
import pickle
from pathlib import Path

from pipeline.plot import plot_interval

if __name__ == "__main__":

    # --- config selection ---
    config_name = sys.argv[1] if len(sys.argv) > 1 else "configs.config_wind_mag"
    config = importlib.import_module(config_name).CONFIG

    # --- optional file index ---
    file_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # --- optional interval index ---
    interval_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    spacecraft = config["spacecraft"]
    instrument = config["instrument"]

    files = sorted(
        glob.iglob(f"data/processed/{spacecraft}/{instrument}/**/*_full_stats.pkl", recursive=True,)
        )

    if not files:
        raise FileNotFoundError(
            f"No *_full_stats.pkl files found under "
            f"data/processed/{spacecraft}/{instrument}/"
        )

    input_filepath = Path(files[file_index])
    print(f"Loading {input_filepath} ...")

    with open(input_filepath, "rb") as f:
        all_stats = pickle.load(f)

    print(f"  {len(all_stats)} interval version(s) found.")

    targets = [iv for iv in all_stats if iv.get("interval_id") == interval_id]

    if not targets:
        available = sorted({iv.get("interval_id") for iv in all_stats})
        raise ValueError(
            f"No intervals with interval_id={interval_id}. Available: {available}"
        )

    print(f"\nPlotting {len(targets)} version(s) for interval_id={interval_id}:")

    for iv in targets:
        gap = iv.get("gap_status",  "complete")
        sim = iv.get("sim_version", 0)
        print(f"  gap_status={gap}, sim_version={sim}")

        out = (
            Path("output/figs") / spacecraft
            / (input_filepath.stem + f"_int{interval_id}_v{sim}_{gap}.png")
        )
        plot_interval(iv, out)

    print("\nDone.")
