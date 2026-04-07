
import glob
import pickle
from pathlib import Path

from pipeline.plot import plot_interval

DATA_PATH_PREFIX = ""
SPACECRAFT       = "voyager"
FILE_INDEX       = 0
INTERVAL_ID      = 1       # all gap variants for this interval_id will be plotted

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

files = sorted(glob.iglob(
    f"{DATA_PATH_PREFIX}data/processed/{SPACECRAFT}/**/*_full_stats.pkl",
    recursive=True,
))

if not files:
    raise FileNotFoundError(
        f"No *_full_stats.pkl files found under "
        f"{DATA_PATH_PREFIX}data/processed/{SPACECRAFT}/"
    )

input_filepath = Path(files[FILE_INDEX])
print(f"Loading {input_filepath} ...")

with open(input_filepath, "rb") as f:
    all_stats = pickle.load(f)

print(f"  {len(all_stats)} interval version(s) found.")

targets = [iv for iv in all_stats if iv.get("interval_id") == INTERVAL_ID]

if not targets:
    available = sorted({iv.get("interval_id") for iv in all_stats})
    raise ValueError(
        f"No intervals with interval_id={INTERVAL_ID}. Available: {available}"
    )

print(f"\nPlotting {len(targets)} version(s) for interval_id={INTERVAL_ID}:")

for iv in targets:
    gap = iv.get("gap_status",  "complete")
    sim = iv.get("sim_version", 0)
    print(f"  gap_status={gap}, sim_version={sim}")

    out = (
        Path("output/figs") / SPACECRAFT
        / (input_filepath.stem + f"_int{INTERVAL_ID}_v{sim}_{gap}.png")
    )
    plot_interval(iv, out)

print("\nDone.")
