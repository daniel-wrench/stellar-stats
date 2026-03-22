"""
plot_turb_stats.py

Produces a diagnostic figure for each interval (and each gap variant) saved
by compute_turb_stats.py.  One PNG per interval version, saved under flipbooks/.

Layout
------
Top row    : time series of vector components (requires store_data=True in pipeline)
Bottom row : [SF] [PSD] [ACF]

Usage
-----
Edit the USER SETTINGS block below and run directly:

    python plot_turb_stats.py

The script loops over every gap variant present for the chosen interval_id,
so it works regardless of gap_mode (none / retain / simulate).
"""

import glob
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

COMPONENT_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a"]
STATS_PALETTE = {
    "psd": "#ff7f00",
    "sf":  "#984ea3",
    "acf": "#f781bf",
    "qk":  "#bd8044",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_timescale(seconds):
    """Return a human-readable string for a time value in seconds."""
    if np.isnan(seconds) or seconds <= 0:
        return "N/A"
    if seconds >= 86_400:
        return f"{seconds / 86_400:.2f} d"
    if seconds >= 3_600:
        return f"{seconds / 3_600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.2f} s"


def _safe_get(d, key, default=np.nan):
    """dict.get with np.nan default, treating None as missing."""
    val = d.get(key, default)
    return default if val is None else val


def _fit_range(iv, key):
    """Safely unpack a [min, max] fit range from iv.

    iv.get() returns None if the key is absent, which cannot be unpacked —
    hence this helper rather than a bare iv.get().
    """
    val = iv.get(key)
    if val is None:
        return None, None
    return val[0], val[1]


def _acf_lr_for_plot(iv):
    """Return (lag_s, acf) for the lr ACF, with lag 0 dropped."""
    lag = iv.get("lag_acf_lr_s")
    acf = iv.get("acf_lr")
    if lag is None or acf is None:
        return None, None
    return lag[1:], acf[1:]


def _acf_from_sf(iv):
    """Derive ACF from SF via  rho(tau) = 1 - SF(tau) / (2 * mean_variance)."""
    sf = iv.get("sf")
    if sf is None:
        return None
    comp_vars = [_safe_get(iv, f"F{c}_std") ** 2 for c in ["x", "y", "z"]]
    finite = [v for v in comp_vars if np.isfinite(v)]
    if not finite:
        return None
    mean_var = np.mean(finite)
    if mean_var == 0:
        return None
    return 1.0 - sf / (2.0 * mean_var)


def _psd_from_sf(iv):
    """Derive pseudo-PSD from SF:  PSD ~ SF * lag / 6."""
    sf  = iv.get("sf")
    lag = iv.get("lag_sf_s")
    if sf is None or lag is None:
        return None, None
    valid = lag > 0
    return 1.0 / lag[valid], (sf[valid] * lag[valid]) / 6.0


# ---------------------------------------------------------------------------
# Scale annotations
# ---------------------------------------------------------------------------


def _annotate_scales(ax_sf, ax_psd, ax_acf, iv):
    """Draw tce and ttu vlines across SF, PSD, and ACF panels.

    tce and ttu are stored in seconds; PSD panel uses 1/t (Hz).
    """
    scales = {
        "tce": (_safe_get(iv, "tce"), "--", r"$\lambda_C$"),
        "ttu": (_safe_get(iv, "ttu"), ":",  r"$\lambda_T$"),
    }
    for _, (val_s, ls, label) in scales.items():
        if np.isnan(val_s) or val_s <= 0:
            continue
        label_full = f"{label} = {format_timescale(val_s)}"
        if ax_sf is not None:
            ax_sf.axvline(val_s, ls=ls, c="k", alpha=0.25, lw=1.2)
        if ax_psd is not None:
            ax_psd.axvline(1.0 / val_s, ls=ls, c="k", alpha=0.25, lw=1.2)
        if ax_acf is not None:
            ax_acf.axvline(val_s, ls=ls, c=STATS_PALETTE["acf"],
                           alpha=0.8, lw=1.2, label=label_full)


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------



def plot_interval(iv, output_path):
    """Produce and save one diagnostic figure for interval version *iv*.

    Parameters
    ----------
    iv : dict
        One element from the full_stats list produced by compute_turb_stats.py.
    output_path : Path
    """
    fig = plt.figure(figsize=(13, 6))
    ax_top = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax_sf  = plt.subplot2grid((2, 3), (1, 0))
    ax_psd = plt.subplot2grid((2, 3), (1, 1))
    ax_acf = plt.subplot2grid((2, 3), (1, 2))

    spacecraft = iv.get("spacecraft", "?")
    field      = iv.get("field", "F")
    gap_status = iv.get("gap_status", "complete")
    tgp        = iv.get("tgp", np.nan)
    cadence_lr = iv.get("cadence_lr", "?")
    cadence_hr = iv.get("cadence_hr", cadence_lr)
    start      = iv.get("start")
    end        = iv.get("end")
    slope_offset = 3  # adjust to prevent slope lines from sitting below the data
    # ── Time series ──────────────────────────────────────────────────────────
    data = iv.get("data_lr")

    if data is not None:
        cols = [c for c in data.columns if c.startswith("F")][:3]
        for i, col in enumerate(cols):
            ax_top.plot(data.index, data[col],
                        color=COMPONENT_PALETTE[i], lw=0.7,
                        label=col.replace("F", field))   # e.g. Fx -> Bx
        ax_top.legend(loc="upper right", fontsize=8, edgecolor="black")
        ax_top.set_ylabel(f"{field} (nT)" if field == "B" else field)
    else:
        ax_top.text(0.5, 0.5,
                    "Time series not stored\n(set store_data=True in pipeline)",
                    ha="center", va="center", transform=ax_top.transAxes,
                    fontsize=10, color="grey")

    tgp_str = f"{tgp * 100:.1f}%" if np.isfinite(tgp) else "N/A"
    ax_top.set_title(
        f"{spacecraft.upper()} · interval {iv.get('interval_id', '?')} · "
        f"gap_status={gap_status}  (tgp={tgp_str})",
        fontsize=10,
    )

    # Metadata annotation box
    duration = "?"
    try:
        delta_s  = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
        duration = format_timescale(delta_s)
    except Exception:
        pass

    n_pts = len(data) if data is not None else "?"

    F0 = _safe_get(iv, "F0")
    dF = _safe_get(iv, "dF")
    bulk_str = (f"$\\bf{{F0}}$={F0:.2f} nT,  $\\bf{{dF}}$={dF:.2f} nT"
                if np.isfinite(F0) and np.isfinite(dF) else "")

    meta = "\n".join(filter(None, [
        f"$\\bf{{Spacecraft}}$: {spacecraft.upper()}",
        f"$\\bf{{Duration}}$: {duration}  (N={n_pts})",
        f"$\\bf{{Cadence}}$: lr={cadence_lr}, hr={cadence_hr}",
        f"$\\bf{{Start}}$: {pd.Timestamp(start).strftime('%Y-%m-%d %H:%M:%S') if start else '?'}",
        f"$\\bf{{End}}$:   {pd.Timestamp(end).strftime('%Y-%m-%d %H:%M:%S') if end else '?'}",
        bulk_str,
    ]))
    ax_top.annotate(meta, xy=(0.01, 0.03), xycoords="axes fraction",
                    fontsize=8, va="bottom", ha="left",
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="grey", boxstyle="round"))

    # ── Structure function ───────────────────────────────────────────────────
    lag_sf = iv.get("lag_sf_s")
    sf     = iv.get("sf")

    if lag_sf is not None and sf is not None:
        ax_sf.loglog(lag_sf, sf, color=STATS_PALETTE["sf"], lw=1.2, label="SF")

        qi     = _safe_get(iv, "qi")
        fi_min, fi_max = _fit_range(iv, "f_fit_range_inertial")

        if np.isfinite(qi) and fi_min and fi_max:
            t_fit = np.logspace(np.log10(1 / fi_max), np.log10(1 / fi_min), 100)

            # Shade the inertial fit range on the lag axis
            ax_sf.axvspan(t_fit[0], t_fit[-1], alpha=0.08, color=STATS_PALETTE["sf"])

            # Anchor the slope line to the data at the midpoint of the fit range.
            # SF ~ tau^(-(qi+1)) for a power-law PSD with slope qi
            # e.g. qi = -5/3  =>  SF exponent = 2/3
            sf_exp = -(qi + 1)
            t_mid  = np.sqrt(t_fit[0] * t_fit[-1])
            sf_ref = np.exp(np.interp(np.log(t_mid), np.log(lag_sf), np.log(sf)))
            y_fit  = sf_ref * (t_fit / t_mid) ** sf_exp * 1.5 # needs different offset than PSD

            ax_sf.loglog(t_fit, y_fit, "--", lw=1.8, color=STATS_PALETTE["sf"],
                         label=rf"$\alpha$={sf_exp:.2f}")

    else:
        ax_sf.text(0.5, 0.5, "SF not computed",
                   ha="center", va="center", transform=ax_sf.transAxes, color="grey")

    ax_sf.set_xlabel("Lag (s)")
    ax_sf.set_ylabel(f"SF ({field}$^2$)")
    ax_sf.set_title("Structure Function", color=STATS_PALETTE["sf"], fontsize=9)

    # ── Power spectral density ───────────────────────────────────────────────
    freq  = iv.get("freq")
    psd   = iv.get("psd")
    psd_s = iv.get("psd_smooth")

    if freq is not None and psd is not None:
        ax_psd.loglog(freq, psd, alpha=0.3, color=STATS_PALETTE["psd"], lw=0.8)
        if psd_s is not None:
            ax_psd.loglog(freq, psd_s, color=STATS_PALETTE["psd"], lw=1.4,
                          label="PSD")

        qi     = _safe_get(iv, "qi")
        qi_int = _safe_get(iv, "qi_intercept")
        fi_min, fi_max = _fit_range(iv, "f_fit_range_inertial")

        if np.isfinite(qi) and np.isfinite(qi_int) and fi_min and fi_max:
            f_fit   = np.logspace(np.log10(fi_min), np.log10(fi_max), 100)
            ref_psd = psd_s if psd_s is not None else psd
            f_mid   = np.sqrt(fi_min * fi_max)
            y_mid   = np.exp(qi * np.log(f_mid) + qi_int)
            psd_mid = np.interp(f_mid, freq, ref_psd)
            scale   = psd_mid / y_mid if y_mid > 0 else 1.0

            ax_psd.axvspan(fi_min, fi_max, alpha=0.08, color=STATS_PALETTE["psd"])
            ax_psd.loglog(f_fit, np.exp(qi * np.log(f_fit) + qi_int) * scale * slope_offset,
                          "--", lw=1.8, color=STATS_PALETTE["psd"],
                          label=rf"$\beta_i$={qi:.2f}")

        qk     = _safe_get(iv, "qk")
        qk_int = _safe_get(iv, "qk_intercept")
        fk_min, fk_max = _fit_range(iv, "f_fit_range_kinetic")

        if np.isfinite(qk) and np.isfinite(qk_int) and fk_min and fk_max:
            f_fit   = np.logspace(np.log10(fk_min), np.log10(fk_max), 100)
            ref_psd = psd_s if psd_s is not None else psd
            f_mid   = np.sqrt(fk_min * fk_max)
            y_mid   = np.exp(qk * np.log(f_mid) + qk_int)
            psd_mid = np.interp(f_mid, freq, ref_psd)
            scale   = psd_mid / y_mid if y_mid > 0 else 1.0

            ax_psd.axvspan(fk_min, fk_max, alpha=0.08, color=STATS_PALETTE["qk"])
            ax_psd.loglog(f_fit, np.exp(qk * np.log(f_fit) + qk_int) * scale * slope_offset,
                          "--", lw=1.8, color=STATS_PALETTE["qk"],
                          label=rf"$\beta_k$={qk:.2f}")

        fb = _safe_get(iv, "fb")
        if np.isfinite(fb) and fb > 0:
            ax_psd.axvline(fb, color=STATS_PALETTE["qk"], alpha=0.7, lw=1.0,
                           label=rf"$f_b$={fb:.3g} Hz")

        f_sf, psd_sf = _psd_from_sf(iv)
        if f_sf is not None:
            ax_psd.loglog(f_sf, psd_sf, color=STATS_PALETTE["sf"],
                          alpha=0.6, lw=1.2, label="From SF")

    else:
        ax_psd.text(0.5, 0.5, "PSD not computed",
                    ha="center", va="center", transform=ax_psd.transAxes, color="grey")

    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel(f"PSD ({field}$^2$/Hz)")
    ax_psd.set_title("Power Spectral Density", color=STATS_PALETTE["psd"], fontsize=9)

    # ── Autocorrelation function ─────────────────────────────────────────────
    lag_lr, acf_lr = _acf_lr_for_plot(iv)

    if lag_lr is not None:
        ax_acf.plot(lag_lr, acf_lr, color=STATS_PALETTE["acf"], lw=1.2,
                    label="ACF (lr)")
        ax_acf.axhline(0, color="k", lw=0.5, ls="-", alpha=0.3)

        # ACF derived from SF.  Note: use iv.get("lag_sf_s") directly here —
        # do not rely on lag_sf from the SF block, which may be None if SF
        # was not computed.
        acf_sf  = _acf_from_sf(iv)
        lag_sf2 = iv.get("lag_sf_s")
        if acf_sf is not None and lag_sf2 is not None:
            ax_acf.plot(lag_sf2, acf_sf, color=STATS_PALETTE["sf"],
                        alpha=0.6, lw=1.2, label="From SF")

    else:
        ax_acf.text(0.5, 0.5, "ACF not computed",
                    ha="center", va="center", transform=ax_acf.transAxes, color="grey")

    ax_acf.set_xlabel("Lag (s)")
    ax_acf.set_ylabel("ACF")
    ax_acf.set_title("Autocorrelation (lr)", color=STATS_PALETTE["acf"], fontsize=9)

    # ── Scale vlines across all panels ───────────────────────────────────────
    _annotate_scales(
        ax_sf  if lag_sf is not None else None,
        ax_psd if freq   is not None else None,
        ax_acf if lag_lr is not None else None,
        iv,
    )

    # Draw legends after _annotate_scales so scale labels are included
    if lag_sf is not None:
        ax_sf.legend(fontsize=8)
    if freq is not None:
        ax_psd.legend(fontsize=8)
    if lag_lr is not None:
        ax_acf.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

DATA_PATH_PREFIX = ""
SPACECRAFT       = "wind"
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
        input_filepath.parent.parent.parent
        / "output/figs" / SPACECRAFT
        / (input_filepath.stem + f"_int{INTERVAL_ID}_v{sim}_{gap}.png")
    )
    plot_interval(iv, out)

print("\nDone.")
