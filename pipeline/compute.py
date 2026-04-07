"""
compute_turb_stats.py

Unified pipeline for computing turbulence statistics from solar wind vector field data
(magnetic field by default; designed to be easily extended to velocity or other vectors).

Designed for embarrassingly parallel execution as a SLURM array job:

    python compute_turb_stats.py <file_index>

    #SBATCH --array=0-N
    python compute_turb_stats.py $SLURM_ARRAY_TASK_ID

Each job processes one CDF file independently and writes its own output files.
No inter-process communication is required or used.

Pipeline summary
----------------
1.  Load a single CDF file and rename columns to generic component names (Fx, Fy, Fz).
2.  Resample to two cadences:
      - High-resolution (hr): for structure functions, PSD, and Taylor scale estimation.
      - Low-resolution  (lr): for correlation scale estimation via ACF.
    If cadence_hr == cadence_lr (or cadence_hr is None), single-cadence mode is used.
3.  Split into fixed-length intervals.
4.  Apply the configured gap mode:
      - "none"     : interpolate if missingness < max_gap_prop, skip otherwise.
      - "retain"   : keep original gaps; produce naive / lint / corrected variants.
      - "simulate" : artificially remove data; produce true / naive / lint / corrected variants.
5.  For each interval version, compute the groups listed in config["compute"]:
      - "bulk"   : mean field magnitude (F0), rms fluctuation (dF), dF/F0, per-component means/stds
      - "acf_lr" : lr ACF → correlation scales tce (exp-trick), tcf (exp-fit), tci (integral)
      - "acf_hr" : hr ACF → Taylor microscale ttu (Chuychai method)
      - "sf"     : hr structure function + optional bias correction
      - "psd"    : hr PSD → spectral slopes (qi, qk) and spectral break frequency (fb)
    Omitting a group skips its computation entirely.  Default is all groups.
    Note: "acf_hr", "sf", and "psd" all require hr data; they are ignored in
    single-cadence mode if cadence_hr is None.
6.  Save scalar statistics as CSV and full interval data (including curve arrays) as pickle.

Custom modules required
-----------------------
    pipeline.params             : pipeline constants (max_lag_prop, tau_min, tau_max, ...)
    pipeline.sf_funcs           : compute_sf(...)
    pipeline.utils_new          : compute_corr_scale_exp_trick, compute_outer_scale_exp_fit,
                             compute_outer_scale_integral, compute_taylor_chuychai, SmoothySpec
    pipeline.ts_dashboard_utils : remove_data (only needed for gap_mode="simulate")
"""

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
import statsmodels.tsa.stattools as tsa


import pipeline.params as params
import pipeline.sf_funcs as sf_funcs
import pipeline.utils_new as un
import pipeline.ts_dashboard_utils as ts

# ---------------------------------------------------------------------------
# Spacecraft / field configuration
# ---------------------------------------------------------------------------
# To add a new spacecraft or field, extend this dict only.
# The rest of the script is agnostic to these names.

FIELD_COLS = {
    "psp": {
        "B": ["psp_fld_l2_mag_RTN_0", "psp_fld_l2_mag_RTN_1", "psp_fld_l2_mag_RTN_2"],
    },
    "wind": {
        "B": ["BGSE_0", "BGSE_1", "BGSE_2"],
        "V": ["P_VELS_0",    "P_VELS_1",    "P_VELS_2"],      # placeholder — wire up params when needed
    },
    "voyager": {
        "B": ["BR", "BT", "BN"],
    },
}

COMPONENTS = ["x", "y", "z"]   # generic component labels; columns become Fx, Fy, Fz

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


import cdflib
import pandas as pd

def load_cdf(file_path, spacecraft, field="B"):
    """Load a CDF file and return a cleaned, renamed DataFrame.

    Raw CDF column names are mapped to generic Fx/Fy/Fz names so that all
    downstream code is field- and spacecraft-agnostic.

    Parameters
    ----------
    file_path : str
    spacecraft : str
    field : str
        "B" or "V".

    Returns
    -------
    pd.DataFrame
        Columns: Fx, Fy, Fz.
    """
    raw_cols = FIELD_COLS[spacecraft][field]
    col_map  = {raw: f"F{c}" for raw, c in zip(raw_cols, COMPONENTS)}

    cdf = cdflib.CDF(file_path)

    # --- load required base variables ---
    base_vars = {col.rsplit("_", 1)[0] for col in raw_cols}

    data = {}

    for var in base_vars:
        values = cdf.varget(var)

        if values.ndim == 1:
            data[var] = values
        else:
            for i in range(values.shape[1]):
                data[f"{var}_{i}"] = values[:, i]

    # --- time index ---
    epoch = cdf.varget("Epoch")
    time_index = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch))

    df = pd.DataFrame(data, index=time_index)

    # --- reuse your original logic unchanged ---
    df = df.loc[:, raw_cols].rename(columns=col_map)

    # --- diagnostics (unchanged) ---
    start, end = df.index[0], df.index[-1]
    print(f"  Loaded {len(df):,} rows  ({(end - start).round('s')})  {start} → {end}")

    time_diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    modal_cadence = time_diffs.value_counts().idxmax()
    prop_near = time_diffs.between(modal_cadence * 0.95, modal_cadence * 1.05).mean()
    print(f"  Modal cadence: {modal_cadence:.4f} s  "
          f"({1 / modal_cadence:.2f} Hz,  {prop_near * 100:.1f}% of samples within ±5%)")

    return df


def resample_field(df, cadence, label=""):
    """Resample df to cadence and print a brief missing-data report."""
    df_res = df.resample(cadence).mean()
    missing = df_res.isna().mean() * 100
    tag = f" ({label})" if label else ""
    if missing.sum() > 0:
        lines = missing.round(2).to_string().replace("\n", "\n    ")
        print(f"  Missing % after resampling{tag}:\n    {lines}")
    else:
        print(f"  No missing data after resampling{tag}.")
    return df_res


# ---------------------------------------------------------------------------
# Interval construction
# ---------------------------------------------------------------------------


def split_into_intervals(df, int_length, spacecraft, field):
    """Split a DataFrame into a list of fixed-length interval dicts.

    Each dict carries its data alongside identifying metadata so it can be
    processed and saved independently.

    Parameters
    ----------
    df : pd.DataFrame
    int_length : str
        Pandas-compatible offset string, e.g. "12h", "14d".
    spacecraft : str
    field : str

    Returns
    -------
    list[dict]
    """
    freq = pd.tseries.frequencies.to_offset(int_length)
    starts = pd.date_range(df.index[0].ceil(int_length), df.index[-1], freq=freq)

    intervals = []
    for i, start in enumerate(starts):
        end = start + pd.Timedelta(int_length) - pd.Timedelta("1ns")
        chunk = df[start:end].copy()
        if chunk.empty:
            continue
        intervals.append({
            "interval_id":  i,
            "spacecraft":   spacecraft,
            "field":        field,
            "start":        start,
            "end":          end,
            "data":         chunk,
        })
    return intervals


# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------
# Gaps are always applied at the hr cadence and the lr data is derived from it,
# ensuring the two resolutions stay consistent.


def _interpolate(df):
    return df.interpolate(method="linear").ffill().bfill()


def _gap_proportion(df):
    return df.isna().mean().mean()


def simulate_gap(data, rng):
    """Randomly remove data in chunks plus a uniform scatter component.

    Wraps ts.remove_data for a two-pass removal that mimics realistic data gaps.
    """
    total_removal  = rng.uniform(0, 0.95)
    ratio_chunks   = rng.uniform(0.7, 1.0)

    gapped, _, prop_chunks = ts.remove_data(
        data.copy(), total_removal * ratio_chunks,
        chunks=int(rng.integers(1, 10)),
    )
    gapped, _, _ = ts.remove_data(gapped, total_removal - prop_chunks)
    return gapped


def prepare_versions(data_hr_raw, cadence_lr, config, rng, interval_id, sim_version=0):
    """Produce all (data_hr, data_lr, meta) versions for one raw interval.

    This is the single place where gap logic lives.  Downstream statistics
    code receives clean, ready-to-use data pairs with no gap-mode awareness.

    Parameters
    ----------
    data_hr_raw : pd.DataFrame
        Raw interval data at hr cadence (may contain NaNs).
    cadence_lr : str
    config : dict
    rng : np.random.Generator
    interval_id : int
    sim_version : int
        Version index for gap_mode="simulate".

    Returns
    -------
    list[dict]
        Each dict has keys: data_hr, data_lr, gap_status, tgp, sim_version.
    """
    def make_lr(data_hr):
        return data_hr.resample(cadence_lr).mean()

    gap_mode   = config["gap_mode"]
    correcting = config.get("correction_lookup") is not None

    # ---- gap_mode = "none" ------------------------------------------------
    if gap_mode == "none":
        tgp = _gap_proportion(data_hr_raw)
        if tgp > config.get("max_gap_prop", 0.1):
            print(f"    Interval {interval_id}: skipping "
                  f"({tgp:.1%} missing > {config['max_gap_prop']:.0%} threshold).")
            return []
        data_hr = _interpolate(data_hr_raw)
        return [dict(data_hr=data_hr, data_lr=make_lr(data_hr),
                     gap_status="complete", tgp=tgp, sim_version=0)]

    # ---- gap_mode = "retain" ----------------------------------------------
    if gap_mode == "retain":
        tgp = _gap_proportion(data_hr_raw)
        if tgp == 0:
            return [dict(data_hr=data_hr_raw.copy(), data_lr=make_lr(data_hr_raw),
                         gap_status="complete", tgp=0.0, sim_version=0)]

        data_hr_lint = _interpolate(data_hr_raw)
        versions = [
            dict(data_hr=data_hr_raw,     data_lr=make_lr(data_hr_raw),  gap_status="naive",  tgp=tgp, sim_version=0),
            dict(data_hr=data_hr_lint,     data_lr=make_lr(data_hr_lint), gap_status="lint",   tgp=tgp, sim_version=0),
        ]
        if correcting:
            versions.append(
                dict(data_hr=data_hr_lint.copy(), data_lr=make_lr(data_hr_lint),
                     gap_status="corrected", tgp=tgp, sim_version=0)
            )
        return versions

    # ---- gap_mode = "simulate" --------------------------------------------
    if gap_mode == "simulate":
        versions = []
        for j in range(config["times_to_gap"]):
            # True (ungapped) version — always included as the reference
            data_hr_true = data_hr_raw.copy()
            versions.append(dict(data_hr=data_hr_true, data_lr=make_lr(data_hr_true),
                                 gap_status="true", tgp=np.nan, sim_version=j))

            data_hr_gapped = simulate_gap(data_hr_raw, rng)
            tgp = _gap_proportion(data_hr_gapped)
            data_hr_lint = _interpolate(data_hr_gapped)

            versions += [
                dict(data_hr=data_hr_gapped, data_lr=make_lr(data_hr_gapped), gap_status="naive",  tgp=tgp, sim_version=j),
                dict(data_hr=data_hr_lint,   data_lr=make_lr(data_hr_lint),   gap_status="lint",   tgp=tgp, sim_version=j),
            ]
            if correcting:
                versions.append(
                    dict(data_hr=data_hr_lint.copy(), data_lr=make_lr(data_hr_lint),
                         gap_status="corrected", tgp=tgp, sim_version=j)
                )
        return versions

    raise ValueError(f"Unknown gap_mode {config['gap_mode']!r}. "
                     "Choose 'none', 'retain', or 'simulate'.")


# ---------------------------------------------------------------------------
# SF bias correction (optional)
# ---------------------------------------------------------------------------


def apply_sf_correction(sf_df, correction_lookup, n_bins=25):
    """Look up and apply scaling factors to a structure-function DataFrame.

    Parameters
    ----------
    sf_df : pd.DataFrame
        Must contain columns lag_tc, gp, sf_2.
    correction_lookup : dict
    n_bins : int

    Returns
    -------
    pd.DataFrame
        sf_df with columns scaling, scaling_lower, scaling_upper appended.
    """

    with open(f"{correction_lookup}", "rb") as f:
        print("Loading correction lookup...")
        correction_lookup = pickle.load(f)

    xedges = correction_lookup["xedges"] * 10 / 10_000
    yedges = correction_lookup["yedges"]
    zedges = correction_lookup["zedges"]

    def _clip_idx(values, edges):
        return np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)

    xi = _clip_idx(sf_df["lag_tc"].values, xedges)
    yi = _clip_idx(sf_df["gp"].values,     yedges)
    zi = _clip_idx(sf_df["sf_2"].values,   zedges)

    result = sf_df.copy()
    for key in ("scaling", "scaling_lower", "scaling_upper"):
        result[key] = correction_lookup[key][xi, yi, zi]
    return result


def smooth_scaling(x, y, num_bins=20):
    """Smooth a scaling curve via log-spaced binning + cubic interpolation."""
    x, y = np.asarray(x), np.asarray(y)
    if x.size == 0:
        return y
    if x.min() <= 0 or x.min() == x.max():
        return np.convolve(np.pad(y, 1, mode="edge"), np.ones(3) / 3, mode="valid")

    edges   = np.logspace(np.log10(x.min()), np.log10(x.max()), num_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned  = np.array([np.nanmean(y[(x >= lo) & (x < hi)])
                        for lo, hi in zip(edges[:-1], edges[1:])])

    if np.all(np.isnan(binned)):
        return np.ones_like(x)

    valid = binned[~np.isnan(binned)]
    full_x = np.concatenate(([edges[0]],  centers, [edges[-1]]))
    full_y = np.nan_to_num(np.concatenate(([valid[0]], binned, [valid[-1]])), nan=1.0)
    return interp1d(full_x, full_y, kind="cubic", fill_value="extrapolate")(x)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------


def _fit_power_law(x, y, mask):
    """Log-log linear regression over *mask*. Returns (slope, intercept, stderr)."""
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    try:
        fit = stats.linregress(np.log(x[mask]), np.log(y[mask]))
        return fit.slope, fit.intercept, fit.stderr
    except Exception:
        return np.nan, np.nan, np.nan


def compute_curves(data_lr, data_hr, cadence_lr_s, cadence_hr_s, config, naive_gp=None):
    """Compute array-valued turbulence curves for one interval version.

    Only the groups listed in config["compute"] are run.  The full set is:
        "bulk"   : mean field magnitude (F0), rms fluctuation (dF), dF/F0,
                   per-component means and standard deviations.
        "acf_lr" : per-component ACF at low resolution → lag_acf_lr_s, acf_lr,
                   acf_lr_comps.
        "acf_hr" : per-component ACF at high resolution → lag_acf_hr_s, acf_hr,
                   acf_hr_comps.
        "sf"     : second-order structure function at low resolution →
                   sf, sf_comps, lag_sf_s, lag_n, gp.
                   Optional bias correction via correction_lookup.
        "psd"    : power spectral density at high resolution →
                   psd, psd_comps, psd_smooth, freq.

    Dual-cadence design
    -------------------
    Low-resolution  : ACF (correlation scales) + SF (inertial range).
        Large-scale quantities benefit from lower noise and cheaper computation.
    High-resolution : ACF (Taylor microscale) + PSD (kinetic range).
        Fine-scale quantities require finer time resolution near lag = 0.

    Scalar statistics derived from these curves (tce, ttu, qi, qk, ...) are
    computed separately in get_derived_stats().  This separation means you can
    re-derive scalars with different parameters without re-running the curves.

    Parameters
    ----------
    data_lr : pd.DataFrame
        Low-resolution data; columns Fx, Fy, Fz.
    data_hr : pd.DataFrame
        High-resolution data; columns Fx, Fy, Fz.
    cadence_lr_s : float
        Low-resolution cadence in seconds.
    cadence_hr_s : float
        High-resolution cadence in seconds.
    config : dict
    naive_gp : np.ndarray or None
        Gap-proportion array from the naive version, required for SF bias correction.

    Returns
    -------
    dict
        Array-valued curves and bulk scalars for whichever groups were requested.
    """
    ALL_GROUPS = {"bulk", "acf_lr", "acf_hr", "sf", "psd"}
    compute    = config.get("compute", ALL_GROUPS)

    cols       = [f"F{c}" for c in COMPONENTS]
    correcting = config.get("correction_lookup") is not None and naive_gp is not None
    out        = {}

    n_hr = len(data_hr)
    n_lr = len(data_lr)

    # ---- bulk : field means, magnitudes, fluctuations ----------------------
    if "bulk" in compute:
        field_data = data_hr[cols]
        means = field_data.mean()
        stds  = field_data.std()
        F0    = np.linalg.norm(means.values)
        dF    = np.sqrt(((field_data - means) ** 2).sum(axis=1, skipna=False).mean())
        out.update({
            "F0":    F0,
            "dF":    dF,
            "dFoF0": dF / F0 if F0 > 0 else np.nan,
            **{f"F{c}_mean": float(means[f"F{c}"]) for c in COMPONENTS},
            **{f"F{c}_std":  float(stds[f"F{c}"])  for c in COMPONENTS},
        })

    # ---- acf_lr : per-component lr ACF -----------------------------------------
    # tsa.acf returns lags 0..nlags inclusive (length nlags+1).
    # lag_acf_lr_s includes lag 0 so stored arrays stay self-consistent.
    if "acf_lr" in compute:
        nlags_lr     = int(params.max_lag_prop * n_lr)
        lag_acf_lr_s = np.arange(0, nlags_lr + 1) * cadence_lr_s   # lag 0 → nlags
        acf_lr_comps = [
            tsa.acf(data_lr[c], nlags=nlags_lr, missing="conservative", adjusted=True)
            for c in cols
        ]
        acf_lr = np.mean(acf_lr_comps, axis=0)   # shape: (nlags_lr+1,), lag 0 first
        out.update(lag_acf_lr_s=lag_acf_lr_s, acf_lr=acf_lr, acf_lr_comps=acf_lr_comps)

    # ---- acf_hr : per-component hr ACF -----------------------------------------
    # lag_acf_hr_s includes lag 0 so stored arrays stay self-consistent.
    if "acf_hr" in compute:
        nlags_hr_acf = int(params.max_lag_prop * n_hr)
        lag_acf_hr_s = np.arange(0, nlags_hr_acf + 1) * cadence_hr_s   # lag 0 → nlags
        acf_hr_comps = [
            tsa.acf(data_hr[c], nlags=nlags_hr_acf, missing="conservative", adjusted=True)
            for c in cols
        ]
        acf_hr = np.mean(acf_hr_comps, axis=0)   # shape: (nlags_hr_acf+1,), lag 0 first
        out.update(lag_acf_hr_s=lag_acf_hr_s, acf_hr=acf_hr, acf_hr_comps=acf_hr_comps)

    # ---- sf : structure function (lr) ------------------------------------------
    # Computed at low resolution, consistent with the correlation-scale ACF.
    # lags_sf are sample counts (required by sf_funcs.compute_sf);
    # lag_sf_s is the corresponding time axis in seconds.
    # TODO: once V0 is available, add lag_sf_km = lag_sf_s * V0 (Taylor's hypothesis).
    if "sf" in compute:
        lags_sf  = (np.unique(np.logspace(0, np.log10(params.max_lag_prop * n_lr), 100).astype(int))
                    if config.get("log_lags") else np.arange(1, int(params.max_lag_prop * n_lr)))
        lags_sf  = lags_sf[lags_sf > 0]
        lag_sf_s = lags_sf * cadence_lr_s

        all_sfs    = []
        sf_df_last = None

        for col in cols:
            series           = data_lr[col]
            col_mean, col_sd = series.mean(), series.std()
            series_std       = (series - col_mean) / col_sd

            sf_df = sf_funcs.compute_sf(
                pd.DataFrame(series_std), lags_sf, [2], False, False, None
            )

            if correcting:
                sf_df["lag_tc"] = sf_df["lag"] * 10 / len(series_std)
                sf_df["gp"]     = naive_gp
                sf_df           = apply_sf_correction(sf_df, config["correction_lookup"])
                scaling         = smooth_scaling(sf_df["lag"], sf_df["scaling"])
                sf_df["sf_2"]  *= scaling

            sf_df["sf_2"] *= col_sd ** 2
            all_sfs.append(sf_df["sf_2"].values)
            sf_df_last = sf_df

        out.update({
            "sf":       np.mean(all_sfs, axis=0),
            "sf_comps": all_sfs,
            "lag_sf_s": lag_sf_s,
            "lag_n":    sf_df_last["n"].values,
            "gp":       (sf_df_last["gp"].values
                         if "gp" in sf_df_last.columns
                         else np.zeros(len(lags_sf))),
        })

    # ---- psd : power spectral density (hr) -------------------------------------
    if "psd" in compute:
        all_psds = []
        for col in cols:
            freqs, psd = signal.periodogram(data_hr[col], fs=1 / cadence_hr_s)
            all_psds.append(psd[1:])

        psd_mean   = np.mean(all_psds, axis=0)
        psd_smooth = un.SmoothySpec(psd_mean)
        out.update({
            "psd":        psd_mean,
            "psd_comps":  all_psds,
            "psd_smooth": psd_smooth,
            "freq":       freqs[1:],
        })

    return out


def get_derived_stats(curves, config):
    """Derive scalar turbulence statistics from pre-computed curves.

    Intentionally separated from compute_curves() so that scale estimates and
    spectral fits can be re-derived with different parameters (e.g. different
    tau_min/tau_max or fit ranges) without repeating expensive curve computation.
    Just reload the pickle, call this function, and update the scalar columns.

    Parameters
    ----------
    curves : dict
        Output of compute_curves() for one interval version.
    config : dict
        Pipeline configuration.  Spectral fit ranges are read from
        config["f_fit_range_inertial"] and config["f_fit_range_kinetic"].

    Returns
    -------
    dict
        Scalar statistics only (no arrays).  Keys that could not be computed
        are set to np.nan so the output schema is always complete.
    """
    out = {}

    # ---- Correlation scales (from lr ACF) --------------------------------------
    if "acf_lr" in curves:
        acf_lr       = curves["acf_lr"]
        lag_acf_lr_s = curves["lag_acf_lr_s"]
        # Drop lag 0 for all scale estimators
        acf_nz  = acf_lr[1:]
        lag_nz  = lag_acf_lr_s[1:]

        tce = tcf = tci = np.nan
        try:
            tce = un.compute_corr_scale_exp_trick(lag_nz, acf_nz)
        except Exception as e:
            print(f"    tce failed: {e}", file=sys.stderr)
        try:
            tcf = un.compute_outer_scale_exp_fit(lag_nz, acf_nz, np.round(2 * tce))
        except Exception as e:
            print(f"    tcf failed: {e}", file=sys.stderr)
        try:
            tci = un.compute_outer_scale_integral(lag_nz, acf_nz)
        except Exception as e:
            print(f"    tci failed: {e}", file=sys.stderr)

        out.update(tce=tce, tcf=tcf, tci=tci)

    # ---- Taylor microscale (from hr ACF) ---------------------------------------
    if "acf_hr" in curves:
        acf_hr       = curves["acf_hr"]
        lag_acf_hr_s = curves["lag_acf_hr_s"]
        acf_nz  = acf_hr[1:]
        lag_nz  = lag_acf_hr_s[1:]

        ttu = ttu_std = np.nan
        try:
            ttu, ttu_std = un.compute_taylor_chuychai(
                lag_nz, acf_nz,
                tau_min=params.tau_min,
                tau_max=params.tau_max,
            )
        except Exception as e:
            print(f"    ttu failed: {e}", file=sys.stderr)

        out.update(ttu=ttu, ttu_std=ttu_std)

    # ---- Spectral slopes and break frequency (from PSD) ------------------------
    if "psd" in curves:
        freq       = curves["freq"]
        psd_smooth = curves["psd_smooth"]

        fi_min, fi_max = config.get("f_fit_range_inertial", (None, None))
        fk_min, fk_max = config.get("f_fit_range_kinetic",  (None, None))

        qi = qi_int = qi_err = np.nan
        qk = qk_int = qk_err = np.nan
        fb = np.nan

        if fi_min and fi_max:
            qi, qi_int, qi_err = _fit_power_law(
                freq, psd_smooth, (freq > fi_min) & (freq < fi_max)
            )
        if fk_min and fk_max:
            qk, qk_int, qk_err = _fit_power_law(
                freq, psd_smooth, (freq > fk_min) & (freq < fk_max)
            )
        if not any(np.isnan([qi, qk, qi_int, qk_int])) and qi != qk:
            try:
                fb = np.exp((qk_int - qi_int) / (qi - qk))
            except Exception as e:
                print(f"    fb failed: {e}", file=sys.stderr)

        out.update(
            qi=qi, qi_intercept=qi_int, qi_stderr=qi_err,
            qk=qk, qk_intercept=qk_int, qk_stderr=qk_err,
            fb=fb,   # Hz; convert to seconds via tb = 1/(2*pi*fb) to compare with tce/ttu
        )

    return out


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def scalars_from_interval(interval):
    """Return a copy of interval with array/DataFrame entries removed."""
    return {
        k: v for k, v in interval.items()
        if not isinstance(v, (np.ndarray, pd.DataFrame, pd.Series, list))
    }


def intervals_to_scalar_df(intervals):
    """Flatten a list of interval dicts to a scalar-only DataFrame."""
    return pd.DataFrame([scalars_from_interval(iv) for iv in intervals])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(file_path, config):
    """Process one CDF file end-to-end.

    Parameters
    ----------
    file_path : str or Path
    config : dict

    Returns
    -------
    tuple[list[dict], pd.DataFrame]
        Full interval list (with curve arrays) and scalar-statistics DataFrame.
    """
    spacecraft  = config["spacecraft"]
    field       = config.get("field", "B")
    cadence_lr  = config["cadence_lr"]
    cadence_hr  = config.get("cadence_hr") or cadence_lr
    cadence_lr_s = pd.Timedelta(cadence_lr).total_seconds()
    cadence_hr_s = pd.Timedelta(cadence_hr).total_seconds()
    dual_cadence = cadence_hr != cadence_lr

    rng = np.random.default_rng(config.get("random_seed", 1))

    print(f"\n{'=' * 64}")
    print(f" Processing data in {'dual' if dual_cadence else 'single'}-cadence mode  "
          f"[hr={cadence_hr}, lr={cadence_lr}]")
    print(f"{'=' * 64}")

    # --- Load ---
    df_raw = load_cdf(file_path, spacecraft, field)

    # --- Resample to hr cadence (lr is derived per-interval inside prepare_versions) ---
    print(f"\nResampling to hr cadence ({cadence_hr})...")
    df_hr = resample_field(df_raw, cadence_hr, label="hr")
    del df_raw

    # Fill any residual NaNs only if not retaining gaps
    if config["gap_mode"] == "none":
        missing_overall = df_hr.isna().mean().mean()
        if missing_overall > 0 and missing_overall <= config.get("max_gap_prop", 0.1):
            df_hr = _interpolate(df_hr)

    # --- Split into intervals ---
    intervals_raw = split_into_intervals(df_hr, config["int_length"], spacecraft, field)

    if config.get("limit_ints") is not None:
        intervals_raw = intervals_raw[: config["limit_ints"]]

    n_iv = len(intervals_raw)
    if n_iv > 0:
        iv_first, iv_last = intervals_raw[0], intervals_raw[-1]
        print(f"\n{n_iv} raw interval(s) found  "
              f"[each of length {config['int_length']}]")
    else:
        print("\nNo intervals found — check cadence and int_length settings.")

    # --- Process each interval ---
    results = []

    for iv in intervals_raw:
        interval_id = iv["interval_id"]
        print(f"\n  ── Interval {interval_id}  "
              f"({iv['start'].strftime('%Y-%m-%d %H:%M:%S')} → "
              f"{iv['end'].strftime('%Y-%m-%d %H:%M:%S')}) ──")

        # Produce all gap versions for this interval
        versions = prepare_versions(
            data_hr_raw=iv["data"],
            cadence_lr=cadence_lr,
            config=config,
            rng=rng,
            interval_id=interval_id,
        )

        if not versions:
            continue    # skipped due to excessive missingness

        # Process versions in order; naive must come before corrected
        naive_gp = None

        for v in versions:
            gap_status  = v["gap_status"]
            data_hr     = v["data_hr"]
            data_lr     = v["data_lr"]
            tgp         = v["tgp"]
            sim_version = v["sim_version"]

            print(f"    gap_status={gap_status}  "
                  f"tgp={tgp * 100:.1f}%" if not np.isnan(tgp or np.nan) else
                  f"    gap_status={gap_status}")

            gp_for_correction = naive_gp if gap_status == "corrected" else None

            curves = compute_curves(
                data_lr, data_hr,
                cadence_lr_s, cadence_hr_s,
                config,
                naive_gp=gp_for_correction,
            )
            derived = get_derived_stats(curves, config)

            # Cache gap-proportion from naive version for the corrected version
            if gap_status == "naive":
                naive_gp = curves.get("gp")
            if gap_status == "corrected":
                naive_gp = None  # reset after corrected version consumed it

            # Build final interval record (no raw data stored to keep pickles lean)
            result = {
                "interval_id":   interval_id,
                "sim_version":   sim_version,
                "gap_status":    gap_status,
                "tgp":           tgp,
                "spacecraft":    spacecraft,
                "field":         field,
                "start":         iv["start"],
                "end":           iv["end"],
                "cadence_lr":    cadence_lr,
                "cadence_hr":    cadence_hr,
                # Store fit ranges so plots and downstream analysis are self-contained
                "f_fit_range_inertial": config.get("f_fit_range_inertial"),
                "f_fit_range_kinetic":  config.get("f_fit_range_kinetic"),
            }
            result.update(curves)
            result.update(derived)

            # Optionally keep the data for plotting / inspection
            if config.get("store_data", False):
                result["data_hr"] = data_hr
                result["data_lr"] = data_lr

            results.append(result)

    print(f"\n{'=' * 64}")
    print(f"  Done.  {len(results)} interval versions processed.")
    print(f"{'=' * 64}\n")

    return results, intervals_to_scalar_df(results)

