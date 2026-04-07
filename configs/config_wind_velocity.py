CONFIG = {

    # --- Spacecraft and field -------------------------------------------
    "spacecraft": "wind",
    "instrument": "3dp/3dp_pm",   # mfi for wind, mag for voyager and psp, 3dp/3dp_pm for wind proton moments
    "field":      "V",      # "B" = magnetic field; "V" = velocity (when ready)

    # --- Cadences --------------------------------------------------------
    # Dual-cadence separates correlation scale estimation (needs long time
    # baseline → low-res) from Taylor scale + SF/PSD (need fine resolution
    # → high-res).  Set cadence_hr = None for single-cadence mode.
    "cadence_lr": "3s",         # low-res: for ACF correlation scales
    "cadence_hr": None,     # high-res: for Taylor scale, SF, PSD
                                # set to None (no quote marks) to disable dual-cadence

    # --- Interval settings -----------------------------------------------
    "int_length": "12h",        # Duration of each analysis interval
    "limit_ints": 2,         # Cap number of intervals per file (None = all)

    # --- Gap handling ----------------------------------------------------
    # "none"     : interpolate gaps if missing < max_gap_prop, skip otherwise
    # "retain"   : keep original gaps; produce naive / lint / corrected variants
    # "simulate" : artificially remove data; produce true / naive / lint / corrected
    "gap_mode":       "none",
    "max_gap_prop":   0.9,      # used only for gap_mode="none"
    "times_to_gap":   3,        # used only for gap_mode="simulate"

    # --- SF bias correction ----------------------------------------------
    # Set to the loaded correction_lookup dict to enable; None to disable.
    "correction_lookup": "correction_lookup_3d_25_bins_lint.pkl",

    # --- Spectral fitting ------------------------------------------------
    "f_fit_range_inertial": [0.005, 0.2],    # Hz
    "f_fit_range_kinetic":  [0.5,   1.4],    # Hz

    # --- Stat groups to compute ------------------------------------------
    # Remove any group name to skip it entirely.  Default (omit key) = all.
    #   "bulk"   : F0, dF, dF/F0, per-component means/stds  (fast, usually keep)
    #   "acf_lr" : lr ACF -> tce, tcf, tci
    #   "acf_hr" : hr ACF -> ttu
    #   "sf"     : structure function  (slowest; requires hr data)
    #   "psd"    : PSD -> qi, qk, fb   (requires hr data)
    "compute": {"bulk", "acf_lr", "acf_hr", "sf", "psd"},

    # --- Output control --------------------------------------------------
    "store_data": True,        # include data_hr / data_lr arrays in pickle
    "log_lags":   False,        # log-spaced structure function lags

    # --- Reproducibility -------------------------------------------------
    "random_seed": 1,
}