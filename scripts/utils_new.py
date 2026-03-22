import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sunpy.timeseries import TimeSeries

sys.path.append("external/Equivalent_Spectrum")
sys.path.append("..")

import external.Equivalent_Spectrum.equiv_spectrum as equiv_spectrum


def standardise(data):
    # These funcs can handle missing data
    mean = data.mean()
    std = data.std()
    result = (data - mean) / std
    return result, std


def filter_scalar_values(d):
    return {
        k: v
        for k, v in d.items()
        if not isinstance(v, (np.ndarray, pd.DataFrame, pd.Series))
    }


def process_list_of_dicts(data_list):
    """Apply filtering to each dictionary in the list \
        and convert the result into a DataFrame."""
    filtered_list = [filter_scalar_values(d) for d in data_list]
    return pd.DataFrame(filtered_list)


def compute_es(data, dt_s, sf, lag, get_fft=False):
    # f1 = data
    D = 1  # Just using 1D data, even though vector time series
    # D = np.ndim(f1)
    # grid_dims = np.shape(f1)
    # N = np.min(grid_dims)
    N = len(data)
    L = N * dt_s  # L = 2.*np.pi
    phys_dims = [L for _ in range(D)]
    # dx = L / N
    dx = dt_s
    # Assuming sampling frequency is in Hz
    dk = 2.0 * np.pi / L

    # Calculate the second order structure function
    ell = lag * dx
    # Mark's code uses the following:
    # vell, sf = mpi_sf.mpi_sf(f1)
    # ell = vell[:,0]*dx

    # Bin and interpolate the second order structure function
    ell_b, sf_b, _ = equiv_spectrum.bin_data(
        ell,
        sf,
        bin_func=np.nanmean,
        cut_excess=True,
        nan_small=False,
        # min_bin=dx,
        # max_bin=L / 2.0,
        num_bins=32,
        bin_loc="true_center",
        log_space=True,
    )
    ell_b2 = ell_b[np.isfinite(sf_b)]
    sf_b2 = sf_b[np.isfinite(sf_b)]
    sf_b = equiv_spectrum.log_log_interpolate(ell_b2, sf_b2, ell_b)
    # Calculate the Uncorrected estimate, and the Debiased estimate
    try:
        ke, BfekS, fekS = equiv_spectrum.equiv_spectrum(ell_b, sf_b, D, 1.0)
    except Exception as e:
        print(f"Error in calculating equivalent spectrum: {e}")
        ke, BfekS, fekS = np.nan, np.nan, np.nan

    if get_fft:
        try:
            # Compare to FFT spectrum
            kvec, fek = equiv_spectrum.per_spectrum(data, phys_dims)
            fek = fek * (dk / (2.0 * np.pi))
            ko, feko = equiv_spectrum.integrate_spectrum(kvec, fek, phys_dims)
        except Exception as e:
            print(f"Error in FFT spectrum calculation: {e}")
            ko, feko = np.nan, np.nan

        return ke, BfekS, fekS, ell_b, sf_b, ko, feko

    else:
        # Return the equivalent spectrum and the binned structure function
        return ke, BfekS, fekS, ell_b, sf_b


def simulate_turbulence(n_points=86400, dt=1.0):
    """
    Simulate turbulence.

    Parameters:
    - n_points: Number of data points (default: 86400, representing 24 hours of 1 Hz data)

    Returns:
    - t: Time array in hours
    - y: Simulated time series data
    """

    # Time array (in hours)
    t = np.arange(n_points) * dt

    # 1. Generate turbulent component with approximate -5/3 power spectrum (Kolmogorov's law)
    # We'll use fractional Gaussian noise for this
    freqs = np.fft.rfftfreq(n_points, d=dt)
    freqs[0] = freqs[1]  # Avoid division by zero

    # Generate complex amplitudes with power law spectrum
    amplitude = freqs ** (
        -5 / 6
    )  # We use -5/6 because we'll square the amplitude later
    phase = 2 * np.pi * np.random.random(len(freqs))

    # Create complex Fourier coefficients
    f_coeffs = amplitude * np.exp(1j * phase)
    f_coeffs[0] = 0  # Remove DC component

    # Generate the turbulent component via inverse FFT
    turbulence = np.fft.irfft(f_coeffs, n=n_points)

    # Scale the turbulence component
    turbulence = turbulence / np.std(turbulence) * 10

    return t, turbulence


def SmoothySpec(a, nums=None):
    """Smooth a curve using a moving average smoothing"""
    b = a.copy()
    if nums is None:
        nums = 2 * len(b) // 3
    for i in range(nums):
        b[i + 1 : -1] = 0.25 * b[i:-2] + 0.5 * b[i + 1 : -1] + 0.25 * b[i + 2 :]
    return b


def fitpowerlaw(ax, ay, xi, xf):
    idxi = np.argmin(abs(ax - xi))
    idxf = np.argmin(abs(ax - xf))
    xx = np.linspace(xi, xf, 100)
    z = np.polyfit(np.log(ax[idxi:idxf]), np.log(ay[idxi:idxf]), 1)
    p = np.poly1d(z)
    pwrl = np.exp(p(np.log(xx)))
    return z, xx, pwrl


def compute_spectral_stats(
    time_series,
    f_min_inertial=None,
    f_max_inertial=None,
    f_min_kinetic=None,
    f_max_kinetic=None,
    timestamp=None,
    di=None,
    velocity=None,
    plot=False,
):
    """Computes the power spectrum for a scalar or vector time series.
    Also computes the power-law fit in the inertial and kinetic ranges,
    and the spectral break between the two ranges, if specified.

    ### Args:

    - time_series: list of 1 (scalar) or 3 (vector) pd.Series. The function automatically detects
    the cadence if timestamped index, otherwise dt = 1s
    - f_min_inertial: (Optional) Minimum frequency for the power-law fit in the inertial range
    - f_max_inertial: (Optional) Maximum frequency for the power-law fit in the inertial range
    - f_min_kinetic: (Optional) Minimum frequency for the power-law fit in the kinetic range
    - f_max_kinetic: (Optional) Maximum frequency for the power-law fit in the kinetic range
    - timestamp: (Optional, only used for plotting) Timestamp of the data
    - di: (Optional, only used for plotting) Ion inertial length in km
    - velocity: (Optional, only used for plotting) Solar wind velocity in km/s
    - plot: (Optional) Whether to plot the PSD

    ### Returns:

    - z_i: Slope in the inertial range
    - z_k: Slope in the kinetic range
    - spectral_break: Frequency of the spectral break between the two ranges
    - f_periodogram: Frequency array of the periodogram
    - power_periodogram: Power array of the periodogram
    - p_smooth: Smoothed power array of the periodogram
    - xi: Frequency array of the power-law fit in the inertial range
    - xk: Frequency array of the power-law fit in the kinetic range
    - pi: Power array of the power-law fit in the inertial range
    - pk: Power array of the power-law fit in the kinetic range


    """

    # Check if the data has a timestamp index
    if isinstance(time_series[0].index, pd.DatetimeIndex):
        # Get the cadence of the data
        dt = time_series[0].index[1] - time_series[0].index[0]
        dt = dt.total_seconds()
    else:
        # If not, assume 1 second cadence
        dt = 1

    x_freq = 1 / dt

    # Convert the time series into a numpy array
    np_array = np.array(time_series)

    if np_array.shape[0] == 3:  # If the input is a vector
        f_periodogram, power_periodogram_0 = signal.periodogram(
            np_array[0], fs=x_freq, window="boxcar", scaling="density"
        )
        power_periodogram_0 = (x_freq / 2) * power_periodogram_0

        f_periodogram, power_periodogram_1 = signal.periodogram(
            np_array[1], fs=x_freq, window="boxcar", scaling="density"
        )
        power_periodogram_1 = (x_freq / 2) * power_periodogram_1

        f_periodogram, power_periodogram_2 = signal.periodogram(
            np_array[2], fs=x_freq, window="boxcar", scaling="density"
        )
        power_periodogram_2 = (x_freq / 2) * power_periodogram_2

        power_periodogram = (
            power_periodogram_0 + power_periodogram_1 + power_periodogram_2
        ) / 3

    elif np_array.shape[0] == 1:  # If the input is a scalar
        f_periodogram, power_periodogram = signal.periodogram(
            np_array[0], fs=x_freq, window="boxcar", scaling="density"
        )
        power_periodogram = (x_freq / 2) * power_periodogram

    # Slowest part of this function - takes ~ 10 seconds
    p_smooth = SmoothySpec(power_periodogram)

    # If the user has specified a range for the power-law fits
    if f_min_inertial is not None:
        qk, xk, pk = fitpowerlaw(
            f_periodogram, p_smooth, f_min_kinetic, f_max_kinetic
        )  # Kinetic range
        qi, xi, pi = fitpowerlaw(
            f_periodogram, p_smooth, f_min_inertial, f_max_inertial
        )  # Inertial range

        try:
            powerlaw_intersection = np.roots(qk - qi)
            spectral_break = np.exp(powerlaw_intersection)
        except Exception as e:
            print("could not compute power-law intersection: {}".format(e))
            spectral_break = [np.nan]

        if round(spectral_break[0], 4) == 0 or spectral_break[0] > 1:
            spectral_break = [np.nan]

    else:
        qi = [np.nan]
        qk = [np.nan]
        spectral_break = [np.nan]
        xi = [np.nan]
        xk = [np.nan]
        pi = [np.nan]
        pk = [np.nan]

    if plot is True:
        fig, ax = plt.subplots(figsize=(3.3, 2), constrained_layout=True)
        ax.set_ylim(1e-6, 1e6)

        ax.semilogy(
            f_periodogram,
            power_periodogram,
            label="Raw periodogram",
            color="black",
            alpha=0.2,
        )
        ax.semilogy(
            f_periodogram, p_smooth, label="Smoothed periodogram", color="black"
        )

        # If the power-law fits have succeeded, plot them
        if not np.isnan(qi[0]):
            ax.semilogy(
                xi,
                pi * 3,
                c="black",
                ls="--",
                lw=0.8,
                label="Inertial range power-law fit: $\\alpha_i$ = {0:.2f}".format(
                    qi[0]
                ),
            )
            ax.semilogy(
                xk,
                pk * 3,
                c="black",
                ls="--",
                lw=0.8,
                label="Kinetic range power-law fit: $\\alpha_k$ = {0:.2f}".format(
                    qk[0]
                ),
            )

        ax.tick_params(which="both", direction="in")
        ax.semilogx()

        if spectral_break[0] is not np.nan:
            ax.axvline(
                np.exp(np.roots(qk - qi)),
                alpha=0.6,
                color="black",
                label="Spectral break: $f_d={0:.2f}$".format(spectral_break[0]),
            )

        # Adding in proton inertial frequency
        if di is not None and velocity is not None:
            f_di = velocity / (2 * np.pi * di)
            ax.axvline(
                f_di,
                color="black",
                alpha=0.6,
                label="Proton inertial frequency: $f_{di}=$" + "{0:.2f}".format(f_di),
            )
            ax.text(f_di * 1.2, 1e-5, "$f_{{di}}$")

        # bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5)
        ax.text(xi[0] * 5, pi[0], "$f^{q_i}$")
        ax.text(xk[0] * 2, pk[0], "$f^{q_k}$")
        ax.text(spectral_break[0] / 2, 1e-5, "$f_b$")

        if timestamp is not None:
            # Add box with timestamp and values of qi and qk
            textstr = "\n".join(
                (
                    str(timestamp[:-3])
                    + "-"
                    + "23:59",  # NOTE - this is a hacky way to get the end timestamp
                    r"$q_i=%.2f$" % (qi[0],),
                    r"$q_k=%.2f$" % (qk[0],),
                    r"$f_b=%.2f$" % (spectral_break[0],),
                    r"$f_{{di}}=%.2f$" % (f_di,),
                )
            )
            props = dict(boxstyle="round", facecolor="gray", alpha=0.2)
            # Place the text box. (x, y) position is in axis coordinates.
            ax.text(
                0.05,
                0.1,
                textstr,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                bbox=props,
            )

        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (nT$^2$Hz$^{-1}$)")
        # plt.grid()
        # plt.show()

        return qi[0], qk[0], spectral_break[0], f_periodogram, p_smooth, fig, ax
    else:
        return (
            qi[0],
            qk[0],
            spectral_break[0],
            f_periodogram,
            power_periodogram,
            p_smooth,
            xi,
            xk,
            pi,
            pk,
        )


def compute_structure_function(data, lag=None, max_lag=None):
    """
    Compute the second-order structure function of a time series.

    Parameters:
    - data: Input time series
    - lag: Array of lag values to calculate (if None, automatically determined)
    - max_lag: Maximum lag to calculate (default: 1/4 of data length)

    Returns:
    - lag: Lag array
    - sf: Second-order structure function
    """
    n = len(data)

    if max_lag is None:
        max_lag = n // 2

    if lag is None:
        # Create logarithmically spaced lag for better visualization
        lag = np.unique(np.logspace(0, np.log10(max_lag), 100).astype(int))
        # lag = lag[lag > 0]  # Ensure no zero lag

    sf = np.zeros(len(lag))
    sf4 = np.zeros(len(lag))

    for i, lag in enumerate(lag):
        # Calculate squared differences for all possible pairs at this lag
        diff = data[lag:] - data[:-lag]
        sf[i] = np.nanmean(diff**2)
        sf4[i] = np.nanmean(diff**4)

    kurtosis = sf4 / (sf**2)  # Excess kurtosis

    return lag, sf, kurtosis


# Function to compute slopes to verify scaling laws
def compute_scaling_slope(x, y, range_start, range_end):
    """Compute the slope of log(y) vs log(x) in the specified range."""
    mask = (x >= range_start) & (x <= range_end)
    if np.sum(mask) < 2:
        return None

    logx = np.log(x)
    logy = np.log(y)

    # Linear regression
    A = np.vstack([logx, np.ones(len(logx))]).T
    slope, _ = np.linalg.lstsq(A, logy, rcond=None)[0]

    return slope


def get_data(input_filepath, mag_vars, dt, retain_original_gaps=False):
    print(f"\n\nREADING FILE {input_filepath}")

    # FOR TESTING ONLY
    # input_filepath = raw_file_list[0]
    # config = config

    # Load data
    data = TimeSeries(input_filepath, concatenate=True)
    df_raw = data.to_dataframe()

    del data

    # Extract variables of interest
    df_raw = df_raw.loc[:, mag_vars]

    start_time = df_raw.index[0]
    end_time = df_raw.index[-1]
    int_length = (end_time - start_time).round("s")

    # Print number of rows and time range
    print(
        f"Loaded {len(df_raw)} rows = {int_length} of data, from {start_time} to {end_time}"
    )

    # Rename the "mag_vars" columns

    df_raw = df_raw.rename(
        columns={
            mag_vars[0]: "Bx",
            mag_vars[1]: "By",
            mag_vars[2]: "Bz",
        }
    )

    # Calculate modal cadence
    # Calculate time differences in seconds
    time_diffs = df_raw.index.to_series().diff().dt.total_seconds().dropna()

    # Find modal cadence and its frequency
    diff_counts = time_diffs.value_counts()
    modal_cadence = diff_counts.idxmax()

    # Count points within 1% of modal cadence
    lower_bound = modal_cadence * 0.95
    upper_bound = modal_cadence * 1.05
    within_range_count = diff_counts[
        (diff_counts.index >= lower_bound) & (diff_counts.index <= upper_bound)
    ].sum()

    # Calculate proportion and missing points
    total_points = len(time_diffs)
    proportion_within_1_percent = within_range_count / total_points
    print(
        f"Modal cadence = {modal_cadence:.5f}s ~ {1/modal_cadence:.2f} samples/s ({proportion_within_1_percent*100:.1f}% of data are within 5% of this cadence)"
    )
    # Resample to modal cadence, to get more accurate missing %
    df_raw_res = df_raw.resample(str(modal_cadence) + "s").mean()
    if df_raw_res.isna().sum().sum() > 0:
        print(
            "Percentage of points missing for each RAW variable, assuming this cadence:"
        )
        if proportion_within_1_percent < 0.9:
            print(
                "(NB: Inconsistent cadence means this may not be an appropriate measure)"
            )
        print((df_raw_res.isna().sum() / len(df_raw_res) * 100).round(4).to_string())
    else:
        print("No missing values in the raw data.")

    del df_raw_res

    # Resample and handle NaN values
    print(f"\nResampling to {dt} cadence...")
    df = df_raw.resample(dt).mean()

    del df_raw

    if df.isna().sum().sum() > 0:
        print("Updated missing percentages:")
        print((df.isna().sum() / len(df) * 100).round(4).to_string())
        if retain_original_gaps:
            print("These gaps are left in the data for now.")
        else:
            print(
                "These remaining missing rows are now filled with linear interpolation."
            )
            df = df.interpolate(method="linear").ffill().bfill()
            if df.isna().sum().sum() > 0:
                print("WARNING: Still NaN values after resampling and interpolation.")
    else:
        print("No missing data after resampling; no interpolation needed.")

    return df


def split_into_intervals(dataframe, config, standardised=False):
    """
    Split dataframe into intervals of specified length

    Parameters:
    - dataframe: pandas DataFrame with timestamp index
    - interval_length: string specifying pandas time offset ('1H', '30min', etc.)

    Returns:
    - List of DataFrames, each containing an interval
    """

    intervals = []
    int_idx = 0
    interval_length = config["int_length"]
    cadence = config["cadence"]

    # Get expected # points in each interval
    expected_n_points = pd.Timedelta(interval_length) / pd.Timedelta(cadence)

    # Get the overall end time of the dataset
    dataset_end_time = dataframe.index.max()

    # Use resample to interval data into intervals
    for start_time, interval in dataframe.resample(
        interval_length, label="left", closed="left"
    ):
        end_time = min(start_time + pd.Timedelta(interval_length), dataset_end_time)
        # This may be shorter than the specified interval if the end time is reached

        if not interval.empty:

            # NOW DOING THIS INSIDE GET_CURVES
            # mean = interval.mean()
            # sd = interval.std()
            # if standardised:
            #     # Standardise the data
            #     interval = (interval-mean)/sd
            #     print(f"Standardised interval: mean changed from {mean} to {interval.mean()}; sd from {sd} to {interval.std()}")

            # Count missing points in the first column
            tgp = interval.isna().sum().iloc[0] / len(interval) * 100  # Total Gap Percentage

            metadata = {
                "spacecraft": config["spacecraft"],
                "interval_id": int_idx,
                "start_time": start_time,
                "end_time": end_time,
                "int_length": interval_length,
                "cadence": config["cadence"],
                "f_fit_range_inertial": config["f_fit_range_inertial"],
                "f_fit_range_kinetic": config["f_fit_range_kinetic"],
                "n_points_complete": len(interval),
                "data": interval.copy(),
                "standardised": standardised,
                "tgp": tgp,  # Total Gap Percentage
            }
            # Check for interval that seems too short
            if len(interval) < 0.9*expected_n_points:
                print(
                    f"Interval starting {start_time} has less than 90% of expected points, skipping"
                )
            else:
                intervals.append(metadata)
            int_idx += 1

    # Print summary of intervals
    if intervals:
        avg_points = sum(interval["n_points_complete"] for interval in intervals) / len(
            intervals
        )
        print(
            f"Split into {len(intervals)} intervals of length {interval_length}, each with ~{avg_points:.0f} points at {dataframe.index.freqstr} resolution"
        )

    return intervals


def compute_corr_scale_exp_trick(
    lags: np.ndarray,
    acf: np.ndarray,
    plot=False,
):
    """
    Computes the correlation scale by finding where the autocorrelation function drops to a specified threshold (default: 1/e).

    Parameters:
    -----------
    lags : np.ndarray
        The x-values (time lags) of the autocorrelation function
    acf : np.ndarray
        The y-values (correlation coefficients) of the autocorrelation function
    threshold : float, optional
        The threshold value to find (default: 1/e ≈ 0.368)
    plot : bool, optional
        Whether to plot the result (default: False)

    Returns:
    --------
    float
        The estimated correlation scale
    """
    # Find the first point where autocorrelation drops below threshold
    threshold = np.exp(-1)

    below_threshold = acf <= threshold

    # If no values are below threshold, return the maximum lag
    if not np.any(below_threshold):
        return lags[-1]

    # Find the index of the first value below threshold
    idx_2 = np.argmax(below_threshold)

    # If it's the first point, we can't interpolate
    if idx_2 == 0:
        return lags[0]

    # Get coordinates for linear interpolation
    idx_1 = idx_2 - 1
    x1, y1 = lags[idx_1], acf[idx_1]
    x2, y2 = lags[idx_2], acf[idx_2]

    # Linear interpolation to find the optimal x value where y = threshold
    x_opt = x1 + ((y1 - threshold) / (y1 - y2)) * (x2 - x1)

    tce = round(x_opt, 3)

    # Optionally plot the result
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), constrained_layout=True)
        # PREVIOUSLY WAS 3.3 WIDE - 2.5 TALL
        # fig.subplots_adjust(left=0.2, top=0.8, bottom=0.8)

        ax.plot(
            lags / 1000,
            acf,
            c="black",
            label="Autocorrelation",
            lw=0.5,
        )
        ax.set_xlabel("$\\tau (10^3$ s)")
        ax.set_ylabel("$R(\\tau)$")

        def sec2km(x):
            return x * 1000 * 400 / 1e6

        def km2sec(x):
            return x / 1000 / 400 * 1e6

        # use of a float for the position:
        secax_x2 = ax.secondary_xaxis("top", functions=(sec2km, km2sec))
        secax_x2.set_xlabel("$r$ ($10^6$ km)")
        secax_x2.tick_params(which="both", direction="in")
        ax.axhline(
            np.exp(-1),
            color="black",
            ls="--",
            label="$1/e\\rightarrow\\lambda_C^{{1/e}}$={:.0f}s".format(tce),
        )
        ax.axvline(tce / 1000, color="black", ls="--")
        ax.tick_params(which="both", direction="in")
        # label="$1/e\\rightarrow \lambda_C^{1/e}=${:.0f}s".format(tce))
        return tce, fig, ax

    return tce


def compute_outer_scale_exp_trick(
    autocorrelation_x: np.ndarray, autocorrelation_y: np.ndarray, plot=False
):
    """
    computes the correlation scale through the "1/e" estimation method.
    autocorrelation_x assumed already in time scale
    """
    for i, j in zip(autocorrelation_y, autocorrelation_x):
        if i <= np.exp(-1):
            # print(i, j)
            idx_2 = np.where(autocorrelation_x == j)[0]
            idx_1 = idx_2 - 1
            x2 = autocorrelation_x[idx_2]
            x1 = autocorrelation_x[idx_1]
            y1 = autocorrelation_y[idx_1]
            y2 = autocorrelation_y[idx_2]
            x_opt = x1 + ((y1 - np.exp(-1)) / (y1 - y2)) * (x2 - x1)
            # print(autocorrelation_x[idx_1], autocorrelation_y[idx_1])
            # print(autocorrelation_x[idx_2], autocorrelation_y[idx_2])
            # print('e:', np.exp(-1))
            # print(x_opt)

            try:
                # Optional plotting, set up to eventually display all 3 corr scale methods
                if plot is True:
                    fig, ax = plt.subplots(
                        1, 1, figsize=(5, 2.5), constrained_layout=True
                    )
                    # PREVIOUSLY WAS 3.3 WIDE - 2.5 TALL
                    # fig.subplots_adjust(left=0.2, top=0.8, bottom=0.8)

                    ax.plot(
                        autocorrelation_x / 1000,
                        autocorrelation_y,
                        c="black",
                        label="Autocorrelation",
                        lw=0.5,
                    )
                    ax.set_xlabel("$\\tau (10^3$ s)")
                    ax.set_ylabel("$R(\\tau)$")

                    def sec2km(x):
                        return x * 1000 * 400 / 1e6

                    def km2sec(x):
                        return x / 1000 / 400 * 1e6

                    # use of a float for the position:
                    secax_x2 = ax.secondary_xaxis("top", functions=(sec2km, km2sec))
                    secax_x2.set_xlabel("$r$ ($10^6$ km)")
                    secax_x2.tick_params(which="both", direction="in")
                    ax.axhline(
                        np.exp(-1),
                        color="black",
                        ls="--",
                        label="$1/e\\rightarrow\\lambda_C^{{1/e}}$={:.0f}s".format(
                            x_opt[0]
                        ),
                    )
                    ax.axvline(x_opt[0] / 1000, color="black", ls="--")
                    ax.tick_params(which="both", direction="in")
                    # label="$1/e\\rightarrow \lambda_C^{1/e}=${:.0f}s".format(x_opt[0]))
                    return round(x_opt[0], 3), fig, ax
                else:
                    return round(x_opt[0], 3)
            except Exception:
                return 0

    # none found
    return np.nan


def exp_fit(r, lambda_c):
    """
    fit function for determining correlation scale, through the optimal lambda_c value
    """
    return np.exp(-1 * r / lambda_c)


def para_fit(x, a):
    """
    fit function for determining taylor scale, through the optimal lambda_c value
    """
    return a * x**2 + 1


def compute_outer_scale_exp_fit(
    time_lags,
    acf,
    seconds_to_fit,
    fig=None,
    ax=None,
    plot=False,
    initial_guess=1000,
):
    dt = time_lags[1] - time_lags[0]
    num_lags_for_lambda_c_fit = int(seconds_to_fit / dt)
    c_opt, c_cov = curve_fit(
        exp_fit,
        time_lags[:num_lags_for_lambda_c_fit],
        acf[:num_lags_for_lambda_c_fit],
        p0=initial_guess,
    )
    lambda_c = c_opt[0]

    # Optional plotting
    if plot is True:
        if fig is not None and ax is not None:
            fig = fig
            ax = ax

            ax.plot(
                np.array(range(int(seconds_to_fit))) / 1000,
                exp_fit(np.array(range(int(seconds_to_fit))), *c_opt),
                label="Exp. fit$\\rightarrow\\lambda_C^{{\\mathrm{{fit}}}}$={:.0f}s".format(
                    lambda_c
                ),
                lw=3,
                c="black",
            )

        return lambda_c, fig, ax
    else:
        return lambda_c


def compute_outer_scale_integral(time_lags, acf, fig=None, ax=None, plot=False):

    dt = time_lags[1] - time_lags[0]

    # Find where ACF changes sign (crosses zero)
    sign_changes = np.where(np.diff(np.signbit(acf)))[0]

    if len(sign_changes) == 0:
        raise ValueError("ACF does not reach zero; will not compute integral")

    # Get index just before first zero crossing
    idx_before = sign_changes[0]

    integral = np.sum(acf[:idx_before]) * dt  # Computing integral up to that index

    # Optional plotting
    if plot is True:
        # Optional plotting
        if fig is not None and ax is not None:
            fig = fig
            ax = ax

        elif fig is None and ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.5), constrained_layout=True)
        print(time_lags[idx_before])
        ax.fill_between(
            time_lags / 1000,
            0,
            acf,
            where=(acf > 0) & (time_lags < time_lags[idx_before]),
            color="black",
            alpha=0.2,
            label="Integral$\\rightarrow\\lambda_C^{{\mathrm{{int}}}}$={:.0f}s".format(
                integral
            ),
        )
        ax.set_xlabel("$\\tau$ ($10^3$s)")
        ax.tick_params(which="both", direction="in")
        # Plot the legend
        ax.legend(loc="upper right")
        return integral, fig, ax
    else:
        return integral


def compute_taylor_scale(
    time_lags, acf, tau_fit, plot=False, show_intercept=False, xlim=None, ylim=None
):
    """Compute the Taylor microscale

    Args:

    - time_lags: The x-values of the ACF, in seconds, given the cadence of measurements
    - acf: The y-values of the ACF
    - tau_fit: number of lags to fit the parabola over
    """

    # If using seconds_fit as the fitting argument instead:

    dt = time_lags[1] - time_lags[0]
    # tau_fit = int(seconds_fit/dt)

    t_opt, t_cov = curve_fit(
        para_fit, time_lags[:tau_fit], acf[:tau_fit], p0=10
    )  # Initial guess for the parameters
    lambda_t = (-1 * t_opt[0]) ** -0.5

    extended_parabola_x = np.arange(0, 1.2 * lambda_t, 0.1)
    extended_parabola_y = para_fit(extended_parabola_x, *t_opt)

    if plot is True:
        fig, ax = plt.subplots(2, 1, figsize=(3.3, 4), constrained_layout=True)
        # fig.subplots_adjust(hspace=0.1, left=0.2, top=0.8)

        ax[0].scatter(
            time_lags / dt,  # Plotting firstly in lag space for clearer visualisation
            acf,
            label="Autocorrelation",
            s=12,
            c="black",
            alpha=0.5,
        )

        ax[0].plot(
            (extended_parabola_x / dt),
            (extended_parabola_y),
            "-y",
            label="Parabolic fit \nup to $\\tau_\mathrm{fit}\\rightarrow\\tau_\mathrm{TS}^\mathrm{est}$",
            c="black",
        )

        ax[0].axvline(
            tau_fit * (time_lags[1] / dt - time_lags[0] / dt),
            ls="--",
            # label=f"$\\tau_{{fit}}={tau_fit}$ lags",
            c="black",
            alpha=0.6,
        )

        if xlim is not None:
            ax[0].set_xlim(xlim[0], xlim[1])
        else:
            ax[0].set_xlim(-1, 45)
        if ylim is not None:
            ax[0].set_ylim(ylim[0], ylim[1])
        else:
            ax[0].set_ylim(0.986, 1.001)

        if show_intercept is True:
            ax[0].set_ylim(0, 1.05)
            ax[0].set_xlim(-1, 200)  # lambda_t/dt + 5
            ax[0].axvline(lambda_t / dt, ls="dotted", c="black", alpha=0.6)

        ax[0].set_xlabel("$\\tau$ (lags)")
        ax[0].xaxis.set_label_position("top")
        ax[0].set_ylabel("$R(\\tau)$")
        ax[0].tick_params(
            which="both",
            direction="in",
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
        )

        # For plotting secondary axis, in units of r(km)
        def lag2km(x):
            return x * dt * 400

        def km2lag(x):
            return x / (dt * 400)

        secax_x = ax[0].secondary_xaxis(1.3, functions=(lag2km, km2lag))
        secax_x.set_xlabel("$r$ (km)")
        secax_x.tick_params(which="both", direction="in")

        ax[0].legend(loc="upper right")
        ax[0].annotate("(a)", (2, 0.9875), transform=ax[0].transAxes, size=12)
        ax[0].annotate(
            "$\\tau_\mathrm{fit}$",
            (10, 0.9875),
            transform=ax[0].transAxes,
            size=12,
            alpha=0.6,
        )
        # ax[0].annotate('$\\tau_\mathrm{TS}^\mathrm{est}\\rightarrow=$', (35, 0.9875), transform=ax[0].transAxes, size=10, alpha=0.6)

        return lambda_t, fig, ax

    else:
        return lambda_t


def compute_taylor_chuychai(
    time_lags,
    acf,
    tau_min,
    tau_max,
    fig=None,
    ax=None,
    q=None,
    tau_fit_single=None,
    save=False,
    figname="",
):
    """Compute a refined estimate of the Taylor microscale using a linear extrapolation method from Chuychai et al. (2014).

    Args:

    - time_lags: The x-values of the ACF, in seconds, given the cadence of measurements
    - acf: The y-values of the ACF
    - tau_min: Minimum value for the upper lag to fit the parabola over. This should not be too small, because the data has finite time resolution and there may be limited data available at the shortest time lags. (You will see divergent behaviour if this happens.)
    - tau_max: Maximum value for the upper lag to fit the parabola over
    - q: Slope of the dissipation range
    """

    dt = time_lags[1] - time_lags[0]

    tau_fit = np.arange(tau_min, tau_max + 1)
    tau_ts = np.array([])

    for i in tau_fit:
        lambda_t = compute_taylor_scale(time_lags, acf, tau_fit=i)
        tau_ts = np.append(tau_ts, lambda_t)

    # Performing linear extrapolation back to tau_fit = 0
    z, cov = np.polyfit(x=tau_fit, y=tau_ts, deg=1, cov=True)
    f = np.poly1d(z)

    ts_est_extra = z[1]  # Extracting y-intercept

    # Getting standard deviation of y-intercept
    # (will plot +- 1 standard deviation)
    ts_est_extra_std = np.sqrt(cov[1, 1])

    # Getting extrapolation line for plotting
    other_x = np.arange(0, tau_max + 1)
    other_y = f(other_x)

    # Applying correction factor q from Chuychai et al. (2014)
    if q is not None:
        q_abs = np.abs(q)
        if q_abs < 2:
            r = -0.64 * (1 / q_abs) + 0.72
        elif q_abs >= 2 and q_abs < 4.5:
            r = -2.61 * (1 / q_abs) + 1.7
        elif q_abs >= 4.5:
            r = -0.16 * (1 / q_abs) + 1.16

    else:
        r = 1

    ts_est = r * ts_est_extra
    ts_est_std = r * ts_est_extra_std

    # Optional plotting
    if fig is not None and ax is not None:
        ax[1].scatter(
            tau_fit,
            tau_ts,
            label="Fitted values $\\tau_\mathrm{TS}^\mathrm{est}$",
            s=12,
            c="black",
            alpha=0.5,
            marker="x",
        )

        ax[1].plot(
            other_x,
            other_y,
            label="R.E.$\\rightarrow\\tau_\mathrm{{TS}}^\mathrm{{ext}}$={:.0f}s".format(
                ts_est_extra
            ),
            c="black",
        )

        if tau_fit_single is not None:
            ax[1].axvline(
                tau_fit_single,
                ls="--",
                # ymin=0.5,
                # ymax=1,
                c="black",
                alpha=0.6,
            )

        if q is not None:
            ax[1].plot(
                0,
                ts_est,
                "*",
                color="green",
                label="C.C.$\\rightarrow\\tau_\mathrm{{TS}}$={:.0f}s".format(ts_est),
                markersize=10,
            )

        ax[1].set_xlabel("")
        ax[1].set_xticks([])

        ax[1].set_ylabel("$\\tau$(s)")
        ax[1].tick_params(which="both", direction="in")

        # For plotting secondary axis, units of tau(s)
        def sec2lag(x):
            return x / dt

        def lag2sec(x):
            return x * dt

        secax_x2 = ax[1].secondary_xaxis(0, functions=(lag2sec, sec2lag))

        secax_x2.set_xlabel("$\\tau_\\mathrm{fit}$(s)")
        secax_x2.tick_params(which="both", direction="in")

        # Add legend with specific font size
        ax[1].legend(loc="lower right")
        ax[1].set_xlim(-1, 45)  # Set to 200 if wanting to see extrapolation
        ax[1].set_ylim(-3, max(tau_ts) + 1)
        ax[1].annotate("(b)", (2, 24), size=12)

        return ts_est, ts_est_std, fig, ax

    else:
        return ts_est, ts_est_std
