import numpy as np
import matplotlib.pyplot as plt
import exoplanet
from scipy.signal import find_peaks

__all__ = ["format_t0_from_table",
           "inspect_rotation",
           "plot_t0"]


def format_t0_from_table(table_entry):
    
    t0_guess = np.zeros(7)
    t0_guess[0]= table_entry['primary transit amplitute']
    t0_guess[1]= table_entry['primary transit time']
    t0_guess[2]= table_entry['primary transit duration']
    t0_guess[3]= table_entry['secondary transit amplitute']
    t0_guess[4]= table_entry['secondary transit time']
    t0_guess[5]= table_entry['secondary transit duration']
    t0_guess[6]= table_entry['orbital period']
    
    return t0_guess


def load_masked_lightcurve(tic_id):

    time, flux, flux_err = np.load(f"masked_lcs/TIC_{tic_id}_rmasked.npy")

    

    return xnew, ynew, enew


def inspect_rotation(time, flux, flux_err, selected_peak=0, ls_period=None, orb_period=None):

    xnew = time[np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)]
    ynew = flux[np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)]
    enew = flux_err[np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)]

    min_period = 0.1
    max_period = (max(time)-min(time))/2

    autocorr = exoplanet.autocorr_estimator(xnew, ynew, yerr=enew, min_period=min_period, max_period=max_period)
    lag_full, acf_full = autocorr['autocorr']

    lag = lag_full[lag_full < max_period]
    acf = acf_full[lag_full < max_period]
    sig_level = 1.96 / np.sqrt(len(lag) - lag)

    peaks, prominence = find_peaks(acf, height=np.mean(sig_level))
    peaks_sorted = np.argsort(acf[peaks])[::-1]

    if len(peaks) > 0:
        max_peak = peaks_sorted[selected_peak]
        best_period = lag[peaks][max_peak]

    fig = plt.figure(figsize=[14,6])
    plt.plot(lag, acf, linewidth=2)
    if len(peaks) > 0:
        for p in peaks:
            plt.axvline(lag[p], linewidth=1)
        plt.axvline(best_period, color='r', label="Rotation period")
    if ls_period is not None:
        plt.axvline(best_period, color='g', label="LS rotation period", linestyle='--')
    if orb_period is not None:
        plt.axvline(orb_period, color='m', label="Orbital period", linestyle='--')
    plt.fill_between(lag, -sig_level, sig_level, color='k', alpha=.1)
    plt.legend(loc="upper right")
    plt.axhline(0, color='k', linewidth=1)
    plt.xlabel('lag [days]', fontsize=20)
    plt.ylabel('ACF', fontsize=20)
    plt.xlim(min(lag), max(lag))
    plt.minorticks_on()
    plt.show()

    # =============================

    if len(peaks) > 0:
        phase = time % best_period
        plt.figure(figsize=[14,6])
        plt.scatter(phase, flux, color='k', s=5)
        plt.xlabel("Phase [days]", fontsize=20)
        plt.ylabel("Flux [normalized]", fontsize=20)
        plt.xlim(min(phase), max(phase))
        plt.show()

    plt.figure(figsize=[14,6])
    plt.scatter(time, flux, color='k', s=5)
    plt.xlabel("Time [days]", fontsize=20)
    plt.ylabel("Flux [normalized]", fontsize=20)
    plt.xlim(min(time), max(time))
    plt.show()
    
    return best_period


def plot_t0(model, t0_guess, tic=None):  
    
    folded_lc = model.get_folded(t0_guess[-1])       
    tfold = folded_lc[0]
    ffold = folded_lc[1]

    init_fit = model.model(t0_guess)

    plt.figure(figsize=[20,6])
    plt.scatter(tfold, ffold, color='k', s=5)
    plt.plot(tfold, init_fit[0], color='b', linewidth=2, alpha=.6)
    plt.title(f"TIC {tic}", fontsize=25)
    plt.xlabel("Orbital Phase [d]", fontsize=22)
    plt.ylabel("Flux", fontsize=22)
    plt.xlim(min(tfold), max(tfold))
    plt.show()