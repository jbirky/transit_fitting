import numpy as np
from scipy.optimize import minimize_scalar

__all__ = ["PhaseLightCurve",
           "phase_dispersion",
           "phase_dispersion_minimization"]

class PhaseLightCurve():
    """
    Class to store and perform operations on individual light curves
    """
    def __init__(self, time, flux, flux_err):

        self.time 		= time
        self.flux 		= flux
        self.flux_err 	= flux_err
        self.baseline = max(self.time) - min(self.time)

    def phase_fold(self, period, shift=True, **kwargs):

        phase = self.time/period - np.floor(self.time/period)
        sort_idx = np.argsort(phase)

        self.phase = phase[sort_idx]
        self.phase_flux = self.flux[sort_idx]
        self.phase_flux_err = self.flux_err[sort_idx]
        # self.norm_phase_flux = self.norm_flux[sort_idx]

        if shift == True:
            self.phase_flux = np.roll(np.array(self.phase_flux), len(self.phase_flux)-np.argmin(self.phase_flux))
            # self.norm_phase_flux = np.roll(np.array(self.norm_phase_flux), len(self.norm_phase_flux)-np.argmin(self.norm_phase_flux))

        return np.vstack([self.phase, self.phase_flux, self.phase_flux_err])

    def smooth_data(self, method='rolling_median', window=128):

        bin_flux = []
        if method == 'rolling_median':
            for i in range(len(self.phase_flux)):
                bin_flux.append(np.nanmedian(self.phase_flux[i-window:i+window]))
            # self.bin_flux = pd.Series(self.norm_phase_flux, center=True).rolling(window).median()
        self.bin_flux = np.array(bin_flux)
        self.norm_bin_flux = self.bin_flux/np.nanmedian(self.bin_flux)

        return self.bin_flux


def phase_dispersion(period, *args):

    lc, window = args
    lc.phase_fold(period=period)
    lc.smooth_data(window=window)

    chi_val = np.nansum(((lc.phase_flux - lc.bin_flux)/lc.phase_flux_err)**2)

    return chi_val


def phase_dispersion_minimization(lc, p0, bound=0.1, window=200, factors=[1.0]):

    rng = bound*p0

    pdm_periods, pdm_chivals = [], []
    nevals = 0
    for f in factors:
        mult_period = f*p0
        if mult_period < lc.baseline/2:
            p = minimize_scalar(phase_dispersion, bounds=(mult_period-rng, mult_period+rng), \
                                method='bounded', args=(lc, window))
            pdm_periods.append(p.x)
            pdm_chivals.append(p.fun)
            nevals += p.nfev
    best_ind = np.argmin(pdm_chivals)
    pdm_best_period = np.array(pdm_periods)[best_ind]
    pdm_best_chival = np.array(pdm_chivals)[best_ind]

    # print("\n{0:<24}{1:>24}".format('TIC ID:', lc.tic_id))
    # print("{0:<24}{1:>24}".format('p0 period (days):', p0))
    # print("{0:<24}{1:>24}".format('PDM period (days):', pdm_best_period))
    # print("{0:<24}{1:>24}".format('Chi^2:', pdm_best_chival))
    # print("{0:<24}{1:>24}".format('Iterations:', nevals))
    # print("{0:<24}{1:>24}".format('Compute time (s):', time.time() - t0))

    return pdm_best_period, pdm_best_chival