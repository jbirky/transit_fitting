import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

from lightkurve import search_lightcurve

import exoplanet
from scipy.signal import find_peaks

# function to find highest peak
def highest_peak(time,flux):
    try:
        x = time
        y = flux
        x_new = x[np.isfinite(x) & np.isfinite(y)]
        y_new = y[np.isfinite(x) & np.isfinite(y)]
    
        autocorr = exoplanet.autocorr_estimator(x_new, y_new, yerr=None, min_period=0.1, max_period=None, oversample=2.0, smooth=2.0, max_peaks=10)
        lag, acf = autocorr['autocorr']
        peaks, prominence = find_peaks(acf, prominence = 0.1)
        peaks, prominence
    
        p = np.argsort(lag[peaks])
        new_array = lag[peaks][p]
    
        if (prominence['prominences'][p][1] > prominence['prominences'][p][0]):
            high_peak = lag[peaks][p][1]
        else:
            high_peak = lag[peaks][p][0]
        return high_peak
    except:
        return -1

#folding on correct period and make graph
def graph(time, flux, period):
    plt.plot(time%period, flux,
            '.')