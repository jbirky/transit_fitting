import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

import astropy.units as u
from astropy.timeseries import BoxLeastSquares

import lightkurve
from lightkurve import search_targetpixelfile, search_lightcurve
from astropy.timeseries import LombScargle 

__all__ = ["get_data",
           "get_arrays",
           "model",
           "chisq",
           "init_optimizer",
           "est_duration",
           "fit_model",
           "fit_best_window",
           "apply_transit_mask",
           "plot_best_fit"]


def get_data(KICID, porb, window=51, plot=False):
    
    warnings.simplefilter("ignore")
    tpf = search_targetpixelfile(KICID, cadence="long").download()
    lc_raw = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    
#     tpf = search_lightcurve(KICID, cadence="long").download_all(download_dir='data')
#     lc_raw = tpf.stitch()

    lc_flat = lc_raw.flatten(window_length=window)
    lc_fold = lc_flat.fold(period=porb)
    
    if plot == True:
        lc_raw.plot();
        lc_flat.plot();
        lc_fold.plot();
    
    return lc_raw, lc_flat, lc_fold


def get_arrays(lc):
    
    x = lc.time.value
    y = lc.flux.value
    s = lc.flux_err.value
    
    return x, y, s


def model(theta, x=None, porb=None):
    
    # Double gaussian transit model
    a1, t1, d1, a2, t2, d2 = theta
    transit1 = a1 * np.exp(-0.5 * (x - t1)**2 / d1**2) 
    transit2 = a2 * np.exp(-0.5 * (x - t2)**2 / d2**2) 
    transits = transit1 + transit2
    
    left_edge1 = a1 * np.exp(-0.5 * (x - t1 + porb)**2 / d1**2)
    left_edge2 = a2 * np.exp(-0.5 * (x - t2 + porb)**2 / d2**2) 
    
    right_edge1 = a1 * np.exp(-0.5 * (x - t1 - porb)**2 / d1**2)
    right_edge2 = a2 * np.exp(-0.5 * (x - t2 - porb)**2 / d2**2) 
    edges = left_edge1 + left_edge2 + right_edge1 + right_edge2
    
    return 1 - transits - edges


def chisq(theta, x=None, y=None, s=None, porb=None):
    
    # chi^2 between model and data
    ypred = model(theta, x=x, porb=porb)
    chi2 = 0.5 * np.sum((y - ypred)**2/s**2)
    
    # req positive stdev
    if (theta[2] < 0) or (theta[5] < 0):
        return np.inf
    
    # req postive amplitude
    if (theta[0] < 0) or (theta[3] < 0):
        return np.inf
    
    # req secondary not w/in duration of primary
    if (theta[1]-theta[2] < theta[4] < theta[1]+theta[2]):
        return np.inf
    
    return chi2


def init_optimizer(x, y, dur_est=.1):
    
    t0 = np.zeros(6)

    # estimate gaussian parameters for primary
    pri_ind = np.argmin(y)
    
    t0[0] = 1 - y[pri_ind]
    t0[1] = x[pri_ind]
    t0[2] = dur_est
    
    # estimate gaussian parameters for secondary
    pri_mask = np.where((x < x[pri_ind]-.25) | (x > x[pri_ind]+.25))[0]
    sec_ind = np.argmin(y[pri_mask])
    
    t0[3] = 1 - y[pri_mask][sec_ind]
    t0[4] = x[pri_mask][sec_ind]
    t0[5] = dur_est
    
    return t0


def est_duration(theta, bound=1e-3):
    
    # Estimate duration by diff of these gaussian percentiles
    perc = [bound, 1-bound]
    
    dmin1 = norm.ppf(perc[0], loc=theta[1], scale=theta[2])
    dmax1 = norm.ppf(perc[1], loc=theta[1], scale=theta[2])
    duration1 = dmax1 - dmin1
    
    dmin2 = norm.ppf(perc[0], loc=theta[4], scale=theta[5])
    dmax2 = norm.ppf(perc[1], loc=theta[4], scale=theta[5])
    duration2 = dmax2 - dmin2
    
    return duration1, duration2


def fit_model(x, y, s, porb):
    
    t0 = init_optimizer(x, y, dur_est=.05)

    warnings.simplefilter("ignore")
    res = minimize(chisq, t0, method='powell', args=(x, y, s, porb))
    
    return t0, res.x, res.fun


def fit_best_window(KICID, porb):

    windows = np.arange(10,120,10)+1

    cfit = []
    for w in windows:
        lc_raw, lc_flat, lc_fold = get_data(KICID, porb, window=w)
        x, y, s = get_arrays(lc_fold)
        t0, sol, chi_fit = fit_model(x, y, s, porb)
        cfit.append(chi_fit)
        
    wbest = windows[np.argmin(cfit)]

    lc_raw, lc_flat, lc_fold = get_data(KICID, porb, window=wbest)
    x, y, s = get_arrays(lc_fold)
    t0, sol, chi_fit = fit_model(x, y, s, porb)
    
    return t0, sol, chi_fit, wbest, lc_raw, lc_flat, lc_fold


def apply_transit_mask(x, y, sol, porb):
    
    bls = BoxLeastSquares(x*u.day, y, dy=None)

    dur1, dur2 = est_duration(sol, bound=1e-3)
    tmask1 = bls.transit_mask(x, porb, dur1, min(x)+sol[1])
    tmask2 = bls.transit_mask(x, porb, dur2, min(x)+sol[4])
    tmask = tmask1 | tmask2
    rmask = ~tmask
    
    return tmask, rmask


def plot_best_fit(KICID, porb, t0, sol, chi_fit, wbest, lc_raw, lc_flat, lc_fold, show=False):
    
    fig, ax = plt.subplots(4,1, figsize=[16,30])

    # Plot raw light curve
    traw, fraw = lc_raw.time.value, lc_raw.flux.value
    ax[0].plot(traw, fraw, color='k')
    ax[0].set_xlim(min(traw), max(traw))

    # Plot flattened light curve
    tflat, fflat = lc_flat.time.value, lc_flat.flux.value
    ax[1].plot(tflat, fflat, color='k', label=f"window={wbest}")
    ax[1].set_xlim(min(tflat), max(tflat))
    ax[1].legend(loc='best', fontsize=22, frameon=False)

    # Plot folded light curve and best model fit
    tfold, ffold = lc_fold.time.value, lc_fold.flux.value
    dur1, dur2 = est_duration(sol, bound=1e-3)
    pname = [r"$a_1$", r"$t_1$", r"$\sigma_1$", r"$a_2$", r"$t_2$", r"$\sigma_2$"]
    label = ''
    for ii in range(len(pname)):
        label += f"{pname[ii]}={np.round(sol[ii],3)}\n"
    label += r"$\chi^2=%s$"%(int(np.round(chi_fit)))

    ax[2].scatter(tfold, ffold, color='k', s=3)
    ax[2].plot(tfold, model(t0, x=tfold, porb=porb), color='b')
    ax[2].plot(tfold, model(sol, x=tfold, porb=porb), color='r', label=label)
    
    ax[2].axvline(sol[1], linestyle='--')
    ax[2].axvline(sol[4], linestyle='--')
    ax[2].axvspan(sol[1]-dur1/2, sol[1]+dur1/2, alpha=.1)
    ax[2].axvspan(sol[4]-dur2/2, sol[4]+dur2/2, alpha=.1)
    ax[2].axvspan((sol[1]+porb)-dur1/2, (sol[1]+porb)+dur1/2, alpha=.1)
    ax[2].axvspan((sol[4]+porb)-dur2/2, (sol[4]+porb)+dur2/2, alpha=.1)
    ax[2].axvspan((sol[1]-porb)-dur1/2, (sol[1]-porb)+dur1/2, alpha=.1)
    ax[2].axvspan((sol[4]-porb)-dur2/2, (sol[4]-porb)+dur2/2, alpha=.1)
    
    ax[2].set_xlim(min(tfold), max(tfold))
    ax[2].legend(loc='best', fontsize=22, frameon=False)

    # Plot transit mask
    tmask, rmask = apply_transit_mask(traw, fraw, sol, porb)
    ax[3].scatter(traw[tmask], fraw[tmask], s=3, color='k')
    ax[3].scatter(traw[rmask], fraw[rmask], s=3, color='r')
    ax[3].set_xlim(min(traw), max(traw))

    ax[0].set_title(KICID, fontsize=25)
    plt.tight_layout()
    if show == True:
        plt.show()
    plt.close()
    
    return fig