import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import matplotlib.pyplot as plt
# from matplotlib import rc
# rc('text', usetex=True)
# rc('xtick', labelsize=20)
# rc('ytick', labelsize=20)

import astropy.units as u
from astropy.timeseries import BoxLeastSquares

import lightkurve
from lightkurve import search_targetpixelfile, search_lightcurve
from astropy.timeseries import LombScargle 

def load_masked_lc(file_name, meta=None):
    file = np.load(file_name)
    lc = lightkurve.LightCurve(file[0], file[1], file[2])
    
    if meta is None:
        return lc
    else:
        lc.meta = meta
        return lc

class TransitModel(object):

    def __init__(self, KICID, window=51, porb_max=None, download_dir=None):

        self.KICID = KICID
        self.window = window
        
        if download_dir is None:
            tpf = search_targetpixelfile(KICID, cadence="long").download()
        else:
            tpf = search_targetpixelfile(KICID, cadence="long").download(download_dir=download_dir)

        self.lc_raw = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
        self.lc_flat = self.lc_raw.flatten(window_length=self.window)
        
        self.meta = self.lc_raw.meta
        
        if porb_max is None:
            porb_max = (np.max(self.lc_raw.time.value) - np.min(self.lc_raw.time.value))/2
        self.porb_max = porb_max

        self.pname = [r"$a_1$", r"$t_1$", r"$\sigma_1$", r"$a_2$", r"$t_2$", r"$\sigma_2$", r"$P_{orb}$"]


    def estimate_period(self, dur_est=None, method='bls'):
        if dur_est is None:
            dur_est = 1
        bls = BoxLeastSquares(self.lc_flat.time, self.lc_flat.flux, dy=self.lc_flat.flux_err)
        results = bls.autopower(dur_est*u.day)
        period = results.period.value
        power = results.power.value
        best = np.argmax(power)
        ind = np.where((period > 2/60/24) & (period < self.porb_max))[0]
        per_cut = period[ind]
        pow_cut = power[ind]
        self.bls_period = per_cut[np.argmax(pow_cut)]
        return self.bls_period


    def get_arrays(self, lc):
    
        x = lc.time.value
        y = lc.flux.value
        s = lc.flux_err.value
        
        return x, y, s


    def get_folded(self, porb):

        self.lc_fold = self.lc_flat.fold(period=porb)

        x, y, s = self.get_arrays(self.lc_fold)

        return x, y, s


    def model(self, theta):
        
        a1, t1, d1, a2, t2, d2, porb = theta

        x, y, s = self.get_folded(porb)

        # Double gaussian transit model
        model_lc = np.ones(len(x))

        offsets = [-porb, 0, porb]
        for tc in offsets:
            model_lc -= a1 * np.exp(-0.5 * (x - t1 + tc)**2 / d1**2) 
            model_lc -= a2 * np.exp(-0.5 * (x - t2 + tc)**2 / d2**2) 
        
        return model_lc, y, s


    def chisq(self, theta, porb_bounds):

        a1, t1, d1, a2, t2, d2, porb = theta
        
        if (porb < porb_bounds[0]) or (porb > porb_bounds[1]):
            return np.inf
        
        # req positive stdev
        if (d1 <= 0) or (d2 <= 0):
            return np.inf
#         elif (d1 >= porb/2) or (d2 >= porb/2):
#             return np.inf
        
        # req postive amplitude
        if (a1 <= 0) or (a2 <= 0):
            return np.inf
        
        # req secondary not w/in duration of primary
        if (t2 < t1) and (t2+d2 > t1-d1):
            return np.inf
        if (t1 < t2) and (t1+d1 > t2-d2):
            return np.inf

        # req transit fit to be between +/-porb/2
        if not (-porb/2 < t1 < porb/2):
            return np.inf
        elif not  (-porb/2 < t2 < porb/2):
            return np.inf
        
        # chi^2 between model and data
        ypred, y, s = self.model(theta)
        chi2 = 0.5 * np.sum((y - ypred)**2/s**2)
        
        return chi2


    def init_optimizer(self, dur_est=.02, porb_est=None):
        
        if porb_est is None:
            porb_est = self.estimate_period()
 
        t0 = np.zeros(7)

        x, y, s = self.get_folded(porb_est)

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

        t0[6] = porb_est

        self.t0 = t0
        
        return self.t0


    def fit_model(self, t0=None, sol=None, method='nelder-mead', porb_bounds=None):
        
        if porb_bounds is None:
            porb_bounds = [0, self.porb_max]
        
        # initial guess 
        if t0 is None:
            t0 = self.t0

        # optimize model fit
        res = minimize(self.chisq, t0, method=method, args=(porb_bounds))
        # self.res = res

        self.sol = res.x
        self.chi_fit = res.fun

        return self.sol, self.chi_fit
    
    def fit_model_period(self, period_guesses=None, method='nelder-mead', dur_est=0.05):
        
        bls_period = self.bls_period
        
        if period_guesses is None:
            period_guesses = [bls_period/4, bls_period/2, bls_period, bls_period*2]
        
        solutions = []
        chis = []
        ts = []
        
        for bls_period in period_guesses:
            
            # initialize model solution guess
            t0 = self.init_optimizer(dur_est=dur_est, porb_est=bls_period)

            # optimize model fit
            sol, chi = self.fit_model(t0=t0, method=method)
            solutions.append(sol)
            chis.append(chi)
            ts.append(t0)
            
        self.chis = chis
        self.sols = solutions
        self.ts = ts
        
        self.sol = solutions[np.argmin(chis)]
        self.t0 = ts[np.argmin(chis)]
        self.chi_fit = np.min(chis)
        
        return self.sol, self.chi_fit
    
    def fit_model_window(self, period_guesses=None, windows=None, method='nelder-mead', dur_est=0.05):
        
        if windows is None:
            windows = [21, 51, 81]
        
        solutions = []
        chis = []
        ts = []
        
        for window in windows:
            
            self.window = window
            self.lc_flat = self.lc_raw.flatten(window_length=self.window)
            self.init_optimizer()
            self.fit_model()
            
            bls_period = self.bls_period
        
            if period_guesses is None:
                period_guesses = [bls_period/4, bls_period/2, bls_period, bls_period*2]
            
            for bls_period in period_guesses:

                # initialize model solution guess
                t0 = self.init_optimizer(dur_est=dur_est, porb_est=bls_period)

                # optimize model fit
                sol, chi = self.fit_model(t0=t0, method=method)
                solutions.append(sol)
                chis.append(chi)
                ts.append(t0)
            
        self.chis = chis
        self.sols = solutions
        self.ts = ts
        
        self.sol = solutions[np.argmin(chis)]
        self.t0 = ts[np.argmin(chis)]
        self.chi_fit = np.min(chis)
        
        return self.sol, self.chi_fit

    def est_duration(self, sol=None, bound=1e-3):
        """
        returns duration estimate, given a solution array (a1, t1, d1, a2, t2, d2, porb)

        TO BE IMPLEMENTED
        """

        if sol is None:
            sol = self.sol
            
        # Estimate duration by diff of these gaussian percentiles
        perc = [bound, 1-bound]

        dmin1 = norm.ppf(perc[0], loc=sol[1], scale=sol[2])
        dmax1 = norm.ppf(perc[1], loc=sol[1], scale=sol[2])
        duration1 = dmax1 - dmin1

        dmin2 = norm.ppf(perc[0], loc=sol[4], scale=sol[5])
        dmax2 = norm.ppf(perc[1], loc=sol[4], scale=sol[5])
        duration2 = dmax2 - dmin2
        
        self.dur1 = duration1
        self.dur2 = duration2
        
        return self.dur1, self.dur2


    def est_eccentricity(self, sol=None):
        """
        returns eccentricity estimate, given a solution array (a1, t1, d1, a2, t2, d2, porb)

        TO BE IMPLEMENTED
        """
        
        if sol is None:
            sol = self.sol
        
        t1 = sol[1]
        t2 = sol[4]
        d1 = sol[2]
        d2 = sol[5]
        porb = sol[6]
        
        t_minus_t = np.abs(t2-t1)
        ecos = 0.5 * np.pi * (t_minus_t/porb - 0.5)

        d_over_d = d2/d1
        esin = (d_over_d - 1) / (d_over_d + 1)

        e2 = ecos**2 + esin**2

        self.ecosw = ecos
        self.esinw = esin
        self.ecc = np.sqrt(e2)

        return self.ecc


    def apply_transit_mask(self, sol=None):
        """
        returns transit and rotation masks, given a solution array (a1, t1, d1, a2, t2, d2, porb)

        TO BE IMPLEMENTED
        """
        if sol is None:
            sol = self.sol
            
        porb = sol[6]
        dur1, dur2 = self.est_duration(sol)
        x, y, s = self.get_arrays(self.lc_raw)
        
        bls = BoxLeastSquares(x*u.day, y, dy=s)

        dur = max(dur1, dur2)
        tmask1 = bls.transit_mask(x, porb, dur, min(x)+sol[1])
        tmask2 = bls.transit_mask(x, porb, dur, min(x)+sol[4])
        tmask = tmask1 | tmask2
        rmask = ~tmask

        self.tmask = tmask
        self.rmask = rmask
    
        return self.tmask, self.rmask


    def save_masked_lcs(self, file_path=None):
        """
        some function which takes self.tmask and self.rmask, 
        applies them to self.lc_raw, and saves the masked arrays to 
        a numpy file

        TO BE IMPLEMENTED
        """
        
        ID = self.KICID[4:] 
        
        if file_path is None:
            file_path = './saved_lightcurves/'
        
        time = self.lc_raw.time.value
        flux = self.lc_raw.flux.value
        err = self.lc_raw.flux_err.value
        
        t_rmask = time[self.rmask]
        f_rmask = flux[self.rmask]
        e_rmask = err[self.rmask]
        
        t_tmask = time[self.tmask]
        f_tmask = flux[self.tmask]
        e_tmask = err[self.tmask]
        
        # Store as np arrays to save
        self.lc_rmask_array = [t_rmask, f_rmask, e_rmask]
        self.lc_tmask_array = [t_tmask, f_tmask, e_tmask]
        
        # Store as lk objects
        self.lc_rmask = lightkurve.LightCurve(time=t_rmask, flux=f_rmask, flux_err=e_rmask)
        self.lc_rmask.meta = self.meta
        self.lc_tmask = lightkurve.LightCurve(time=t_tmask, flux=f_tmask, flux_err=e_tmask)
        self.lc_tmask.meta = self.meta
        
        np.save(file_path + f'KIC_{ID}_rmasked', self.lc_rmask_array)
        np.save(file_path + f'KIC_{ID}_tmasked', self.lc_tmask_array)
    
    
    def model_fit_summary(self):
        """
        returns a dictionary of all the measured parameters
        """

        summary = {"KICID": self.KICID,
                   "window": self.window,
                   "dur1": self.dur1,
                   "dur2": self.dur2, 
                   "ecosw": self.ecosw,
                   "esinw": self.esinw,
                   "ecc": self.ecc}

        for ii, param in enumerate(self.pname):
            summary[param] = self.sol[ii] 

        return summary


    def plot_best_fit(self, figsize=None, t0=None, sol=None):
        """
        4-panel plot with raw lightcurve, flattened lc, model fit, and masked lc

        TO BE IMPLEMENTED
        """
        
        if t0 is None:
            t0 = self.t0
        if sol is None:
            sol = self.sol
        dur1 = self.dur1
        dur2 = self.dur2
        
        self.get_folded(sol[6])       
        tfold = self.lc_fold.time.value
        ffold = self.lc_fold.flux.value
        
        lc_init, _, _ = self.model(t0)
        lc_model, _, _ = self.model(sol)
        
        if figsize is None:
            figsize = (16,30)
        
        fig, ax = plt.subplots(4, 1, figsize=figsize)
        
        self.lc_raw.scatter(ax=ax[0], c='black')
        self.lc_flat.scatter(ax=ax[1], c='black')
        
        label = ''
        for ii in range(len(self.pname)):
            label += f"{self.pname[ii]}={np.round(sol[ii],3)}\n"
        label += r"$\chi^2=%s$"%(int(np.round(self.chi_fit)))
        
        ax[2].scatter(tfold, ffold, color='k', s=3)
        ax[2].plot(tfold, lc_init, color='b')
        ax[2].plot(tfold, lc_model, color='r', label=label)
        ax[2].axvline(sol[1], linestyle='--')
        ax[2].axvline(sol[4], linestyle='--')
        ax[2].axvspan(sol[1]-dur1/2, sol[1]+dur1/2, alpha=.1)
        ax[2].axvspan(sol[4]-dur2/2, sol[4]+dur2/2, alpha=.1)
        ax[2].set_xlim(min(tfold), max(tfold))
        ax[2].legend(loc='best', fontsize=22, frameon=False)
        
        self.lc_rmask.scatter(ax=ax[3], c='black')
        self.lc_tmask.scatter(ax=ax[3], c='red')