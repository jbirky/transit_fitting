import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import matplotlib.pyplot as plt
from matplotlib import rc
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
import os

import astropy.units as u
from astropy.timeseries import BoxLeastSquares

import lightkurve
from lightkurve import search_targetpixelfile, search_lightcurve
import astropy
from astropy.timeseries import LombScargle

import exoplanet
from scipy.signal import find_peaks

__all__ = ["load_masked_lc",
           "TransitModel"]



def load_masked_lc(file_name, meta=None):
    file = np.load(file_name)
    lc = lightkurve.LightCurve(file[0], file[1], file[2])
    
    if meta is None:
        return lc
    else:
        lc.meta = meta
        return lc

class TransitModel(object):

    def __init__(self,
                 ID,
                 search=None,
                 mission=None,
                 pipeline=None,
                 window=51, 
                 porb_max=None, 
                 download_dir=None,
                 download_all=False,
                 cadence="long", 
                 fill_gaps=False):
        """     
        ID: Target identification number
        mission: Prefix of ID. 'KIC' for Kepler or 'TIC' for TESS. None if included in ID
        pipeline: TESS pipeline to download. Defaults to first found if None given.
        window: Window parameter for flattening function
        porb_max: Maximum orbital period for search range.
        download_dir: Location to download lightcurve data.
        download_all: If True, downloads all quarters.
        quarters: Download all quarters if quarter == 'all', first n integers if quaters is an int, specific quarters if quarters is a list of ints, or first quarter if None. Overrides download_all.
        """
        
        # Ensure proper formatting of ID
        ID = str(ID)
        if (mission == 'Kepler') or (mission == 'K2'):
            mission = 'KIC'
        if (mission == 'TESS'):
            mission = 'TIC'
            
        if mission == None:
            self.ID = ID
            self.mission = ID[:3]
        else:
            self.ID = mission + ID
            self.mission = mission
            
        self.pipeline = pipeline
        self.window = window
        
        self.download_dir = download_dir
        self.fill_gaps = fill_gaps
        self.cadence = cadence 
        
        if search is not None:
            self.search = search
            self.lc_collection = self.search.download_all(download_dir=self.download_dir)
            self.lc_raw = self.stitch()

        elif (search is None) and (download_all == True):
            self.search = search_lightcurve(self.ID, mission=self.mission, cadence=self.cadence)
            self.lc_collection = self.search.download_all(download_dir=self.download_dir)
            self.lc_raw = self.stitch()

        elif (search is None) and (download_all == False):
            self.search = search_lightcurve(self.ID, mission=self.mission, cadence=self.cadence)
            self.raw = self.search.download(download_dir=self.download_dir)

        # flatten lightcurve
        self.lc_flat = self.lc_raw.flatten(window_length=self.window)

        # offset time array to zero
        self.lc_raw.time = self.lc_raw.time.jd #- min(self.lc_raw.time.jd)

        # create flattened lightcurve 
        self.lc_flat = self.lc_raw.flatten(window_length=self.window)
        
        if self.fill_gaps:
            self.lc_flat = self.lc_flat.fill_gaps()
        
        self.meta = self.lc_raw.meta
        
        if porb_max is None:
            porb_max = (np.max(self.lc_raw.time.value) - np.min(self.lc_raw.time.value))/2
        self.porb_max = porb_max
        
        self.bls_period = None
        self.prot = None
        
        self.lc_tmask = None
        self.lc_rmask = None
        
        self.dur1 = None
        self.dur2 = None
        
        self.ecosw = None
        self.esinw = None
        self.ecc = None
        
        self.pname = [r"$a_1$", r"$t_1$", r"$\sigma_1$", r"$a_2$", r"$t_2$", r"$\sigma_2$", r"$P_{orb}$"]
        

    def stitch(self, lc_collection=None):
        """
        Stitches together lightcurve collection into single lightcurve. Gives modulo of time to handle inconsistent BJD time in TESS lightcurves.
        """
        if lc_collection == None:
            lc_collection = self.lc_collection
            
        self.nsector = len(self.lc_collection)
        for ii in range(self.nsector):
            self.lc_collection[ii].time = self.lc_collection[ii].time.jd
            self.lc_collection[ii].flux = (self.lc_collection[ii].flux / np.nanmedian(self.lc_collection[ii].flux)).value
            self.lc_collection[ii].flux_err = (self.lc_collection[ii].flux_err / np.nanmedian(self.lc_collection[ii].flux)).value

        return self.lc_collection.stitch()


    def estimate_period(self, dur_est=None, method='bls'):
        """
        Estimates period as a starting point for searching for orbital period.
        """
        if dur_est is None:
            dur_est = 1
        bls = BoxLeastSquares(self.lc_flat.time, self.lc_flat.flux, dy=self.lc_flat.flux_err)
        results = bls.autopower(dur_est*u.day)
        period = results.period.value
        power = results.power.value
        ind = np.where((period > 2/60/24) & (period < self.porb_max))[0]
        per_cut = period[ind]
        pow_cut = power[ind]
        self.bls_period = per_cut[np.argmax(pow_cut)]
        return self.bls_period


    def get_arrays(self, lc):
        """
        Returns lightcurve data as separate arrays.
        """
        x = lc.time.value
        y = lc.flux.value
        s = lc.flux_err.value
        
        return x, y, s


    def get_folded(self, porb):
        """
        Returns folded lightcurve arrays.
        """
        self.lc_fold = self.lc_flat.fold(period=porb)

        x, y, s = self.get_arrays(self.lc_fold)

        return x, y, s


    def model(self, theta):
        """
        Model of eclipsing binary lightcurves.
        """
        
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
        """
        Calculates chi-squared of model fit.
        """
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
        
        if self.bls_period is None:
            self.init_optimizer()
            
        if porb_bounds is None:
            porb_bounds = [0, self.porb_max]
        
        # initial guess 
        if t0 is None:
            t0 = self.t0

        # optimize model fit
        res = minimize(self.chisq, t0, method=method, args=(porb_bounds))
        self.res = res

        self.sol = res.x
        self.chi_fit = res.fun

        return self.sol, self.chi_fit
    
    def fit_model_period(self, period_guesses=None, method='nelder-mead', dur_est=0.05):
        """
        Fits transits over a range of periods
        """
        
        if self.bls_period is None:
            self.init_optimizer()
            
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
        """
        Fits transits over both a range of periods and a range of (flattening) window paramters
        """
        
        if self.bls_period is None:
            self.init_optimizer()
        
        if windows is None:
            windows = [21, 51, 81]
        
        solutions = []
        chis = []
        ts = []
        window_guesses = []
        
        for window in windows:      
            self.lc_flat = self.lc_raw.flatten(window_length=window)
            self.init_optimizer()
            self.fit_model()
            
            bls_period = self.bls_period
        
            if period_guesses is None:
                period_guesses = [bls_period/4, bls_period/2, bls_period, bls_period*2]
            
            for bls_period in period_guesses:
                
                # Need list of window values corresponding to each chi2 value
                window_guesses.append(window)
                
                # initialize model solution guess
                t0 = self.init_optimizer(dur_est=dur_est, porb_est=bls_period)

                # optimize model fit
                sol, chi = self.fit_model(t0=t0, method=method)
                solutions.append(sol)
                chis.append(chi)
                ts.append(t0)
                
        self.window = window_guesses[np.nanargmin(chis)]
        
        # Fold lightcurve again with best value for plotting function
        self.lc_flat = self.lc_raw.flatten(window_length=self.window)
        
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
        """
        
        if sol is None:
            sol = self.sol
            
        if (self.dur1 is None) or (self.dur2 is None):
            self.est_duration()
        
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


    def apply_transit_mask(self, sol=None, sigma_clip=3, remove_outliers=False):
        """
        returns transit and rotation masks, given a solution array (a1, t1, d1, a2, t2, d2, porb)
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
        self.lc_tmask = lightkurve.LightCurve(time=t_tmask, flux=f_tmask, flux_err=e_tmask)
        
        if remove_outliers:
            self.lc_rmask = self.lc_rmask.remove_outliers(sigma=sigma_clip)
            self.lc_tmask = self.lc_tmask.remove_outliers(sigma=sigma_clip)
        
        self.lc_rmask.meta = self.meta
        self.lc_tmask.meta = self.meta
    
        return self.tmask, self.rmask


    def save_masked_lcs(self, file_path=None):
        """
        some function which takes self.tmask and self.rmask, 
        applies them to self.lc_raw, and saves the masked arrays to 
        a numpy file
        """
        
        ID = self.ID[3:] 
        
        if file_path is None:
            file_path = './saved_lightcurves/'
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        np.save(file_path + f'TIC_{ID}_rmasked', self.lc_rmask_array)
        np.save(file_path + f'TIC_{ID}_tmasked', self.lc_tmask_array)

        
    def fit_rotation(self, time=None, flux=None, yerr=None, min_period=0.1, max_period=None, oversample=2.0, smooth=0.5, max_peaks=10, prominence=0.01, method='acf'):
        """
        Fits rotation period to lightcurve with transits masked.
        """
        
        rmask = self.lc_rmask
        
        
        if time is None:
            time = rmask.time
            
        if flux is None:
            flux = rmask.flux
            
        
        try:
            x = time.value
            y = flux.value
            x_new = x[np.isfinite(x) & np.isfinite(y)]
            y_new = y[np.isfinite(x) & np.isfinite(y)]

            if method =='acf':
                autocorr = exoplanet.autocorr_estimator(x_new, y_new, yerr=yerr, min_period=min_period, max_period=max_period, oversample=oversample, smooth=smooth, max_peaks=max_peaks)
                lag, acf = autocorr['autocorr']
                peaks, prominence = find_peaks(acf, prominence=prominence)
                peaks, prominence

                p = np.argsort(lag[peaks])
                new_array = lag[peaks][p]

                if (prominence['prominences'][p][1] > prominence['prominences'][p][0]):
                    high_peak = lag[peaks][p][1]
                else:
                    high_peak = lag[peaks][p][0]


            if method =='ls':
                    periodogram = rmask.to_periodogram(method='ls')
                    high_peak = periodogram.period_at_max_power.value
            self.prot = high_peak

        except:
            return -1
        
    def model_fit_summary(self):
        """
        returns a dictionary of all the measured parameters
        """

        summary = {"ID": self.ID,
                   "window": self.window,
                   "dur1": self.dur1,
                   "dur2": self.dur2, 
                   "ecosw": self.ecosw,
                   "esinw": self.esinw,
                   "ecc": self.ecc,
                   "$P_{rot}$": self.prot}

        for ii, param in enumerate(self.pname):
            summary[param] = self.sol[ii] 

        return summary
    
    

    def plot_best_fit(self, 
                      figsize=None, 
                      t0=None, 
                      sol=None, 
                      show=True, 
                      save_dir=None, 
                      save_name=None,
                      tlim=None,
                      label_font=25):
        """
        4-panel plot with raw lightcurve, flattened lc, model fit, and masked lc

        If a save directory is provided, saves figure as "/<save_dir>/ID.png"
        """
        
        if t0 is None:
            t0 = self.t0
        if sol is None:
            sol = self.sol
        if tlim is None:
            tlim = [np.min(self.lc_raw.time.value), np.max(self.lc_raw.time.value)]
        
        self.get_folded(sol[6])       
        tfold = self.lc_fold.time.value
        ffold = self.lc_fold.flux.value
        
        lc_init, _, _ = self.model(t0)
        lc_model, _, _ = self.model(sol)
        
        if figsize is None:
            figsize = (16,30)
        
        # Add an extra plot if the rotation period has been found
        plot_number = 4
        if self.prot is not None:
            plot_number = 5
            
        fig, ax = plt.subplots(plot_number, 1, figsize=figsize)
        
        ax[0].set_title(self.ID, fontsize=label_font+5)
        ax[0].scatter(self.lc_raw.time.value, self.lc_raw.flux.value, s=1, c="k", label=f"{self.nsector} sectors")
        ax[0].set_xlim(min(tlim), max(tlim))
        ax[0].set_xlabel("Time [d]", fontsize=label_font)
        ax[0].legend(loc="upper right", fontsize=label_font)
        ax[0].minorticks_on()

        ax[1].scatter(self.lc_flat.time.value, self.lc_flat.flux.value, s=1, c="k")
        ax[1].set_xlim(min(tlim), max(tlim))
        ax[1].set_xlabel("Time [d]", fontsize=label_font)
        ax[1].minorticks_on()
        
        label = ''
        for ii in range(len(self.pname)):
            label += f"{self.pname[ii]}={np.round(sol[ii],3)}\n"
        try:    
            label += r"$\chi^2=%s$"%(int(np.round(self.chi_fit)))
        except:
            label += r"$\chi^2=inf$"
            
        ax[2].scatter(tfold, ffold, color='k', s=1)
        ax[2].plot(tfold, lc_init, color='b')
        ax[2].plot(tfold, lc_model, color='r', label=label)
        ax[2].axvline(sol[1], linestyle='--')
        ax[2].axvline(sol[4], linestyle='--')
        ax[2].axvspan(sol[1]-sol[2]/2, sol[1]+sol[2]/2, alpha=.1)
        ax[2].axvspan(sol[4]-sol[5]/2, sol[4]+sol[5]/2, alpha=.1)
        ax[2].axvspan(sol[1]-self.dur1/2, sol[1]+self.dur1/2, alpha=.1)
        ax[2].axvspan(sol[4]-self.dur2/2, sol[4]+self.dur2/2, alpha=.1)
        ax[2].set_xlim(min(tfold), max(tfold))
        ax[2].set_xlabel("Orbital Phase [d]", fontsize=label_font)
        ax[2].legend(loc='best', fontsize=22, frameon=False)
        ax[2].minorticks_on()
        
        ax[3].scatter(self.lc_rmask.time.value, self.lc_rmask.flux.value, s=1, c="k")
        ax[3].scatter(self.lc_tmask.time.value, self.lc_tmask.flux.value, s=1, c="r")
        ax[3].set_xlim(min(tlim), max(tlim))
        ax[3].set_xlabel("Time [d]", fontsize=label_font)
        ax[3].minorticks_on()
        
        if self.prot is not None:
            prot_phase = self.lc_rmask.time.value % self.prot
            ax[4].scatter(prot_phase, self.lc_rmask.flux, s=1, c='k')
            ax[4].set_xlim(min(prot_phase), max(prot_phase))
            ax[4].set_xlabel("Rotational Phase [d]", fontsize=label_font)
            ax[4].minorticks_on()
        
        if save_name is None:
            save_name = f'{self.ID}.png'
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_dir + save_name)

        
        if show:
            plt.show()
        else:
            plt.close()