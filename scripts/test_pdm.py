import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=25)
rc('ytick', labelsize=25)
font = {'family' : 'normal',
        'weight' : 'light'}
rc('font', **font)

import sys
sys.path.append("../src")
from inspection import *
from pdm import *


flag_id = 52280468
rtime, rflux, rflux_err = np.load(f"../notebooks/masked_lcs/TIC_{flag_id}_rmasked.npy")
rtime = rtime[~np.isnan(rflux)]
rflux_err = rflux_err[~np.isnan(rflux)]
rflux = rflux[~np.isnan(rflux)]

xp = np.arange(min(rtime), max(rtime), 0.1)
z = np.polyfit(rtime, rflux, 3)
poly = np.poly1d(z)
pflux = rflux - poly(rtime) + np.median(rflux)

csample = pd.read_csv("../notebooks/combined_sample_verified_flags.csv")
flag_sample = csample[csample['flag binary'] == 1]
table_entry = flag_sample[flag_sample['TIC'] == flag_id]

prot_ls = float(table_entry["rotation period inspected"])
plc = PhaseLightCurve(rtime, pflux, rflux_err)
breakpoint()

pdm_best_period, pdm_best_chival = phase_dispersion_minimization(plc, prot_ls, bound=0.1, window=128)