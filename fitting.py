from multiprocessing import Pool
import os
import pandas as pd
from transit_model import TransitModel
import warnings
warnings.simplefilter('ignore')

os.nice(10)

windows =[]

for i in range(1, 11):
    windows.append((i*10) + 1)
    

    
def fitting(KICID):
    ID = 'KIC ' + str(KICID)
    try:
        print('FITTING', ID)
        tm = TransitModel(ID)
        tm.init_optimizer()
        tm.fit_model()
        bls_period = tm.bls_period
        period_guesses = [bls_period/4, bls_period/2, bls_period, bls_period*2]
        tm.fit_model_window(period_guesses=period_guesses, windows=windows)
        tm.est_duration()
        tm.est_eccentricity()
        tm.apply_transit_mask()
        tm.save_masked_lcs()
        tm.plot_best_fit(show=False, save_dir='./Plots/')
        print(ID, 'SUCCESS:', tm.res.success)

        return tm.model_fit_summary()
    
    except:
        print('Error:', ID)

df = pd.read_csv('lurie_ebs.csv')
df = df[df['Porb'] <= 15]
KICIDS = df['KIC']


if __name__ == '__main__':
    pool = Pool(20) 
    res = pool.map(fitting, KICIDS)
    not_none = []
    for val in res:
        if val != None :
            not_none.append(val)
    res_df = pd.DataFrame(not_none)
    res_df.to_csv('res.csv', index=False)