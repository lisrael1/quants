print('starting')
import numpy as np
import pandas as pd
# from sklearn.cluster import DBSCAN
# import pylab as plt
import warnings
from scipy.stats import multivariate_normal
import itertools

pd.set_option("display.max_columns",1000) # don’t put … instead of multi columns
pd.set_option('expand_frame_repr',False) # for not wrapping columns if you have many
pd.set_option("display.max_rows",8)
pd.set_option('display.max_colwidth',1000)

print('done imports')

if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    sys.path.append(r'C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2/')
    # from int_force.rand_data import rand_data
    plot=True

    import int_force
    for i in range(10):
        print('*'*150)

        cov=np.mat([[0, 0],[0, 1]])
        cov=np.mat([[1, 1],[1, 1.2]])
        cov=np.mat([[0.53749846, 0.35644121],[0.35644121, 0.23651739]])
        cov=int_force.rand_data.rand_data.rand_cov(snr=10000)

        number_of_bins=17
        mod_size=1.8
        samples=1000
        quant_size=mod_size/number_of_bins
        snr=100000
        res=int_force.methods.ml_modulo.ml_map_method(samples, quant_size, number_of_bins, snr=snr, plot=False)
        print(res)

        # shifts=int_force.methods.ml_modulo.ml_map(cov, number_of_bins, mod_size, number_of_modulos=5, plots=plot)
        #
        # data=int_force.rand_data.rand_data.random_data(cov, 1000)
        # tmp=int_force.methods.methods.sign_mod(data, mod_size)
        # recovered=int_force.methods.methods.to_codebook(tmp, mod_size/number_of_bins)
        # recovered=int_force.methods.methods.from_codebook(recovered, mod_size/number_of_bins)
        # shifts.index.names=['X','Y']
        # shifts=shifts.reset_index(drop=False)
        # shifts=shifts.sort_values(['X', 'Y']).round(8)
        # recovered=recovered.sort_values(['X', 'Y']).round(8)
        # recovered=pd.merge(recovered, shifts, on=['X', 'Y'], how='left')
        # recovered['new_x']=recovered.X+recovered.x_shift*mod_size
        # recovered['new_y']=recovered.Y+recovered.y_shift*mod_size
        # recovered=recovered[['new_x', 'new_y']]
        #
        # recovered.columns=[['recovered']*2, ['X', 'Y']]
        # tmp.columns=[['after']*2, ['X', 'Y']]
        # data.columns=[['before']*2, ['X', 'Y']]
        #
        # data=data.join(tmp).join(recovered)
        # data=data.stack(0).reset_index(drop=False)
        #
        # if plot:
        #     import plotly as py
        #     import cufflinks
        #     fig=data.figure(kind='scatter', x='X', y='Y', categories='level_1', size=4)
        #     py.offline.plot(fig, auto_open=True, filename='data.html')



