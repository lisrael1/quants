import numpy as np
import pandas as pd
# import plotly as py
# import cufflinks
import sys
sys.path.append('../../')
sys.path.append(r'C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2/')
from scipy.optimize import fmin, brent, minimize
import int_force
import pylab as plt
import plotly as py
import cufflinks


pd.set_option("display.max_columns",1000) # don’t put … instead of multi columns
pd.set_option('expand_frame_repr',False) # for not wrapping columns if you have many
pd.set_option("display.max_rows",10)
pd.set_option('display.max_colwidth',1000)
debug=True


def ml_modulo_rmse(quant_size, samples, number_of_bins, snr):
    quant_size=quant_size[0]
    print("%f, " % quant_size, end="")
    rmse=np.mean([int_force.methods.ml_modulo.ml_modulo_method(samples=samples, number_of_bins=number_of_bins, quant_size=quant_size, snr=snr, debug=False)['rmse'] for _ in range(1)])
    print("%f" % rmse)
    return rmse


if __name__ == '__main__':
    cov=np.mat([[0, 0],[0, 1]])
    cov=np.mat([[1, 1],[1, 1.2]])
    cov=np.mat([[0.53749846, 0.35644121],[0.35644121, 0.23651739]])
    samples=1000
    number_of_bins=400
    mod_size=3.8
    quant_size=mod_size/number_of_bins
    snr=10000
    # rmse=ml_modulo_rmse(quant_size, samples, number_of_bins, snr)
    # not working! the output is too noisy, so it stuck at the first value, or some values after
    min2 = minimize(ml_modulo_rmse, x0=[1.0], args=(samples, number_of_bins, snr), bounds=((0, 100), ), constraints=(dict(type='ineq', fun=lambda x: x)), options={'disp': True})


    print('done')

