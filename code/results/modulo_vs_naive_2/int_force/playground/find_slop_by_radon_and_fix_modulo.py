import numpy as np
import pandas as pd
import plotly as py
import cufflinks
from sklearn.cluster import DBSCAN
import pylab as plt
from skimage.transform import radon, rescale, iradon, iradon_sart, hough_line
import warnings
from scipy import signal
import itertools


if __name__ == '__main__':
    for i in range(1):
        print('*'*150)
        import sys
        sys.path.append('../../')
        sys.path.append(r'C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2/')
        import int_force
        # from int_force.rand_data import rand_data

        cov=np.mat([[0, 0],[0, 1]])
        cov=np.mat([[1, 1],[1, 1.2]])
        cov=np.mat([[0.53749846, 0.35644121],[0.35644121, 0.23651739]])
        cov=int_force.rand_data.rand_data.rand_cov(snr=10000)

        samples=50
        data=int_force.rand_data.rand_data.random_data(cov, samples)
        quant_size=0.1
        number_of_bins=17
        mod_size=1.8
        tmp=int_force.methods.methods.sign_mod(data, mod_size)
        tmp.columns=[['after']*2,tmp.columns.values]
        data.columns=[['before']*2,data.columns.values]
        data=data.join(tmp)

        int_force.rand_data.rand_data.all_data_origin_options(data.after, mod_size, 5)
        sinogram_dict=int_force.methods.ml_modulo.calc_sinogram(data.after.X.values, data.after.Y.values, bins=600)
        y_per_x_ration=np.tan(sinogram_dict['angle_by_std'])
