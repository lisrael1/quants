import numpy as np
import pandas as pd
# import plotly as py
# import cufflinks
import sys, os
sys.path.append('../../')
# sys.path.append(r'C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2/')
root_folder = os.path.realpath(__file__).replace('\\', '/').rsplit('modulo_vs_naive_2', 1)[0]+'/modulo_vs_naive_2/'
sys.path.append(root_folder)
import int_force
import pylab as plt
import plotly as py
import cufflinks

pd.set_option("display.max_columns",1000) # don’t put … instead of multi columns
pd.set_option('expand_frame_repr',False) # for not wrapping columns if you have many
pd.set_option("display.max_rows",10)
pd.set_option('display.max_colwidth',1000)
debug=True

if __name__ == '__main__':
    for i in range(100):
        print('*'*20+'iteration number %d'%i+'*'*20)
        # from int_force.rand_data import rand_data

        cov=np.mat([[0, 0],[0, 1]])
        cov=np.mat([[1, 1],[1, 1.2]])
        cov=np.mat([[0.53749846, 0.35644121],[0.35644121, 0.23651739]])
        samples=1000
        number_of_bins=170
        mod_size=1.8
        quant_size=mod_size/number_of_bins
        snr=10000

        int_force.methods.ml_modulo.ml_modulo_method(samples=samples, number_of_bins=number_of_bins, quant_size=quant_size, snr=snr, debug=True)

        print('done')

