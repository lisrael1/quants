import numpy as np
import pandas as pd
pd.set_option("display.max_columns",1000) # don’t put … instead of multi columns
pd.set_option('expand_frame_repr',False) # for not wrapping columns if you have many

from optparse import OptionParser
from tqdm import tqdm
tqdm.pandas()
from sys import platform


import plotly as py
import cufflinks

import sys
sys.path.append('../../')
import int_force

if __name__ == '__main__':
    timer=int_force.global_imports.timer()
    help_text='''
    examples:
        seq 0 40 |xargs -I ^ echo python3 "%prog" -s ^ \&
        sbatch --mem=1800m -c1 --time=0:50:0 --array=0-399 --wrap 'python3 %prog -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
        sbatch --mem=1800m -c1 --time=0:50:0 --array=0-199 --wrap 'python3 %prog -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} -q "[0,3,150]" -b "[5,17,19,10001]" -m 15'
    '''
    parser = OptionParser(usage=help_text, version="%prog 1.0 beta")
    parser.add_option("-n", dest="samples", type="int", default=100, help='number of dots X2 because you have x and y. for example 1000. you better use 5 [default: %default]')
    parser.add_option("-s", dest="split_id", type="str", default='0', help='the split unique id so it will not override old output [default: %default]')
    parser.add_option("-a", dest="A_max_num", type="int", default=10,help='A max number for example for 2 you can get [[-2,1],[2,0]]. for number 10, you will get 189,776 options at A. at 5 you will have 13608. . you better use 10 [default: %default]')
    parser.add_option("-m", dest="multiply_simulations", type="int", default=1, help='multiply those simulations n times [default: %default]')
    parser.add_option("-b", dest="number_of_bins_list", type="str", default='[5,17,19,101]', help='number of bins, for example [3,,13,25] [default: %default]')
    parser.add_option("-q", dest="quant_size_linspace_params", type="str", default='[0,3.3,150]', help='quant size np.linspace args, for example [0,3.3,10] will be np.linspace(*[0,3.3,10] ) [default: %default]')
    parser.add_option("--a_matrix", dest="A", type="str", default='', help='A Matrix, for example [[1,0],[-2,1]] [default: %default]')
    parser.add_option("--methods", dest="methods", type="str", default='''['clipping_method', 'modulo_method', 'ml_map_method', 'ml_radon_method']''', help='''[default: %default]''')
    parser.add_option("--snr", dest="snr", type="str", default='[10,100,1000]', help='[default: %default]')

    # parser.add_option("--methods", dest="methods", type="str", default='''['ml_method']''', help='''[default: %default]''')
    # parser.add_option("-m", dest="multiply_simulations", type="int", default=10,help='multiply those simulations n times [default: %default]')
    # parser.add_option("-q", dest="quant_size_linspace_params", type="str", default='[0,2.3,10]', help='quant size np.linspace args, for example [0,3.3,10] will be np.linspace(*[0,3.3,10] ) [default: %default]')
    (u, args) = parser.parse_args()

    if len(u.A):
        A=np.mat(eval(u.A))
        print('input cov is ')
        print(int_force.rand_data.rand_data.rand_cov(snr=None, A=A))
    else:
        A=None
    modulo_method=int_force.methods.modulo.modulo_method
    ml_radon_method=int_force.methods.ml_modulo.ml_radon_method
    clipping_method=int_force.methods.clipping.clipping_method
    ml_map_method=int_force.methods.ml_modulo.ml_map_method

    quant_size = np.linspace(*eval(u.quant_size_linspace_params))[1:]
    method=eval(u.methods)
    if "win" in platform:
        quant_size=quant_size[::15]
        # method=['ml_method']
    number_of_bins = eval(u.number_of_bins_list)
    snr_values=eval(u.snr)
    inx = pd.MultiIndex.from_product([quant_size, number_of_bins, method, snr_values], names=['quant_size', 'number_of_bins', 'method', 'snr'])
    print('generated %d simulations' % inx.shape[0])
    df = pd.DataFrame(index=inx).reset_index(drop=False)

    df = pd.concat([df] * u.multiply_simulations, ignore_index=True)

    print('running the simulations')
    sim_output=df.progress_apply(lambda x: globals()[x.method](samples=u.samples, quant_size=x.quant_size, number_of_bins=x.number_of_bins, snr=x.snr, A=A),axis=1)
    # sim_output=sim_output.apply(lambda x: pd.Series(x, index=['sampled_std', 'mse', 'error_per', 'pearson']))
    sim_output=sim_output.apply(pd.Series)
    df=df.join(sim_output)

    print(df.head())
    print(df.describe())
    df.to_csv('results_del_%s.csv.gz'%u.split_id,compression='gzip')
    if 0:
        print('%d uniques A that the script found'%df.A.dropna().astype(str).nunique())
        print(df.A.dropna().astype(str).value_counts())

        fig = df.pivot_table(index='quant_size', columns=['method', 'number_of_bins'], values='rmse').figure(yTitle='rmse', xTitle='quant size')
        py.offline.plot(fig)

