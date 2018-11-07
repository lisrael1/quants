#!/usr/bin/python3
import pandas as pd
import numpy as np
import itertools
from optparse import OptionParser
import pathos.multiprocessing as mp
from tqdm import tqdm
import time, subprocess
from sys import platform
from tqdm import tqdm
tqdm.pandas()

pd.set_option('expand_frame_repr', False)
'''
this script is: - TODO update this...
'''


def get_all_a_rows(a_max_num):
    comb = list(range(-a_max_num, a_max_num + 1))
    '''we dont want 0,0. we want only 0,1 not 0,-1 not 0,2 etc.'''
    comb.remove(0)
    comb = [comb, comb]
    all_rows = list(itertools.product(*comb))
    '''we removed all starting with 0, but we want 0,1. we dont want 0,2 because we already have 0,1'''
    all_rows += [(0, 1)]
    all_rows = pd.DataFrame(all_rows, columns=['a', 'b'])
    '''now removing dependencies by normalizing all first numbers to 1 and removing duplicates'''
    all_rows['normal_b'] = all_rows.b / all_rows.a
    all_rows['abs_a'] = np.abs(all_rows.a)
    all_rows.sort_values(by='abs_a', inplace=True)  # sorting for taking 1,2 and not 2,4
    all_rows.drop_duplicates(subset='normal_b', keep='first', inplace=True)
    '''now just set them as list of matrix'''
    all_rows_in_list = [np.mat(i).reshape(1, 2) for i in all_rows[['a', 'b']].values]
    # print("we have %0d rows for A" % len(all_rows))
    return np.mat(all_rows[['a', 'b']].values)


def rand_cov_det(dimensions=2, snr=1000):
    cov = np.matrix(np.random.normal(0, 1, [1, dimensions]))
    cov = cov.T * cov
    cov+=np.matrix(np.diag([1/snr]*dimensions))
    return cov


def random_data(cov, samples):
    xy = pd.DataFrame(np.random.multivariate_normal([0, 0], cov, samples), columns=['X', 'Y'])
    return xy


def visualizing_cov_case(cov):
    import pylab as plt
    df_original_multi_samples = random_data(cov, 100000)
    df_original_multi_samples.plot.scatter(x='X', y='Y', alpha=0.01)
    m = df_original_multi_samples.max().max()
    plt.axis([-m, m, -m, m])
    plt.show()


def visualize_data(data):
    import pylab as plt
    data.plot.scatter(x='X', y='Y', alpha=0.2, color='red')
    plt.show()


def visualize_data_defore_after(before, after):
    import pylab as plt
    ax = before.plot.scatter(x='X', y='Y', alpha=0.4, color='blue')
    after.plot.scatter(x='X', y='Y', alpha=0.4, color='red', ax=ax)
    plt.show()


def sign_mod(xy, modulo_size_edge_to_edge):
    xy=xy.copy()
    xy += modulo_size_edge_to_edge / 2.0
    xy = xy%modulo_size_edge_to_edge - modulo_size_edge_to_edge / 2.0
    return xy


def to_codebook(df, quantizer_size, number_of_bins=False):
    '''
    for example:
        df=pd.DataFrame([0.6,1,-0.3,-0.29,0.31,0.55])
        a=to_codebook(df,0.2,)
    :param df:
    :param quantizer_size:
    :param number_of_bins: for example, if you want 4 levels 2 bits, with bin size of 0.2, modulo size will be 0.2*4 because 4 level have 3 gaps and
            you add another one for the margins
            0 is for endless quantizer without modulo
    :return: we add offset of half modulo_size so you should clip or modulo from 0 to modulo_size
    '''
    if number_of_bins:
        modulo_size = int(number_of_bins) * quantizer_size
        df = df.copy() + (modulo_size / 2)
    return (df / quantizer_size).round(0).astype(int)


def from_codebook(df, quantizer_size, number_of_bins=False):
    '''
    example:
        quant_size=0.1
        bins=8
        df = pd.DataFrame([0.1,0.2,0.6, 1, -0.3, -0.29, 0.31, 0.55]).T
        a = to_codebook(df, quant_size,bins)
        print(a)
        a%=bins
        print(a)
        print(from_codebook(a,quant_size,bins))
    :param df:
    :param quantizer_size:
    :param number_of_bins:
    :return:
    '''
    offset = quantizer_size * int(number_of_bins) / 2
    return df * quantizer_size - offset


def expected_mse_from_quantization(quant_size):
    return quant_size ** 2 / 12


def clipping_method(samples, quant_size, number_of_bins, std_threshold=None, A_rows=None, snr=1000): # TODO maybe we can return nan when we have more than given mse instead of std_threadold. or maybe the TX always transmite so the std checker is after the cutting
    '''
    for quants with cutting high values to max quant value
    example:
        quant_size = 0.2
        cov = rand_cov_det_1()
        data = random_data(cov, 1000)
        mse = clipping_method(data, quant_size, 100)
        print('mse = %g' % mse)
        print('uniform mse should be %g' % (quant_size ** 2 / 12)) # when all data is inside the module, you should get mse like uniform mse
    :param data:
    :param quant_size:
    :param number_of_bins: int
    :return:
    '''
    number_of_bins = int(number_of_bins)
    cov=rand_cov_det(snr=snr)
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    original = pd.DataFrame(random_data(cov,samples).values.flatten())
    q = to_codebook(original, quant_size, number_of_bins)

    '''now cutting the edges:'''
    q[q > number_of_bins] = number_of_bins
    q[q < 0] = 0
    if std_threshold and (q.std()>std_threshold).astype(int).sum()>0:
        return float(q.std()),original.values.flatten().var()
        # return np.nan
    o = from_codebook(q, quant_size, number_of_bins)
    return float(q.std()),(o - original).values.flatten().var(),((o-original).abs().values.flatten()>quant_size).astype(int).mean(),pearson


def basic_method(samples,quant_size, number_of_bins=False, A_rows=None, snr=1000):
    '''
    for endless quants without modulo
    example:
        quant_size=0.02
        cov=rand_cov_det_1()
        data=random_data(cov,1000)
        mse=basic_method(data,quant_size)
        print('mse = %g' % mse)
        print('uniform mse should be %g'%(quant_size**2/12))

    :param data:
    :param quant_size:
    :return:
    '''
    original = random_data(rand_cov_det(snr=snr),samples)#data.copy()
    q = to_codebook(original, quant_size, 0)
    o = from_codebook(q, quant_size, 0)
    return 0,(o - original).values.flatten().var(),((o-original).abs().values.flatten()>quant_size).astype(int).mean()


def modulo_method(samples, quant_size, number_of_bins, std_threshold=None, A_rows=None, snr=1000):
    '''
    for modulo method
    example:
        quant_size = 0.01
        number_of_bins = 1001
        cov = rand_cov_det_1()
        data = random_data(cov, 1000)
        mse = modulo_method(data, quant_size, number_of_bins)
        print('mse = %g' % mse)
        print('uniform mse should be %g' % (quant_size ** 2 / 12))
    :param data:
    :param quant_size:
    :param number_of_bins:
    :return:
    '''
    cov=rand_cov_det(snr=snr)
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    original = random_data(cov,samples) #data.copy()
    q = to_codebook(original, quant_size, 0)
    # q = to_codebook(original, quant_size, number_of_bins)
    # m1 = q % number_of_bins
    m1 = sign_mod(q,number_of_bins)
    '''finding best rows by sampled std'''
    a_rows = A_rows#get_all_a_rows(15)
    # best_std=m1.dot(a_rows.T).std().sort_values().head(2)
    best_std=m1.dot(a_rows.T).std().nsmallest(2)
    if std_threshold and (best_std>std_threshold).astype(int).sum()>0:
        return best_std.max(),original.values.flatten().var()
        # return np.nan
    best_inx = best_std.index.values
    A = a_rows[best_inx].T
    # print('A det is : %g'%np.linalg.det(A))

    r1=m1.copy().dot(A)
    r1.columns=['X', 'Y']
    # r2=r1%number_of_bins
    r2=sign_mod(r1,number_of_bins)
    # visualize_data(q)
    r3=r2.copy().dot(A.I)
    r3.columns=['X', 'Y']

    o = from_codebook(r3, quant_size, 0)
    # o = from_codebook(r, quant_size, number_of_bins)
    # visualize_data_defore_after(original, o)
    return best_std.max(),(o - original).values.flatten().var(),((o-original).abs().values.flatten()>quant_size).astype(int).mean(),pearson


if __name__ == '__main__':
    start = time.time()
    help_text='''
    examples:
        seq 0 40 |xargs -I ^ echo python3 "2.\ run\ simulation\ cases.py" --number_of_splits 40 --number_of_split ^ \&
        sbatch --mem=1800m -c1 --time=0:50:0 --array=0-399 --wrap "python3 2.\ run\ simulation\ cases.py --number_of_splits \$SLURM_ARRAY_TASK_COUNT --number_of_split \$SLURM_ARRAY_TASK_ID"
    '''
    parser = OptionParser(usage=help_text, version="%prog 1.0 beta")
    parser.add_option("-n", dest="samples", type="int", default=100, help='number of dots X2 because you have x and y. for example 1000. you better use 5 [default: %default]')
    parser.add_option("-s", dest="number_of_split", type="int", default=0, help='the split number from all splits [default: %default]')
    parser.add_option("-m", dest="A_max_num", type="int", default=15,help='A max number for example for 2 you can get [[-2,1],[2,0]]. for number 10, you will get 189,776 options at A. at 5 you will have 13608. . you better use 10 [default: %default]')
    parser.add_option("-p", dest="run_serial", action='store_false', help="dont run parallel. disables when you have splits [default: %default]",default=True)
    parser.add_option("-b", dest="number_of_bins_range", type="str", default='[13,25]', help='number of bins, for example [3,25] [default: %default]')
    parser.add_option("-q", dest="quant_size_range", type="str", default='[0,3.3]', help='number of bins, for example [0,2] [default: %default]')
    number_of_quant_size = 50
    (u, args) = parser.parse_args()

    A_rows=get_all_a_rows(u.A_max_num)

    quant_size = np.linspace(eval(u.quant_size_range)[0], eval(u.quant_size_range)[1], number_of_quant_size + 1)[1:]
    number_of_bins = range(eval(u.number_of_bins_range)[0], eval(u.number_of_bins_range)[1], 2)
    snr_values=[10,100,1000]
    method = ['basic_method', 'modulo_method', 'clipping_method']
    method = ['modulo_method']
    method = ['clipping_method', 'modulo_method']
    inx = pd.MultiIndex.from_product([quant_size, number_of_bins, method, snr_values], names=['quant_size', 'number_of_bins', 'method', 'snr'])
    print('generated %d simulations' % inx.shape[0])
    df = pd.DataFrame(index=inx).reset_index(drop=False)

    print('running the simulations')
    sim_output=df.progress_apply(lambda x: globals()[x.method](samples=u.samples, quant_size=x.quant_size, number_of_bins=x.number_of_bins, snr=x.snr, A_rows=A_rows),axis=1)
    sim_output=sim_output.apply(lambda x: pd.Series(x, index=['sampled_std', 'mse', 'error_per', 'pearson']))
    df=df.join(sim_output)

    print(df.head())
    print(df.describe())
    df.to_csv('results_del_%08d.csv.gz'%u.number_of_split,compression='gzip')

    print(time.time() - start,"sec")