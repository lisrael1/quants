#!/usr/bin/python3
import pandas as pd
import numpy as np
import itertools
from optparse import OptionParser
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
    if 0:
        print('{s} ERROR - remove independent rows! {s}'.format(s='*'*10))
        comb = [comb, comb]
        all_rows = list(itertools.product(*comb))
        all_rows.remove((0, 0))
        return np.mat(all_rows)
    '''we dont want 0,0. we want only 0,1 not 0,-1 not 0,2 etc.'''
    comb.remove(0)
    comb = [comb, comb]
    all_rows = list(itertools.product(*comb))
    '''we removed all starting with 0, but we want 0,1. we dont want 0,2 because we already have 0,1'''
    all_rows += [(0, 1)]
    all_rows += [(1, 0)]
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


def rand_cov(dimensions=2, snr=1000):
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
    max_num=7.5
    data.plot.scatter(x='X', y='Y', alpha=0.2, color='red', xlim=[-max_num,max_num], ylim=[-max_num,max_num], grid=True)
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
    importand note - this will not work on even number of bins!!!
    for quants with cutting high values to max quant value
    example:
        quant_size = 0.2
        cov = rand_cov_1()
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
    cov=rand_cov(snr=snr)
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    original = pd.DataFrame(random_data(cov,samples).values)#.flatten())
    q = to_codebook(original, quant_size, 0)

    '''now cutting the edges:'''
    q=np.clip(q,np.ceil(-number_of_bins/2),np.floor(number_of_bins/2)).astype(int)
    q_std=q.std()
    if std_threshold and (q_std>std_threshold).astype(int).sum()>0:
        return 'deprecated'
        return float(q.std()),original.values.flatten().var()
    o = from_codebook(q, quant_size, 0)
    res=dict(sampled_std_1=q_std.min(), sampled_std_2=q_std.max(), mse=(o - original).values.flatten().var(), error_per=((o-original).abs().values.flatten()>quant_size).astype(int).mean(), pearson=pearson, cov=str(cov.tolist()))
    return res


def basic_method(samples, quant_size, number_of_bins=False, A_rows=None, snr=1000):
    '''
    for endless quants without modulo
    example:
        quant_size=0.02
        cov=rand_cov_1()
        data=random_data(cov,1000)
        mse=basic_method(data,quant_size)
        print('mse = %g' % mse)
        print('uniform mse should be %g'%(quant_size**2/12))

    :param data:
    :param quant_size:
    :return:
    '''
    cov=rand_cov(snr=snr)
    original = random_data(cov, samples)
    q = to_codebook(original, quant_size, 0)
    o = from_codebook(q, quant_size, 0)
    res=dict(sampled_std_1=0,
             sampled_std_2=0,
             mse=(o - original).values.flatten().var(),
             error_per=((o-original).abs().values.flatten()>quant_size).astype(int).mean(),
             pearson=0,
             cov=str(cov.tolist()))
    return res


def modulo_method(samples, quant_size, number_of_bins, std_threshold=None, A_rows=None, snr=1000):
    '''
    for modulo method
    example:
        quant_size = 0.01
        number_of_bins = 1001
        cov = rand_cov_1()
        data = random_data(cov, 1000)
        mse = modulo_method(data, quant_size, number_of_bins)
        print('mse = %g' % mse)
        print('uniform mse should be %g' % (quant_size ** 2 / 12))
    :param data:
    :param quant_size:
    :param number_of_bins:
    :return:
    '''
    cov=rand_cov(snr=snr)
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    original = random_data(cov,samples) #data.copy()
    q = to_codebook(original, quant_size, 0)
    # q = to_codebook(original, quant_size, number_of_bins)
    # m1 = q % number_of_bins
    m1 = sign_mod(q,number_of_bins).astype(int)
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
    res=dict(sampled_std_1=best_std.min(),
             sampled_std_2=best_std.max(),
             mse=(o - original).values.flatten().var(),
             error_per=((o-original).abs().values.flatten()>quant_size).astype(int).mean(),
             pearson=pearson,
             cov = str(cov.tolist()))
    return res


if __name__ == '__main__':
    start = time.time()
    help_text='''
    examples:
        seq 0 40 |xargs -I ^ echo python3 "%prog" -s ^ \&
        sbatch --mem=1800m -c1 --time=0:50:0 --array=0-399 --wrap 'python3 %prog -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}'
        sbatch --mem=1800m -c1 --time=0:50:0 --array=0-199 --wrap 'python3 %prog -s ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} -q "[0,3,150]" -b "[5,17,19,10001]" -m 15'
    '''
    parser = OptionParser(usage=help_text, version="%prog 1.0 beta")
    parser.add_option("-n", dest="samples", type="int", default=100, help='number of dots X2 because you have x and y. for example 1000. you better use 5 [default: %default]')
    parser.add_option("-s", dest="split_id", type="str", default='0', help='the split unique id so it will not override old output [default: %default]')
    parser.add_option("-a", dest="A_max_num", type="int", default=15,help='A max number for example for 2 you can get [[-2,1],[2,0]]. for number 10, you will get 189,776 options at A. at 5 you will have 13608. . you better use 10 [default: %default]')
    parser.add_option("-m", dest="multiply_simulations", type="int", default=1,help='multiply those simulations n times [default: %default]')
    parser.add_option("-b", dest="number_of_bins_list", type="str", default='[5,17,19,101]', help='number of bins, for example [3,,13,25] [default: %default]')
    parser.add_option("-q", dest="quant_size_linspace_params", type="str", default='[0,3.3,15]', help='quant size np.linspace args, for example [0,3.3,10] will be np.linspace(*[0,3.3,10] ) [default: %default]')
    (u, args) = parser.parse_args()

    A_rows=get_all_a_rows(u.A_max_num)

    quant_size = np.linspace(*eval(u.quant_size_linspace_params))[1:]
    number_of_bins = eval(u.number_of_bins_list)
    snr_values=[10,100,1000]
    method = ['modulo_method']
    method = ['clipping_method', 'modulo_method']
    method = ['basic_method', 'modulo_method', 'clipping_method']
    inx = pd.MultiIndex.from_product([quant_size, number_of_bins, method, snr_values], names=['quant_size', 'number_of_bins', 'method', 'snr'])
    print('generated %d simulations' % inx.shape[0])
    df = pd.DataFrame(index=inx).reset_index(drop=False)

    df = pd.concat([df] * u.multiply_simulations, ignore_index=True)

    print('running the simulations')
    sim_output=df.progress_apply(lambda x: globals()[x.method](samples=u.samples, quant_size=x.quant_size, number_of_bins=x.number_of_bins, snr=x.snr, A_rows=A_rows),axis=1)
    # sim_output=sim_output.apply(lambda x: pd.Series(x, index=['sampled_std', 'mse', 'error_per', 'pearson']))
    sim_output=sim_output.apply(pd.Series)
    df=df.join(sim_output)

    print(df.head())
    print(df.describe())
    df.to_csv('results_del_%s.csv.gz'%u.split_id,compression='gzip')

    print(time.time() - start,"sec")