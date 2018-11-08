#!/usr/bin/python3
import pandas as pd
import numpy as np
import itertools
from optparse import OptionParser
import pathos.multiprocessing as mp
from tqdm import tqdm
import time, subprocess
from sys import platform


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


def rand_cov_det_1(dimensions=2):
    cov = np.matrix(np.random.normal(0, 1, [dimensions, dimensions]))
    cov = cov.T * cov
    cov /= np.power(np.linalg.det(cov), 1 / dimensions)  # for 2X2 we use sqrt because we have 2 dimensions and we have a*c-b*d
    return cov


def rand_cov_det_2(dimensions=2,noise_std=0.01):
    cov = np.matrix(np.random.normal(0, 1, [1, dimensions]))
    cov = cov.T * cov
    # cov+=np.matrix(np.diag(np.abs(np.random.normal(0, noise_std, dimensions))))
    cov+=np.matrix(np.diag([np.random.choice([1/10,1/100,1/1000])]*dimensions))
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


def naive_method(samples, quant_size, number_of_bins, std_threshold=3, A_rows=None): # TODO maybe we can return nan when we have more than given mse instead of std_threadold. or maybe the TX always transmite so the std checker is after the cutting
    '''
    for quants with cutting high values to max quant value
    example:
        quant_size = 0.2
        cov = rand_cov_det_1()
        data = random_data(cov, 1000)
        mse = naive_method(data, quant_size, 100)
        print('mse = %g' % mse)
        print('uniform mse should be %g' % (quant_size ** 2 / 12)) # when all data is inside the module, you should get mse like uniform mse
    :param data:
    :param quant_size:
    :param number_of_bins: int
    :return:
    '''
    number_of_bins = int(number_of_bins)
    cov=rand_cov_det_2()
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    original = pd.DataFrame(random_data(cov,samples).values.flatten())
    q = to_codebook(original, quant_size, number_of_bins)

    '''now cutting the edges:'''
    q[q > number_of_bins] = number_of_bins
    q[q < 0] = 0
    if (q.std()>std_threshold).astype(int).sum()>0:
        return float(q.std()),original.values.flatten().var()
        # return np.nan
    o = from_codebook(q, quant_size, number_of_bins)
    return float(q.std()),(o - original).values.flatten().var(),((o-original).abs().values.flatten()>quant_size).astype(int).mean(),pearson


def basic_method(samples,quant_size,number_of_bins=False, A_rows=None):
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
    original = random_data(rand_cov_det_2(),samples)#data.copy()
    q = to_codebook(original, quant_size, 0)
    o = from_codebook(q, quant_size, 0)
    return 0,(o - original).values.flatten().var(),((o-original).abs().values.flatten()>quant_size).astype(int).mean()


def modulo_method(samples, quant_size, number_of_bins, std_threshold=3, A_rows=None):
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
    cov=rand_cov_det_2()
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    original = random_data(cov,samples) #data.copy()
    q = to_codebook(original, quant_size, 0)
    # q = to_codebook(original, quant_size, number_of_bins)
    # m1 = q % number_of_bins
    m1 = sign_mod(q,number_of_bins)
    '''finding best rows by sampled std'''
    a_rows = A_rows#get_all_a_rows(15)
    best_std=m1.dot(a_rows.T).std().sort_values().head(2)
    if (best_std>std_threshold).astype(int).sum()>0:
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
    parser.add_option("-n", "", dest="samples", type="int", default=100, help='number of dots X2 because you have x and y. for example 1000. you better use 5 [default: %default]')
    parser.add_option("--number_of_splits", dest="number_of_splits", type="int", default=1, help='you have 1k cases, and you want to run them with x splits [default: %default]')
    parser.add_option("--number_of_split", dest="number_of_split", type="int", default=0, help='the split number from all splits [default: %default]')
    # parser.add_option("-s", "", dest="simulations", type="int", default=200, help='number of simulations, for example 50. you better use 400') # about 8360 sims per unit here
    # parser.add_option("-s", "", dest="simulations", type="int", default=20000, help='number of simulations, for example 50. you better use 400') # about 8360 sims per unit here
    # parser.add_option("-b", "", dest="number_of_bins_range", type="str", default='[3,25]', help='number of bins, for example [3,25]')
    # parser.add_option("-b", "", dest="number_of_bins_range", type="str", default='[23,25]', help='number of bins, for example [3,25]')
    # parser.add_option("-q", "", dest="quant_size_range", type="str", default='[0,2]', help='number of bins, for example [0,2]')
    # parser.add_option("-q", "", dest="quant_size_range", type="str", default='[0.39,0.4]', help='number of bins, for example [0,2]')
    # number_of_quant_size=1#20
    # parser.add_option("-t", "", dest="std_threshold_range", type="str", default='[0.6,3]', help='number of bins, for example [0,2]')
    # number_of_thresholds=20
    parser.add_option("-m", "", dest="A_max_num", type="int", default=15,help='A max number for example for 2 you can get [[-2,1],[2,0]]. for number 10, you will get 189,776 options at A. at 5 you will have 13608. . you better use 10 [default: %default]')
    parser.add_option("-p", dest="run_serial", action='store_false', help="dont run parallel. disables when you have splits [default: %default]",default=True)
    (u, args) = parser.parse_args()

    A_rows=get_all_a_rows(u.A_max_num)

    '''420,000 rows took me 1645 sec'''
    if "win" in platform:
        df = pd.read_csv('simulation_cases.csv.gz',index_col=[0]).reset_index(drop=True)
        print('taking part of the simulation for this split')
        df=df.iloc[np.array_split(df.index.values,u.number_of_splits)[u.number_of_split]]
    else:
        print('checking file length')
        process = subprocess.Popen("pigz -dc simulation_cases.csv.gz|wc -l", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        lines = int(process.stdout.read().strip())
        c = np.arange(lines - 1)
        print('splitting file index')
        b = np.array_split(c, u.number_of_splits)[u.number_of_split]
        print('reading file')
        df = pd.read_csv('simulation_cases.csv.gz', index_col=[0], nrows=len(b),skiprows=np.arange(1, b[0]+1)).reset_index(drop=True)

    df.replace('mse_from_','',regex=True,inplace=True)
    print('done reading simulation cases')

    df['sampled_std']=None
    df['mse']=None
    df['error_per']=None
    df['pearson']=None

    if 0:
        '''removing non interesting cases'''
        df=df[(df.quant_size*df.number_of_bins<15)&(df.quant_size*df.number_of_bins>6)]
        df=df[df.quant_size*df.number_of_bins>8.9]
        print('dropping non interesting cases - running only %d simulations'%df.shape[0])
        df.reset_index(drop=True,inplace=True)
    if not u.run_serial and u.number_of_splits==1:
        '''multi cores'''
        num_of_cpu = mp.cpu_count()
        if 0:
            print('disabling parallel work!!!!!')
            num_of_cpu = 1
        print('number of cpus: %d' % num_of_cpu)
        p = mp.Pool(num_of_cpu)
        args = [dict(df=df.iloc[inx], A_rows=A_rows, samples=u.samples) for inx in np.array_split(range(df.shape[0]), num_of_cpu)]
        df[['sampled_std','mse','error_per','pearson']] = pd.concat(p.map(lambda y: y['df'].apply(lambda x:pd.Series(globals()[x.method](samples=y['samples'] ,quant_size=x.quant_size, std_threshold=x.std_threshold,number_of_bins=x.number_of_bins,A_rows=y['A_rows']),index=['sampled_std','mse','error_per','pearson']),axis=1),args))
    else:
        for inx in tqdm(np.array_split(range(df.shape[0]),100)):
            df.loc[inx+df.index.values[0],['sampled_std','mse','error_per','pearson']]=df.iloc[inx].apply(lambda x:pd.Series(globals()[x.method](samples=u.samples ,quant_size=x.quant_size,number_of_bins=x.number_of_bins, std_threshold=x.std_threshold,A_rows=A_rows),index=['sampled_std','mse','error_per','pearson']),axis=1)
    print(df.head())
    df.to_csv('resutls_%08d.csv.gz'%u.number_of_split,compression='gzip')
    # df.dropna(how='any',inplace=True)
    # res=df.pivot_table(values='mse',columns=['number_of_bins','method'],index=['quant_size'],aggfunc=[np.mean,np.median])
    # res.to_csv('7.2 resutls.csv')

    print(time.time() - start,"sec")