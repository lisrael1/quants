#!/usr/bin/python3
import pandas as pd
import numpy as np
import itertools
from optparse import OptionParser
from threading import Thread
import pathos.multiprocessing as mp
import pylab as plt
import cufflinks
import plotly as py
from tqdm import tqdm
import time
from threading import Thread


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
    :return:
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


def mse_from_naive_method(data, quant_size, number_of_bins, A_rows=None):
    '''
    for quants with cutting high values to max quant value
    example:
        quant_size = 0.2
        cov = rand_cov_det_1()
        data = random_data(cov, 1000)
        mse = mse_from_naive_method(data, quant_size, 100)
        print('mse = %g' % mse)
        print('uniform mse should be %g' % (quant_size ** 2 / 12)) # when all data is inside the module, you should get mse like uniform mse
    :param data:
    :param quant_size:
    :param number_of_bins: int
    :return:
    '''
    number_of_bins = int(number_of_bins)
    original = pd.DataFrame(data.values.flatten())
    q = to_codebook(original, quant_size, number_of_bins)

    '''now cutting the edges:'''
    q[q > number_of_bins] = number_of_bins
    q[q < 0] = 0
    o = from_codebook(q, quant_size, number_of_bins)
    return (o - original).values.flatten().var()


def mse_from_basic_method(data, quant_size,number_of_bins=False, A_rows=None):
    '''
    for endless quants without modulo
    example:
        quant_size=0.02
        cov=rand_cov_det_1()
        data=random_data(cov,1000)
        mse=mse_from_basic_method(data,quant_size)
        print('mse = %g' % mse)
        print('uniform mse should be %g'%(quant_size**2/12))

    :param data:
    :param quant_size:
    :return:
    '''
    original = data.copy()
    q = to_codebook(original, quant_size, 0)
    o = from_codebook(q, quant_size, 0)
    return (o - original).values.flatten().var()


def mse_from_modulo_method(data, quant_size, number_of_bins, A_rows=None):
    '''
    for modulo method
    example:
        quant_size = 0.01
        number_of_bins = 1001
        cov = rand_cov_det_1()
        data = random_data(cov, 1000)
        mse = mse_from_modulo_method(data, quant_size, number_of_bins)
        print('mse = %g' % mse)
        print('uniform mse should be %g' % (quant_size ** 2 / 12))
    :param data:
    :param quant_size:
    :param number_of_bins:
    :return:
    '''
    original = data.copy()
    q = to_codebook(original, quant_size, 0)
    # q = to_codebook(original, quant_size, number_of_bins)
    # m1 = q % number_of_bins
    m1 = sign_mod(q,number_of_bins)
    '''finding best rows by sampled std'''
    a_rows = A_rows#get_all_a_rows(15)
    best_std=m1.dot(a_rows.T).std().sort_values().head(2)
    if (best_std>3).astype(int).sum()>0:
        return np.nan
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
    return (o - original).values.flatten().var()

if __name__ == '__main__':
    start = time.time()

    '''multi cores'''
    num_of_cpu = mp.cpu_count()
    if 0:
        print('disabling parallel work!!!!!')
        num_of_cpu = 1
    print('number of cpus: %d' % num_of_cpu)
    p = mp.Pool(num_of_cpu)

    '''plot results'''
    if 0:
        res=pd.read_csv('7.2 resutls.csv',header=[0,1,2],index_col=[0])
        df=pd.read_csv('7.2 resutls df.csv',header=[0],index_col=[0])
        print(res.head())
        res['mean'].plot()
        fig=res['median'].iplot(asFigure=True)
        py.offline.plot(fig)
        # plt.show()
        print('hi')
        exit()

    cov = rand_cov_det_1()
    data = random_data(cov, 100)
    A_rows=get_all_a_rows(15)

    '''420,000 rows took me 1645 sec'''
    quant_size=np.linspace(0,1,11)[1:]
    number_of_bins=range(3,25,2)[:-2]
    method=['mse_from_basic_method','mse_from_modulo_method','mse_from_naive_method']
    method=['mse_from_basic_method','mse_from_modulo_method']
    number_of_sims=range(100)[:-4]
    inx = pd.MultiIndex.from_product([quant_size,number_of_bins,method,number_of_sims], names=['quant_size', 'number_of_bins', 'method', 'number_of_sims'])
    df = pd.DataFrame(index=inx).reset_index(drop=False)
    '''removing non interesting cases'''
    print('generated %d simulations'%df.shape[0])
    # df=df[(df.quant_size*df.number_of_bins<15)&(df.quant_size*df.number_of_bins>6)]
    df=df[df.quant_size*df.number_of_bins>8.9]
    df.reset_index(drop=True,inplace=True)
    print('dropping non interesting cases - running only %d simulations'%df.shape[0])

    if 1:
        args = [dict(df=df, inx=inx, A_rows=A_rows) for inx in np.array_split(range(df.shape[0]), num_of_cpu)]
        df['mse'] = pd.concat(p.map(lambda y: y['df'].iloc[y['inx']].apply(lambda x:globals()[x.method](data=random_data(rand_cov_det_1(), 100),quant_size=x.quant_size,number_of_bins=x.number_of_bins,A_rows=y['A_rows']),axis=1),args))
    else:
        for inx in tqdm(np.array_split(range(df.shape[0]),1000)):
            df.loc[inx,'mse']=df.iloc[inx].apply(lambda x:globals()[x.method](data=random_data(rand_cov_det_1(), 100),quant_size=x.quant_size,number_of_bins=x.number_of_bins,A_rows=A_rows),axis=1)
    print(df.head())
    df.dropna(how='any',inplace=True)
    res=df.pivot_table(values='mse',columns=['number_of_bins','method'],index=['quant_size'],aggfunc=[np.mean,np.median])
    res.to_csv('7.2 resutls.csv')
    df.to_csv('7.2 resutls df.csv')

    print(time.time() - start,"sec")


if 0:
        parser = OptionParser()
        parser.add_option("-n", "", dest="samples", type="int", default=10,
                          help='number of dots X2 because you have x and y. for example 1000. you better use 5')
        parser.add_option("-s", "", dest="simulations", type="int", default=10, help='number of simulations, for example 50. you better use 400')
        parser.add_option("-b", "", dest="bins", type="int", default=200, help='number of bins, for example 200')
        parser.add_option("-t", "", dest="threshold", type="float", default=2.5,
                          help='this threshold is for deciding if data is U or N, by checking if there are samples over the threshold. this defines the tail size. bigger number will result of more detecting output as N')
        parser.add_option("-m", "", dest="A_max_num", type="int", default=2,
                          help='A max number for example for 2 you can get [[-2,1],[2,0]]. for number 10, you will get 189,776 options at A. at 5 you will have 13608. . you better use 10')
        parser.add_option("-o", dest="different_cov", help="if set, using same cov matrix for all simulations", default=False, action="store_true")
        (u, args) = parser.parse_args()



