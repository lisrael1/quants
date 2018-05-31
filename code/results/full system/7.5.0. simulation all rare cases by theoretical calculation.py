#!/usr/bin/python3
import pandas as pd
import numpy as np
import itertools
from optparse import OptionParser
import pathos.multiprocessing as mp
# import cufflinks
from tqdm import tqdm
import time
import pylab as plt
from scipy.stats import multivariate_normal

pd.set_option('expand_frame_repr', False)

# def cdf_of_square(multi_variant_dist,start_point,end_point):
def cdf_of_square(multi_variant_dist,x_start,x_end,y_start,y_end):
    # multi_variant_dist type is from scipy.stats import multivariate_normal
    return multi_variant_dist.cdf([x_end,y_end])-multi_variant_dist.cdf([x_start,y_end])-multi_variant_dist.cdf([x_end,y_start])+multi_variant_dist.cdf([x_start,y_start])

def get_samples_from_cov(cov,samples=20):
    # we will take 7 sigma, so we get 20 from 1e9 cases, not more
    multi_variant_dist=multivariate_normal([0,0],cov)
    x_max=np.sqrt(cov[0,0])*7
    y_max=np.sqrt(cov[1,1])*7
    inx = pd.MultiIndex.from_product([np.linspace(-x_max,x_max,samples),np.linspace(-y_max,y_max,samples)], names=['x', 'y'])
    xy = pd.DataFrame(index=inx).reset_index(drop=False)
    # x_jumps=x_max/samples
    # y_jumps=y_max/samples
    x_jumps=(xy.x.unique()[1]-xy.x.unique()[0])/2
    y_jumps=(xy.y.unique()[1]-xy.y.unique()[0])/2
    xy['x_start']=xy.x-x_jumps
    xy['x_end']=xy.x+x_jumps
    xy['y_start']=xy.y-y_jumps
    xy['y_end']=xy.y+y_jumps
    xy['p']=xy.apply(lambda row: cdf_of_square(multi_variant_dist, row.x_start, row.x_end, row.y_start, row.y_end),axis=1)
    # xy['p'] = xy.apply(lambda row: cdf_of_square(multi_variant_dist, row.x - x_jumps, row.x + x_jumps, row.y - y_jumps, row.y + y_jumps),axis=1)
    if xy.p.sum()<0.9999:
        print('here is some issue with the probability calculation')
        print(xy.head())
        print(xy.tail())
    return xy[['x','y','p']]



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

def mse_from_basic_method(data,quant_size,number_of_bins=False, A_rows=None):
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
    return o
    # return (o - original).values.flatten().var()

def mse_from_naive_method(data, quant_size, number_of_bins, std_threshold=3, A_rows=None): # TODO maybe we can return nan when we have more than given mse instead of std_threadold. or maybe the TX always transmite so the std checker is after the cutting
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
    original = data.copy()
    q = to_codebook(original, quant_size, number_of_bins)

    '''now cutting the edges:'''
    q[q > number_of_bins] = number_of_bins
    q[q < 0] = 0
    if (q.std()>std_threshold).astype(int).sum()>0:
        return original*0
        # return original.values.flatten().var()
        # return np.nan
    o = from_codebook(q, quant_size, number_of_bins)
    return o
    # return (o - original).values.flatten().var()

def mse_from_modulo_method(data, quant_size, number_of_bins, std_threshold=3, A_rows=None):
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
    if (best_std>std_threshold).astype(int).sum()>0:
        return original*0
        # return original.values.flatten().var()
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
    return o
    # return (o - original).values.flatten().var()


cov=np.mat([[1,0.5],[0.5,2]])
all_a_rows=get_all_a_rows(10)
xy=get_samples_from_cov(cov,20)
TODO the problem here is that the next function look for std but we just give it the whole range
and even if you check std by 1 group of samples, it will stil be 1 group and not the real world
outputs=mse_from_modulo_method(xy[['x','y']],0.01,1000,40000,all_a_rows).rename(columns=dict(X='x_out',Y='y_out'))
xy=pd.concat([xy,outputs],axis=1)
xy['x_se']=(xy.x-xy.x_out).pow(2)*xy.p
xy['y_se']=(xy.y-xy.y_out).pow(2)*xy.p
print(xy.head())
print(xy.x_se.sum())
print(xy.y_se.sum())
print('hi')