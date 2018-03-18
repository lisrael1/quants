#!/usr/bin/python3
import pandas as pd
from tqdm import tqdm
import numpy as np
import itertools
# import random
from optparse import OptionParser
import pylab as plt
import seaborn as sns
import cufflinks
import plotly as py
from scipy.stats import chi, chi2, norm
from threading import Thread
# from multiprocessing import Process, Pool
import pathos.multiprocessing as mp
# import time

'''

this script is: - TODO update this...
    generating all combinations of rows to A
    in loop:
        randomizing data det
        getting few samples from data
        for each row:
            giving score to that row by tails of the samples.
                i think just by worst one the biggest value
                the system flow is modulo->quantize->A->modulo
                i think we can skip the quantization...
        finding best rows to combine into A in condition that they are not dependend
        check by A*cov*A if this A is good
'''


def rand_cov():
    m = np.matrix(np.random.normal(0, 1, [2, 2]))
    m = m.T * m
    '''now changing the cov det. when m/=scale: for 0.5 you will have all bad A, for 10 almost all good A'''
    scale = np.random.uniform(.02, 1.5)
    # scale=10

    if 1:
        cov=np.sqrt(m[0,0]*m[1,1])*np.random.uniform(0.99,0.999999)
        cov=np.random.choice([-1,1])*cov
        m[0, 1]=cov
        m[1, 0]=m[0, 1]
        # if m[0,1]/np.sqrt(m[0,0]*m[1,1])<0.9:
        #     return rand_cov()
    m /= np.sqrt(np.linalg.det(m))
    m *= scale ** 2
    if 0:
        print(np.linalg.det(m))
    return np.mat(m)


def hist_plot(df, title):
    # sns.jointplot(data=df, x="X", y="Y",xlim=[-4.4,4.4],ylim=[-4.4,4.4])#,ax=axes.pop())#, kind="kde"
    sns.jointplot(data=df, x="X", y="Y")
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title)


def random_data(cov, samples):
    xy = pd.DataFrame(np.random.multivariate_normal([0, 0], cov, samples), columns=['X', 'Y'])
    return xy


def sign_mod(xy, modulo_size_edge_to_edge):
    xy = xy.copy()
    xy += modulo_size_edge_to_edge / 2.0
    xy = xy.mod(modulo_size_edge_to_edge) - modulo_size_edge_to_edge / 2.0
    # xy.columns=['X','Y']
    return xy


def quantize(xy, modulo_size_edge_to_edge, number_of_bins):
    if number_of_bins > 1e3:
        return xy
    hlf = modulo_size_edge_to_edge / 2.0
    bins = np.linspace(-hlf, hlf, number_of_bins + 1)
    center = (bins[:-1] + bins[1:]) / 2  # which is also (q[:-1]+(q[1]-q[0])/2)
    bins[0] = -float("inf")  # otherwise the values outside the bins will get NaN
    bins[-1] = float("inf")
    df = pd.DataFrame()
    df['X'] = pd.cut(xy.X, bins, labels=center).astype(float)
    df['Y'] = pd.cut(xy.Y, bins, labels=center).astype(float)
    return df


def threshold_for_N(max_error, samples, desired_std):  # TODO each threshold for X and Y should be seperate, otherwise we do mean of 2 std and we might have little and big std
    # the threshold for std for getting the sigma smaller than desired_std with the given max error
    # the error might be less than max_error, because if we set threshold and we never reach it, we will never get wrong...
    # even if we set initial det to be small, after multiply in A it might be large and we will reach the threshold
    threshold = chi(samples).ppf(max_error) * (desired_std) / np.sqrt(samples)
    print('for max error of %g, with desired std of %g and %d samples, threshold is %g' % (max_error, desired_std, samples, threshold))
    return threshold

def vitualizing_cov_case(cov):
    df_original_multi_samples = random_data(cov, 100000)
    df_original_multi_samples.plot.scatter(x='X', y='Y', alpha=0.01)
    m=df_original_multi_samples.max().max()
    plt.axis([-m,m,-m,m])
    plt.show()

def run_all_rows_on_cov(cov, all_rows, threshold, bins, modulo_size_edge_to_edge, samples):
    if 0:
        global run_num
        print('hi %d' % run_num)
        run_num += 1
    if 0:
        print('hi')
    df_original = random_data(cov, samples)
    df_mod1 = sign_mod(df_original, modulo_size_edge_to_edge)
    df_quant = quantize(df_mod1, modulo_size_edge_to_edge, bins)
    results = pd.DataFrame()
    for row in all_rows:
        df_dot_row = df_quant.dot(row)
        df_mod2 = sign_mod(df_dot_row, modulo_size_edge_to_edge)
        A = np.mat([row.A1, row.A1]).T
        output_cov_by_single_row = A.T * cov * A
        sampled_std = float(df_mod2.values.std())
        results = results.append(dict(a=row[0].A1.tolist()[0],
            b=row[1].A1.tolist()[0],
            sampled_std=sampled_std,
            var1=output_cov_by_single_row[0, 0],
            var2=output_cov_by_single_row[1, 1],
            var_max=max(output_cov_by_single_row[0, 0],output_cov_by_single_row[1, 1])
        ), ignore_index=True)
    results.sort_values(by='sampled_std', inplace=True) # we want the 2 first best sampled output
    # now need to remove all lines that a=0 except the first one, and the ones that are not starting with 0 divide them in a and remove duplicates
    # A=np.mat(results[['a','b']].sort_values(by='var_max').head(2).values.T)
    A = np.mat(results[['a', 'b']].head(2).values.T)
    output_cov = A.T * cov * A
    df_after_A = pd.DataFrame(np.mat(df_original) * A, columns=['X', 'Y'])
    df_after_second_mod = sign_mod(df_after_A, modulo_size_edge_to_edge)
    df_after_AI = pd.DataFrame(np.mat(df_after_second_mod) * (A.I), columns=['X', 'Y'])
    e = df_after_AI - df_original
    mse = (e ** 2).sum().sum() / e.size
    return_dict = {}
    return_dict['cov_det_sqrt_sqrt'] = np.sqrt(np.sqrt(np.linalg.det(cov)))
    if 0:# and return_dict['cov_det_sqrt_sqrt']<0.5: # visualizing the input case
        vitualizing_cov_case(cov)
    return_dict['max_std'] = np.sqrt(results.head(2).var_max.max()) # note that we check if A is truly good or not, but we dont check if there is another good A
    return_dict['sampled_std'] = results.head(2).sampled_std.max()
    return_dict['smse'] = np.sqrt(mse)
    return_dict['max_original_std'] = np.sqrt(max(cov[0, 0], cov[1, 1]))
    return_dict['original_abs_pearson'] = np.abs(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    return_dict['output_abs_pearson'] = np.abs(output_cov[0, 1] / np.sqrt(output_cov[0, 0] * output_cov[1, 1]))
    return_dict['system_detection'] = 1 if return_dict['sampled_std'] < threshold else 0
    return_dict['true_detection'] = 1 if return_dict['max_std'] < modulo_size_edge_to_edge * 0.12850167052 else 0
    return_dict['true_detection'] = 1 if return_dict['max_std'] < 1 else 0
    return_dict['error'] = np.bitwise_xor(return_dict['system_detection'], return_dict['true_detection'])
    return_dict['giving_bad_A'] = np.bitwise_and(return_dict['system_detection'], return_dict['error'])
    return_dict['detecting_bad_A'] = np.bitwise_and(not return_dict['system_detection'], not return_dict['true_detection'])
    return_dict['missing_good_A'] = np.bitwise_and(return_dict['true_detection'], return_dict['error'])
    if 0:
        print('done')
    return return_dict

if __name__ == '__main__':
    pd.set_option('expand_frame_repr', False)
    if 0:
        df=pd.read_excel('5.1. threshold_by_sampled_std2.xlsx',header=[0,1])
        print('done reading excel')
        # exit()
    else:

        run_num=1
        parser = OptionParser()
        parser.add_option("-n","", dest="samples", type="int", default=10,help='number of dots X2 because you have x and y. for example 1000. you better use 5')
        parser.add_option("-s","", dest="simulations", type="int", default=10,help='number of simulations, for example 50. you better use 400')
        parser.add_option("-b","", dest="bins", type="int", default=200,help='number of bins, for example 200')
        parser.add_option("-t","", dest="threshold", type="float", default=2.5,help='this threshold is for deciding if data is U or N, by checking if there are samples over the threshold. this defines the tail size. bigger number will result of more detecting output as N')
        parser.add_option("-m","", dest="A_max_num", type="int", default=2,help='A max number for example for 2 you can get [[-2,1],[2,0]]. for number 10, you will get 189,776 options at A. at 5 you will have 13608. . you better use 10')
        parser.add_option("-o",    dest="different_cov", help="if set, using same cov matrix for all simulations", default=False, action="store_true")
        (u,args)=parser.parse_args()

        num_of_cpu = mp.cpu_count()
        if 0:
            print('disabling parallel work!!!!!')
            num_of_cpu=1
        print('number of cpus: %d' % num_of_cpu)
        p = mp.Pool(num_of_cpu)

        modulo_size_edge_to_edge = 8.8
        modulo_size_edge_to_edge = 10

        # print(threshold_for_N(0.1,5,1))
        # exit()
        if 1:
            u.samples = 10
            u.A_max_num = 10
            u.simulations = 100 # for best graphs, put 50k. for basic use 100 or 1k
            u.bins = 10000
            max_error=100/u.simulations
            max_error=0.1
            u.threshold=threshold_for_N(max_error,u.samples,1) # if first number is 0.1 we expect 10% of the simulation to error on giving bad A
        print('the odds to get wrong with modulo of %g, edge to edge, is one to %g samples' % (modulo_size_edge_to_edge, 0.5 * (norm.cdf(-modulo_size_edge_to_edge / 2)) ** -1))

        comb=list(range(-u.A_max_num,u.A_max_num+1))
        comb.remove(0) # we dont want 0,0. we want only 0,1 not 0,-1 not 0,2 etc.
        comb=[comb,comb]
        all_rows=list(itertools.product(*comb))
        all_rows+=[(0,1)]
        all_rows=pd.DataFrame(all_rows,columns=['a','b'])
        all_rows['normal_b']=all_rows.b/all_rows.a
        all_rows['abs_a']=np.abs(all_rows.a)
        all_rows.sort_values(by='abs_a',inplace=True)
        all_rows.drop_duplicates(subset='normal_b',keep='first',inplace=True)
        all_rows=[np.mat(i).reshape(2,1) for i in all_rows[['a','b']].values]
        print("we have %0d rows"%len(all_rows))

        if 0:
            all=[]
            for i in tqdm(range(u.simulations)):
                # print('sim %d'%i)
                cov = rand_cov()
                outputs=run_all_rows_on_cov(cov,all_rows,u.threshold,u.bins,modulo_size_edge_to_edge,u.samples)
                all+=[outputs]
        if 0: # not working parallel...
            t,all=[],[]
            [t.append(Thread(target=lambda :all.append(run_all_rows_on_cov(rand_cov(),all_rows,u.threshold,u.bins,modulo_size_edge_to_edge,u.samples)))) for i in range(u.simulations)]
            print('starting threads')
            [i.start() for i in t]
            print('joining threads')
            # [i.join() for i in t]
            [i.join() for i in tqdm(t)]
        if 1:
            all=p.map(lambda x:run_all_rows_on_cov(*x),[[rand_cov(),all_rows,u.threshold,u.bins,modulo_size_edge_to_edge,u.samples] for i in range(u.simulations)])
        df=pd.DataFrame(all)#.round(2)
        n=pd.DataFrame(['data']*df.columns.size,index=df.columns.values).T
        n[['error','giving_bad_A','missing_good_A','system_detection','true_detection']]='error'
        df.columns=n.T.reset_index(drop=False).T.values.tolist()[::-1]
        df=df.T.sort_index().T
        df['error']=df.error.astype(int,inplace=True)

        df['detection','good_A_detected']=(df.error.system_detection==1)&(df.error.true_detection==1)
        df['detection','good_A_miss']=(df.error.system_detection==0)&(df.error.true_detection==1)
        df['detection','bad_A_detected']=(df.error.system_detection==0)&(df.error.true_detection==0)
        df['detection','bad_A_miss']=(df.error.system_detection==1)&(df.error.true_detection==0)
        print(df.round(3))
        df.to_excel('resutls.xlsx')

    results=pd.DataFrame(columns=['detected','miss'],index=['good A','bad A'])
    # results.loc['good A','detected']=df.error.loc[(df.error.system_detection==1)&(df.error.true_detection==1)].system_detection.sum()
    results.loc['good A','detected']=((df.error.system_detection==1)&(df.error.true_detection==1)).sum()
    results.loc['good A','miss']=((df.error.system_detection==0)&(df.error.true_detection==1)).sum()
    results.loc['bad A','detected']=((df.error.system_detection==0)&(df.error.true_detection==0)).sum()
    results.loc['bad A','miss']=((df.error.system_detection==1)&(df.error.true_detection==0)).sum()
    results_per=results.copy().apply(lambda x:x/x.sum(),axis=1).round(2)

    print(results)
    print(results_per)
    if 1:
        # fig1 = pd.concat([df.data.cov_det_sqrt_sqrt, df.detection], axis=1).set_index('cov_det_sqrt_sqrt').astype(int).iplot(asFigure=True, kind='scatter', mode='markers')
        fig2 = df.data.cov_det_sqrt_sqrt.iplot(asFigure=True, kind='histogram', bins=50)
        data = pd.concat([df.data.cov_det_sqrt_sqrt, df.detection], axis=1).set_index('cov_det_sqrt_sqrt').astype(int).replace(0, np.nan)
        fig = [{'x': data[col].dropna().index.values, 'name': col, 'type': 'histogram'} for col in data.columns if not data[col].dropna().empty]
        py.offline.plot(fig , filename = 'c.html')
        # py.offline.plot(fig1, filename = 'a.html')
        py.offline.plot(fig2, filename = 'b.html')

    if 0:
        # fig=df.set_index('max_tail').sort_index()[['max_var','mse']].iplot(asFigure=True)
        df.columns = df.columns.droplevel(0)
        fig=df.set_index('max_tail').sort_index().iplot(asFigure=True,kind='histogram',xTitle='tail max value')
        py.offline.plot(fig)
        # df=df.reindex_axis([i for i in df.columns if i!='cov']+['cov'],axis=1)
        # df.to_excel("all results_n_%d_t_%g_m_%d.xlsx"%(u.samples,u.threshold,u.A_max_num))