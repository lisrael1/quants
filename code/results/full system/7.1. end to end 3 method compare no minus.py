#!/usr/bin/python3
import pandas as pd
import numpy as np
import itertools
from optparse import OptionParser
from threading import Thread
import pathos.multiprocessing as mp
import pylab as plt

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
    print("we have %0d rows for A" % len(all_rows))
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

def to_codebook(df, quantizer_size, number_of_bins=0):
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


def mse_from_naive_method(data, quant_size, number_of_bins):
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


def mse_from_basic_method(data, quant_size):
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


def mse_from_modulo_method(data, quant_size, number_of_bins):
    '''
    for modulo method
    example:
    :param data:
    :param quant_size:
    :param number_of_bins:
    :return:
    '''
    original = data.copy()
    # q = to_codebook(original, quant_size, 0)
    q = to_codebook(original, quant_size, number_of_bins)
    m1 = q % number_of_bins
    # m1 = sign_mod(q,number_of_bins)
    '''finding best rows by sampled std'''
    a_rows = get_all_a_rows(15)
    best_inx = m1.dot(a_rows.T).std().sort_values().head(2).index.values
    A = a_rows[best_inx].T
    print('A det is : %g'%np.linalg.det(A))

    r1=m1.copy().dot(A)
    r1.columns=['X', 'Y']
    r2=r1%number_of_bins
    # r2=sign_mod(r1,number_of_bins)
    # visualize_data(q)
    # visualize_data(m1)
    # visualize_data(r1)
    # visualize_data(r2)
    r3=r2.copy().dot(A.I)
    r3.columns=['X', 'Y']

    # r = (m1.dot(A) % number_of_bins).dot(A.I)
    # r.columns = ['X', 'Y']
    o = from_codebook(r3, quant_size, 0)
    # o = from_codebook(r, quant_size, number_of_bins)
    visualize_data_defore_after(original, o)
    return (o - original).values.flatten().var()


quant_size = 0.01
number_of_bins = 1001
cov = rand_cov_det_1()
data = random_data(cov, 1000)
mse = mse_from_modulo_method(data, quant_size, number_of_bins)
print('mse = %g' % mse)
print('uniform mse should be %g' % (quant_size ** 2 / 12))
exit()


def deladsf():
    df_original = random_data(cov, samples)
    df_mod1 = sign_mod(df_original, modulo_size_edge_to_edge)
    # df_quant = quantize(df_mod1, modulo_size_edge_to_edge, bins)
    results = pd.DataFrame()
    # got till here in adapting content to the new one
    for row in all_rows:
        df_dot_row = df_mod1.dot(row)
        df_mod2 = sign_mod(df_dot_row, modulo_size_edge_to_edge)
        A = np.mat([row.A1, row.A1]).T
        output_cov_by_single_row = A.T * cov * A
        sampled_std = float(df_mod2.values.std())
        results = results.append(dict(a=row[0].A1.tolist()[0],
                                      b=row[1].A1.tolist()[0],
                                      sampled_std=sampled_std,
                                      var1=output_cov_by_single_row[0, 0],
                                      var2=output_cov_by_single_row[1, 1],
                                      var_max=max(output_cov_by_single_row[0, 0], output_cov_by_single_row[1, 1])
                                      ), ignore_index=True)
    results.sort_values(by='sampled_std', inplace=True)  # we want the 2 first best sampled output
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
    if 0:  # and return_dict['cov_det_sqrt_sqrt']<0.5: # visualizing the input case
        vitualizing_cov_case(cov)
    return_dict['max_std'] = np.sqrt(
        results.head(2).var_max.max())  # note that we check if A is truly good or not, but we dont check if there is another good A
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
        df = pd.read_excel('5.1. threshold_by_sampled_std2.xlsx', header=[0, 1])
        print('done reading excel')
        # exit()
    else:

        run_num = 1
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

        num_of_cpu = mp.cpu_count()
        if 0:
            print('disabling parallel work!!!!!')
            num_of_cpu = 1
        print('number of cpus: %d' % num_of_cpu)
        p = mp.Pool(num_of_cpu)

        modulo_size_edge_to_edge = 8.8
        modulo_size_edge_to_edge = 10

        # print(threshold_for_N(0.1,5,1))
        # exit()
        if 1:
            u.samples = 10
            u.A_max_num = 10
            u.simulations = 100  # for best graphs, put 50k. for basic use 100 or 1k
            u.bins = 10000
            max_error = 100 / u.simulations
            max_error = 0.1
            u.threshold = threshold_for_N(max_error, u.samples, 1)  # if first number is 0.1 we expect 10% of the simulation to error on giving bad A
        print('the odds to get wrong with modulo of %g, edge to edge, is one to %g samples' % (
            modulo_size_edge_to_edge, 0.5 * (norm.cdf(-modulo_size_edge_to_edge / 2)) ** -1))

        if 0:
            all = []
            for i in tqdm(range(u.simulations)):
                # print('sim %d'%i)
                cov = rand_cov()
                outputs = run_all_rows_on_cov(cov, all_rows, u.threshold, u.bins, modulo_size_edge_to_edge, u.samples)
                all += [outputs]
        if 0:  # not working parallel...
            t, all = [], []
            [t.append(Thread(
                target=lambda: all.append(run_all_rows_on_cov(rand_cov(), all_rows, u.threshold, u.bins, modulo_size_edge_to_edge, u.samples)))) for i
                in range(u.simulations)]
            print('starting threads')
            [i.start() for i in t]
            print('joining threads')
            # [i.join() for i in t]
            [i.join() for i in tqdm(t)]
        if 1:
            all = p.map(lambda x: run_all_rows_on_cov(*x),
                        [[rand_cov(), all_rows, u.threshold, u.bins, modulo_size_edge_to_edge, u.samples] for i in range(u.simulations)])
        df = pd.DataFrame(all)  # .round(2)
        n = pd.DataFrame(['data'] * df.columns.size, index=df.columns.values).T
        n[['error', 'giving_bad_A', 'missing_good_A', 'system_detection', 'true_detection']] = 'error'
        df.columns = n.T.reset_index(drop=False).T.values.tolist()[::-1]
        df = df.T.sort_index().T
        df['error'] = df.error.astype(int, inplace=True)

        df['detection', 'good_A_detected'] = (df.error.system_detection == 1) & (df.error.true_detection == 1)
        df['detection', 'good_A_miss'] = (df.error.system_detection == 0) & (df.error.true_detection == 1)
        df['detection', 'bad_A_detected'] = (df.error.system_detection == 0) & (df.error.true_detection == 0)
        df['detection', 'bad_A_miss'] = (df.error.system_detection == 1) & (df.error.true_detection == 0)
        print(df.round(3))
        df.to_excel('resutls.xlsx')

    results = pd.DataFrame(columns=['detected', 'miss'], index=['good A', 'bad A'])
    # results.loc['good A','detected']=df.error.loc[(df.error.system_detection==1)&(df.error.true_detection==1)].system_detection.sum()
    results.loc['good A', 'detected'] = ((df.error.system_detection == 1) & (df.error.true_detection == 1)).sum()
    results.loc['good A', 'miss'] = ((df.error.system_detection == 0) & (df.error.true_detection == 1)).sum()
    results.loc['bad A', 'detected'] = ((df.error.system_detection == 0) & (df.error.true_detection == 0)).sum()
    results.loc['bad A', 'miss'] = ((df.error.system_detection == 1) & (df.error.true_detection == 0)).sum()
    results_per = results.copy().apply(lambda x: x / x.sum(), axis=1).round(2)

    print(results)
    print(results_per)
    if 1:
        # fig1 = pd.concat([df.data.cov_det_sqrt_sqrt, df.detection], axis=1).set_index('cov_det_sqrt_sqrt').astype(int).iplot(asFigure=True, kind='scatter', mode='markers')
        fig2 = df.data.cov_det_sqrt_sqrt.iplot(asFigure=True, kind='histogram', bins=50)
        data = pd.concat([df.data.cov_det_sqrt_sqrt, df.detection], axis=1).set_index('cov_det_sqrt_sqrt').astype(int).replace(0, np.nan)
        fig = [{'x': data[col].dropna().index.values, 'name': col, 'type': 'histogram'} for col in data.columns if not data[col].dropna().empty]
        py.offline.plot(fig, filename='c.html')
        # py.offline.plot(fig1, filename = 'a.html')
        py.offline.plot(fig2, filename='b.html')

    if 0:
        # fig=df.set_index('max_tail').sort_index()[['max_var','mse']].iplot(asFigure=True)
        df.columns = df.columns.droplevel(0)
        fig = df.set_index('max_tail').sort_index().iplot(asFigure=True, kind='histogram', xTitle='tail max value')
        py.offline.plot(fig)
        # df=df.reindex_axis([i for i in df.columns if i!='cov']+['cov'],axis=1)
        # df.to_excel("all results_n_%d_t_%g_m_%d.xlsx"%(u.samples,u.threshold,u.A_max_num))
