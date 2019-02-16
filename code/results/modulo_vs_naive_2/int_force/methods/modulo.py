import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
import int_force


def modulo_flow(input_data, quant_size, number_of_bins, A=None):
    q = int_force.methods.methods.to_codebook(input_data, quant_size, 0)
    # q = to_codebook(original, quant_size, number_of_bins)
    # m1 = q % number_of_bins
    m1 = int_force.methods.methods.sign_mod(q, number_of_bins).astype(int)
    if type(A) == type(None):
        '''finding best rows by sampled std'''
        if 0:
            A_rows=int_force.A_mat.A_rows.all_A_rows(10)
            A = int_force.A_mat.A_rows.find_best_A(m1, A_rows)
        else:
            A = int_force.A_mat.A_rows.find_best_A(m1)
    # print('A det is : %g'%np.linalg.det(A))

    r1 = m1.copy().dot(A)
    r1.columns = ['X', 'Y']
    # r2=r1%number_of_bins
    r2 = int_force.methods.methods.sign_mod(r1, number_of_bins)
    # visualize_data(q)
    r3 = r2.copy().dot(A.I)
    r3.columns = ['X', 'Y']

    output_data = int_force.methods.methods.from_codebook(r3, quant_size, 0)
    return dict(output_data=output_data, A=A, after_A=r1, after_A_after_mod=r2, after_A_inv=r3)


def modulo_method(samples, quant_size, number_of_bins, A=None, snr=1000, dont_look_for_A=True):
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
    cov=int_force.rand_data.rand_data.rand_cov(snr=snr, A=A)
    pearson=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1])
    input_data = int_force.rand_data.rand_data.random_data(cov,samples) #data.copy()
    if dont_look_for_A:
        given_A=None
    else:
        given_A=A
    outs=modulo_flow(input_data, quant_size, number_of_bins, A=given_A)
    o=outs['output_data']
    A=outs['A']
    # visualize_data_defore_after(original, o)
    res=dict(rmse=(o - input_data).pow(2).values.mean() ** 0.5,
             error_per=((o-input_data).abs().values.flatten()>quant_size).astype(int).mean(),
             pearson=pearson,
             A=A,
             cov = str(cov.tolist()))
    return res


def visualize_input_output(cov, samples, number_of_bins, quant_size, A=None, A2=None, plot_place='output_data'):
    '''

    :param cov:
    :param samples:
    :param number_of_bins:
    :param quant_size:
    :param A:  if you already have A, put it here, if not, use A_rows
    :param A2: if you want another output by this second A, to compare between those 2 A
    :param plot_place: can be: 'output_data', 'after_A', 'after_A_after_mod','after_A_inv'
    :return:
    '''
    original = int_force.rand_data.rand_data.random_data(cov, samples)
    q = int_force.methods.methods.to_codebook(original, quant_size, 0)
    m1 = int_force.methods.methods.sign_mod(q, number_of_bins).astype(int)
    '''finding best rows by sampled std'''
    if type(A)==type(None):
        A_rows=int_force.A_mat.A_rows.all_A_rows(10)
        A = int_force.A_mat.A_rows.find_best_A(inputs=m1, a_rows=A_rows)
    output=modulo_flow(original, quant_size, number_of_bins, A=A)[plot_place]

    plots=original.copy()
    plots['place']='inputs'
    plots = pd.concat([plots, output.copy()], sort=True).fillna('outputs')
    title='rmse = {:.4f}'.format((output-original).pow(2).values.mean()**0.5)

    if type(A2)!=type(None):
        output2 = modulo_flow(original, quant_size, number_of_bins, A=A2)[plot_place]
        plots = pd.concat([plots, output2.copy()], sort=True).fillna('outputs2')
        title+=', rmse on 2 input A = {:.4f}'.format((output - output2).pow(2).values.mean() ** 0.5)

    '''setting max value as zoom place'''
    import itertools
    max_num=plots[['X', 'Y']].abs().max().max()
    max_nums = list(itertools.product(*[[max_num, -max_num]] * 2))
    max_nums=pd.DataFrame(max_nums, columns=['X', 'Y'])
    max_nums['place']='edges'
    plots=plots.append(max_nums).reset_index()
    fig = plots.figure(kind='scatter', x='X', y='Y', categories='place', size=7, opacity=0.2, title=title)
    py.offline.plot(fig)

    print('done')


if __name__ == '__main__':
    A=np.mat([[1,0],[-2,1]])
    cov = int_force.rand_data.rand_data.rand_cov(A=A)
    A2=np.mat([[0,-1],[1,1]])

    samples, number_of_bins, quant_size=1000, 17, 0.785235
    visualize_input_output(cov, samples, number_of_bins, quant_size, A, A2=A2, plot_place='output_data')
