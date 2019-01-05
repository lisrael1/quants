import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
import int_force


def clipping_method(samples, quant_size, number_of_bins, std_threshold=None, A=None, snr=1000): # TODO maybe we can return nan when we have more than given mse instead of std_threadold. or maybe the TX always transmite so the std checker is after the cutting
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
    cov = int_force.rand_data.rand_data.rand_cov(snr=snr, A=A)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    original = int_force.rand_data.rand_data.random_data(cov, samples)  # data.copy()

    q = int_force.methods.methods.to_codebook(original, quant_size, 0)

    '''now cutting the edges:'''
    q=np.clip(q,np.ceil(-number_of_bins/2),np.floor(number_of_bins/2)).astype(int)
    q_std=q.std()
    if std_threshold and (q_std>std_threshold).astype(int).sum()>0:
        return 'deprecated'
        return float(q.std()),original.values.flatten().var()
    o = int_force.methods.methods.from_codebook(q, quant_size, 0)
    res=dict(rmse=np.sqrt((o - original).values.flatten().var()),
             error_per=((o-original).abs().values.flatten()>quant_size).astype(int).mean(),
             pearson=pearson,
             cov=str(cov.tolist()))
    return res


if __name__ == '__main__':
    A=np.mat([[1,0],[-2,1]])
    cov = int_force.rand_data.rand_data.rand_cov(A=A)
    samples=100
    number_of_bins=17
    quant_size=0.785235
    visualize_input_output(cov, samples, number_of_bins, quant_size, A)