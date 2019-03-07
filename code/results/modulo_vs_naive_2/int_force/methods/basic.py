import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
import int_force


def basic_method(samples, quant_size, number_of_bins=False, A_rows=None, snr=1000, A=False):
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
    cov=int_force.rand_data.rand_data.rand_cov(snr=snr)
    original = int_force.rand_data.rand_data.random_data(cov, samples)
    q = int_force.methods.methods.to_codebook(original, quant_size, 0)
    o = int_force.methods.methods.from_codebook(q, quant_size, 0)
    res=dict(sampled_std_1=0,
             sampled_std_2=0,
             mse=(o - original).values.flatten().var(),
             error_per=((o-original).abs().values.flatten()>quant_size).astype(int).mean(),
             pearson=0,
             cov=str(cov.tolist()))
    return res


if __name__ == '__main__':
    samples=100
    number_of_bins=17
    quant_size=0.785235
    outputs=basic_method(samples=samples, number_of_bins=number_of_bins, quant_size=quant_size)
    print(outputs)