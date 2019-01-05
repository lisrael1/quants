import numpy as np
import pandas as pd


if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    import int_force
    # from int_force.rand_data import rand_data

    A=np.mat([[1,0],[-2,1]])
    A=np.mat([[2,0],[-4,2]])
    cov=int_force.rand_data.rand_data.rand_cov(A=A)
    data=int_force.rand_data.rand_data.random_data(cov,100)
    # cov=rand_data.rand_cov()
    print(cov)
    samples=1000
    quant_size=0.1
    number_of_bins=17
    # int_force.methods.modulo.modulo_method(samples, quant_size, number_of_bins, std_threshold=None, A_rows=None, snr=1000)
    a_rows=int_force.A_mat.A_rows.get_all_a_rows(10)
    A=int_force.A_mat.A_rows.find_best_A(a_rows, data, True)
    print(A)