import numpy as np
import pandas as pd


if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    import int_force

    A = np.mat([[1, 0], [-2, 1]])
    A=None
    cov = int_force.rand_data.rand_data.rand_cov(A=A, snr=1)
    A2 = np.mat([[0, -1], [1, 1]])

    samples, number_of_bins, quant_size = 1000, 17, 0.785235
    int_force.methods.modulo.visualize_input_output(cov, samples, number_of_bins, quant_size, A, A2=A2, plot_place='output_data')

