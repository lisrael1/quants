import numpy as np
import pandas as pd


def rand_cov(dimensions=2, snr=1000, A=None):
    '''
    :param dimensions:
    :param snr:
    :param A: if you enter matrix here, it will find the match cov matrix for this A
    :return:
    '''
    if type(A)==type(None):
        cov = np.matrix(np.random.normal(0, 1, [1, dimensions]))
        cov = cov.T * cov
        cov+=np.matrix(np.diag([1/snr]*dimensions))
    else:
        # A=np.mat([[1,0],[-2,1]])
        A=np.mat(A)
        cov=A.T.I*A.I
    return cov


def random_data(cov, samples):
    xy = pd.DataFrame(np.random.multivariate_normal([0, 0], cov, samples), columns=['X', 'Y'])
    return xy