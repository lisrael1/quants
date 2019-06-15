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


def stack_specific_columns(df, list_of_columns_to_keep):
    df=df.copy()
    letters=df.drop(list_of_columns_to_keep, axis=1).columns.unique().shape[0]
    df.set_index(list_of_columns_to_keep, inplace=True)
    df = df.stack().to_frame()
    # we cannot do unstack because we have the same index names at the same sub index.
    # so we do a trick - adding dummy index:
    df['dummy_idx'] = sorted(list(range(int(df.shape[0] / letters))) * letters)
    df.set_index(['dummy_idx'], inplace=True, append=True)
    df.index.names = list_of_columns_to_keep+['let', 'dummy_idx']
    df = df.unstack(level='let')

    # now removing all multi columns and indexes:
    df.columns = df.columns.get_level_values('let').values
    df.index = df.index.droplevel('dummy_idx')
    df.reset_index(inplace=True)

    return df


def all_data_origin_options(data_db, modulo_size, number_of_shift_per_direction, debug=False):
    '''

    :param data: df with x and y columns
    :param modulo_size:
    :param number_of_shift_per_direction: for example, 1 is adding 1 time the modulo to the right and 1 times to the left,
            and the same with up and down, so we get total of 9 time the modulo.
            so each point duplicated 9 times, and we need to find the best option from all those 9
    :return:
    '''
    import itertools
    # modulo_edges=np.arange(-multiple_x, multiple_x)*modulo_size+mod_size/2  # max number should be multiple_x*modulo_size-modulo_size/2 because the middle modulo is half left to 0 and half right to 0
    modulo_shifts = np.arange(-number_of_shift_per_direction, number_of_shift_per_direction + 1) * modulo_size
    modulo_shifts = pd.DataFrame(list(itertools.product(*[modulo_shifts] * 2)), columns='modulo_center_x,modulo_center_y'.split(','))

    a = modulo_shifts.stack()
    a.index = a.index.to_frame().iloc[:, 1].values
    if debug and 0:
        import pylab as plt
        plt.figure()
        data_db.set_index('X').Y.plot(style='.', title='bla')
    a = data_db.join(a.to_frame().T).ffill()
    a = stack_specific_columns(a, list('XY'))
    a['out_x'] = a.X + a.modulo_center_x
    a['out_y'] = a.Y + a.modulo_center_y
    a=a.rename(columns=dict(X='x_at_mod', Y='y_at_mod', out_x='X', out_y='Y'))

    return a
