def vector_to_angle(x,y):
    import numpy as np
    return np.angle(complex(x, y), deg=True)


def get_cov_ev(cov, plot=False):
    import numpy as np
    import pandas as pd
    cov = np.mat(cov)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    df = pd.DataFrame()
    df['ev_x'] = eig_vecs[0, :].tolist()[0]
    df['ev_y'] = eig_vecs[1, :].tolist()[0]
    df['e_values'] = eig_vals
    df['e_values_sqrt'] = df.e_values.apply(np.sqrt)
    df.fillna(0, inplace=True)
    df = df.sort_values(by='e_values', ascending=False).reset_index()
    main_axis=df.ev_x[0] * df.e_values_sqrt[0], df.ev_y[0] * df.e_values_sqrt[0]
    angle = vector_to_angle(main_axis[0], main_axis[1])  # from -135 to 45 degrees
    angle = angle if angle >= -90 else 180+angle  # to make it from -90 to 90

    if plot:
        print('cov\n\t' + str(cov).replace('\n', '\n\t'))
        import pylab as plt
        samples = 100
        df_original_multi_samples = pd.DataFrame(np.random.multivariate_normal([0, 0], cov, samples), columns=['X', 'Y'])
        df_original_multi_samples.plot.scatter(x='X', y='Y', alpha=0.1)
        m = df_original_multi_samples.max().max()
        plt.axis([-m, m, -m, m])

        for i in [0, 1]:  # 0 is the main eigenvector
            if df.ev_x[i] * df.e_values_sqrt[i] + df.ev_y[i] * df.e_values_sqrt[i]:  # cannot plot 0 size vector
                plt.arrow(0, 0,
                          df.ev_x[i] * df.e_values_sqrt[i], df.ev_y[i] * df.e_values_sqrt[i],
                          head_width=0.15, head_length=0.15,
                          length_includes_head=True, fc='k', ec='k')
            else:
                print('zero length eigenvector, skipping')
        plt.show()
    return df, angle


if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    sys.path.append(r'C:\Users\lisrael1\Documents\myThings\lior_docs\HU\thesis\quants\code\results\modulo_vs_naive_2/')
    import int_force

    import numpy as np
    import pandas as pd
    import plotly as py
    import cufflinks
    # from sklearn.cluster import DBSCAN
    # import pylab as plt
    # from skimage.transform import radon, rescale
    # import warnings

    df=pd.DataFrame()
    for i in range(100):
        print('*'*150)
        samples=1000

        cov=np.mat([[0, 0],[0, 1]])
        cov=np.mat([[1, 1],[1, 1.2]])
        cov=np.mat([[0.53749846, 0.35644121],[0.35644121, 0.23651739]])
        cov=int_force.rand_data.rand_data.rand_cov(snr=10000)
        data = int_force.rand_data.rand_data.random_data(cov, samples)

        hist_bins=300
        sinogram_dict = int_force.methods.ml_modulo.calc_sinogram(data.X.values, data.Y.values, bins=hist_bins)

        # print(cov)
        get_cov_ev(cov, False)
        df=df.append(pd.Series(dict(sinogram=sinogram_dict['angle_by_std'], ev=get_cov_ev(cov, False)[1])), ignore_index=True)
        df['error']=df.ev-df.sinogram
        if df['error'].abs().values[-1]>5:
            print('hi')
    print(df[df.error.abs()>5])
    print(df.describe())

